import os
import gzip
import json
import numpy as np
import torch
import onnx

from omegaconf import OmegaConf
from experiment import nLLsExperiment
from misc import get_device


def load_model_gz(path, device):
    with gzip.open(path, "rb") as f:
        return torch.load(f, map_location=device)


def main(run_dir, rafal_onnx_path, out_onnx=None, run_idx=0):
    device = get_device()
    
    # Set default output path in the run's models folder
    if out_onnx is None:
        out_onnx = os.path.join(run_dir, "models", "model_with_metadata.onnx")
    
    # ------------------------------------------------------------------
    # 1. Load config from existing run
    # ------------------------------------------------------------------
    cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")
    if not os.path.exists(cfg_path):
        # Retraining runs start at index 1+; find the lowest available config
        candidates = sorted(
            int(f[len("config_"):-len(".yaml")])
            for f in os.listdir(run_dir)
            if f.startswith("config_") and f.endswith(".yaml")
        )
        assert candidates, f"No config_*.yaml found in {run_dir}"
        run_idx = candidates[0]
        cfg_path = os.path.join(run_dir, f"config_{run_idx}.yaml")
        print(f"[INFO] config_0.yaml not found; using config_{run_idx}.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # ------------------------------------------------------------------
    # 2. Recreate experiment JUST enough to get preprocessing + model
    # ------------------------------------------------------------------
    exp = nLLsExperiment(cfg, device)
    # minimal attributes normally set by BaseExperiment._init()
    exp.warm_start = False
    exp.dtype = torch.float32
    exp.init_physics()
    exp.init_data()
    exp.init_model()
    
    # ------------------------------------------------------------------
    # 3. Load trained weights
    # ------------------------------------------------------------------
    model_path_gz = os.path.join(run_dir, "models", f"model_run{run_idx}.pt.gz")
    model_path_pt = os.path.join(run_dir, "models", f"model_run{run_idx}.pt")
    if os.path.exists(model_path_gz):
        model_path = model_path_gz
        checkpoint = load_model_gz(model_path, device)
    elif os.path.exists(model_path_pt):
        model_path = model_path_pt
        checkpoint = torch.load(model_path, map_location=device)
    else:
        raise AssertionError(f"Missing {model_path_gz} and {model_path_pt}")
    state_dict = checkpoint["model"]
    exp.model.load_state_dict(state_dict)
    exp.model.eval()
    
    # ------------------------------------------------------------------
    # 4. Export ONNX
    # ------------------------------------------------------------------
    batch_size = 1
    global_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    dummy_features = torch.zeros(
        1,
        cfg.model.net.n_features,
        device=device,
    )
    dummy_global_token = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )
    
    torch.onnx.export(
        exp.model,
        (dummy_features, dummy_global_token),
        "_tmp.onnx",
        opset_version=17,
        input_names=["features", "global_token"],
        output_names=["nLLs"],
        dynamic_axes={
            "features": {0: "batch"},
            "global_token": {0: "batch"},
            "nLLs": {0: "batch"},
        },
    )
    
    # ------------------------------------------------------------------
    # 5. Read Rafal's ONNX metadata
    # ------------------------------------------------------------------
    rafal_onnx = onnx.load(rafal_onnx_path)
    rafal_metadata = {p.key: p.value for p in rafal_onnx.metadata_props}

    # Print Rafal's ONNX size
    rafal_size_mb = os.path.getsize(rafal_onnx_path) / (1024 * 1024)
    print(f"Rafal's ONNX size: {rafal_size_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 5b. Compute x/y min/max from training split
    # ------------------------------------------------------------------
    train_frac = cfg.data.train_test_val[0]
    subsample = cfg.data.get("subsample")
    features_raw = exp.features[0]   # raw (unpreprocessed) features
    nLLs_raw = exp.nLLs[0]           # raw delta-nLLs (4 differences, unpreprocessed)
    N = features_raw.shape[0]
    n_train = int(N * train_frac)
    if subsample is not None:
        n_train = min(int(subsample), n_train)
    x_train = features_raw[:n_train]
    y_train = nLLs_raw[:n_train]
    x_min = x_train.min(axis=0).tolist()
    x_max = x_train.max(axis=0).tolist()
    y_min = y_train.min(axis=0).tolist()
    y_max = y_train.max(axis=0).tolist()

    # ------------------------------------------------------------------
    # 6. Attach metadata to new ONNX
    # ------------------------------------------------------------------
    new_onnx = onnx.load("_tmp.onnx")

    # Keys from Rafal's metadata that we drop or replace
    KEYS_TO_DROP = {"starting_points", "standardization_mean", "standardization_std"}
    KEYS_TO_REPLACE = {"x_min", "x_max", "y_min", "y_max"}

    # Rafal's metadata — strip prefix, skip dropped/replaced keys
    for k, v in rafal_metadata.items():
        clean_key = k[len("rafal::"):] if k.startswith("rafal::") else k
        if clean_key in KEYS_TO_DROP or clean_key in KEYS_TO_REPLACE:
            continue
        p = new_onnx.metadata_props.add()
        p.key = clean_key
        p.value = v

    # Computed min/max from Joaquin's training data
    for key, val in [("x_min", x_min), ("x_max", x_max), ("y_min", y_min), ("y_max", y_max)]:
        p = new_onnx.metadata_props.add()
        p.key = key
        p.value = json.dumps(val)

    # Run config
    p = new_onnx.metadata_props.add()
    p.key = "run_config"
    p.value = OmegaConf.to_yaml(cfg)

    # Standardization stats
    stats = {
        "nLLs_mean": [m.tolist() for m in exp.prepd_mean],
        "nLLs_std": [s.tolist() for s in exp.prepd_std],
        "features_mean": [m.tolist() for m in exp.prepd_mean_features],
        "features_std": [s.tolist() for s in exp.prepd_std_features],
    }
    bounds = exp.prepd_nll_bounds[0] if exp.prepd_nll_bounds else {}
    if "lo" in bounds:  # logit_bounded only; per-output spec carries no lo/hi
        stats["nLLs_lo"] = np.asarray(bounds["lo"]).tolist()
        stats["nLLs_hi"] = np.asarray(bounds["hi"]).tolist()
    p = new_onnx.metadata_props.add()
    p.key = "standardization"
    p.value = json.dumps(stats)

    # Preprocessing pipeline description (derived from the actual training config)
    trafos = OmegaConf.to_container(cfg.data.get("trafos") or {}, resolve=True)
    feat_pipeline = []
    for trafo_fns in trafos.values():
        if isinstance(trafo_fns, list):
            feat_pipeline.extend(trafo_fns)
    # keep the container as-is: a flat list, or the per-output mapping
    # {"per_output": [...], "asinh_scale": s} (list(dict) would drop to keys)
    nll_pipeline = OmegaConf.to_container(cfg.data.get("nLL_trafos") or [], resolve=True)

    p = new_onnx.metadata_props.add()
    p.key = "preprocessing"
    p.value = json.dumps({
        "features_pipeline": feat_pipeline,
        "nLLs_pipeline": nll_pipeline,
        "note": (
            "log_w_negatives = log(|x|+1)*sign(x). "
            "Standardization parameters (mean, std per feature/output) "
            "are stored in the 'standardization' key."
        ),
    })
    
    onnx.save(new_onnx, out_onnx)
    os.remove("_tmp.onnx")
    
    # Print final model size
    final_size_mb = os.path.getsize(out_onnx) / (1024 * 1024)
    print(f"Final ONNX size: {final_size_mb:.2f} MB")
    print(f"Saved to: {out_onnx}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python export_onnx_from_run.py <run_dir> <rafal_onnx> [run_idx]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], run_idx=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
