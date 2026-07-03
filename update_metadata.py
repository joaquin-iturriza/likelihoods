"""
Update metadata in all ONNX models in models_onnx/:
  1. Strip the temporary 'rafal::' prefix from all copied keys
  2. Drop starting_points, standardization_mean/std (removed entirely)
  3. Drop operational/pipeline fields irrelevant for publication
  4. Fix model_author, model_name, model_parameters, training_date, training_duration
  5. Recompute x_min/x_max/y_min/y_max from Joaquin's training split (mmap, fast)
  6. Add 'preprocessing' key describing the transform pipeline
"""

import os
import sys
import json
import datetime
import numpy as np
import onnx
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")

MODEL_AUTHOR = "Joaquin Iturriaga"

# Keys copied from Rafal's metadata that are dropped entirely:
#   - starting_points: large scan artifact, unnecessary in a published model
#   - standardization_mean/std: Rafal's simple linear standardization, not what we do
#   - folder_name, input_folder, output_folder: Rafal's pipeline paths, not meaningful to users
#   - buffer_size, keep_files: Rafal's MCMC implementation details
#   - bkg_unc_samples, low_lim_samples: Rafal's sampling parameters
#   - processes: number of parallel processes Rafal used
#   - spey_verbose_lvl: verbosity flag for Rafal's likelihood tool
#   - "start method" (with space): typo duplicate of start_method
KEYS_TO_DROP = {
    "starting_points",
    "standardization_mean",
    "standardization_std",
    "folder_name",
    "input_folder",
    "output_folder",
    "buffer_size",
    "keep_files",
    "bkg_unc_samples",
    "low_lim_samples",
    "processes",
    "spey_verbose_lvl",
    "start method",          # typo version with space — kept as start_method below
}

# These are replaced with values derived from our training run
KEYS_TO_REPLACE = {"x_min", "x_max", "y_min", "y_max",
                   "model_author", "model_name", "model_parameters",
                   "training_date", "training_duration"}

PREPROCESSING_META = json.dumps({
    "features_pipeline": ["log_w_negatives", "standardization"],
    "nLLs_pipeline": ["log_w_negatives", "standardization"],
    "note": (
        "log_w_negatives = log(|x|+1)*sign(x). "
        "Standardization parameters (mean, std per feature/output) "
        "are stored in the 'standardization' key."
    ),
})


def compute_training_minmax(cfg: dict) -> tuple[list, list, list, list]:
    """
    Replicate experiment.py data loading + split to get per-feature and per-output
    min/max over the training portion of the dataset.

    Uses mmap_mode + permutation to avoid loading the full array into RAM:
    np.random.permutation with seed 1234 produces the same row ordering as
    np.random.seed(1234); np.random.shuffle(data), so we can index directly.
    """
    dataset = cfg["data"]["dataset"]
    if isinstance(dataset, list):
        dataset = dataset[0]

    data_path = cfg["data"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    train_frac = cfg["data"]["train_test_val"][0]
    subsample = cfg["data"].get("subsample")

    npy_path = os.path.join(data_path, f"{dataset}.npy")
    val_path = os.path.join(data_path, f"{dataset}_val.npy")
    test_path = os.path.join(data_path, f"{dataset}_test.npy")

    if os.path.exists(val_path):
        # Concatenated split files — load them individually with mmap then concat indices
        train_raw = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        val_raw   = np.load(val_path, mmap_mode="r", allow_pickle=True)
        test_raw  = np.load(test_path, mmap_mode="r", allow_pickle=True)
        N_total = len(train_raw) + len(val_raw) + len(test_raw)
        np.random.seed(1234)
        perm = np.random.permutation(N_total)
        n_train = int(N_total * train_frac)
        if subsample is not None:
            n_train = min(int(subsample), n_train)
        train_idx = perm[:n_train]
        # Build the concatenated array only for the training rows we need
        # (split across the three files by offset)
        n0, n1 = len(train_raw), len(val_raw)
        rows = []
        for i in sorted(train_idx):
            if i < n0:
                rows.append(train_raw[i])
            elif i < n0 + n1:
                rows.append(val_raw[i - n0])
            else:
                rows.append(test_raw[i - n0 - n1])
        data_train = np.stack(rows)
    else:
        data_mmap = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        N_total = len(data_mmap)
        np.random.seed(1234)
        perm = np.random.permutation(N_total)
        n_train = int(N_total * train_frac)
        if subsample is not None:
            n_train = min(int(subsample), n_train)
        train_idx = np.sort(perm[:n_train])   # sort for sequential disk reads
        data_train = data_mmap[train_idx]

    features = data_train[:, :-8]
    nLLs_raw = data_train[:, -8:].astype(float)
    for i in range(4):
        nLLs_raw[:, 2 * i + 1] -= nLLs_raw[:, 2 * i]
    nLLs = nLLs_raw[:, 1::2]   # 4 delta-nLL columns

    return (
        features.min(axis=0).tolist(),
        features.max(axis=0).tolist(),
        nLLs.min(axis=0).tolist(),
        nLLs.max(axis=0).tolist(),
    )


def extract_model_info(cfg: dict) -> dict:
    """Derive model_name, model_parameters, training_date, training_duration from run_config."""
    net_cfg      = cfg.get("model", {}).get("net", {})
    training_cfg = cfg.get("training", {})

    arch = net_cfg.get("_target_", "").rsplit(".", 1)[-1]   # e.g. "MuMLP"
    hc   = net_cfg.get("hidden_channels", "?")
    hl   = net_cfg.get("hidden_layers", "?")
    act  = net_cfg.get("activation", "?")
    model_name = f"{arch}_c{hc}_l{hl}_{act}"

    model_parameters = json.dumps({
        "architecture":          arch,
        "hidden_channels":       hc,
        "hidden_layers":         hl,
        "activation":            act,
        "out_shape":             net_cfg.get("out_shape"),
        "loss":                  training_cfg.get("loss"),
        "optimizer":             training_cfg.get("optimizer"),
        "lr":                    training_cfg.get("lr"),
        "regularization":        training_cfg.get("regularization"),
        "regularization_lambda": training_cfg.get("regularization_lambda"),
        "iterations":            training_cfg.get("iterations"),
    })

    # Parse start time from run_name: "20260203_060932_MuMLP_8815"
    run_name = cfg.get("run_name", "")
    training_date = "unknown"
    start_dt = None
    if run_name:
        try:
            d, t = run_name.split("_")[:2]
            training_date = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
            start_dt = datetime.datetime.strptime(training_date, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass

    # Estimate training duration from model checkpoint mtime vs start time
    training_duration = "unknown"
    run_dir = cfg.get("run_dir", "")
    if run_dir and start_dt:
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(BASE_DIR, run_dir)
        model_ckpt = os.path.join(run_dir, "models", "model_run0.pt.gz")
        if os.path.exists(model_ckpt):
            end_dt   = datetime.datetime.fromtimestamp(os.path.getmtime(model_ckpt))
            total_s  = max(0, int((end_dt - start_dt).total_seconds()))
            h, rem   = divmod(total_s, 3600)
            m, s     = divmod(rem, 60)
            training_duration = f"{h} hours {m} minutes {s} seconds"

    return {
        "model_name":         model_name,
        "model_parameters":   model_parameters,
        "training_date":      training_date,
        "training_duration":  training_duration,
    }


def update_model(onnx_path: str) -> None:
    m  = onnx.load(onnx_path)
    md = {p.key: p.value for p in m.metadata_props}

    rafal_md  = {k[len("rafal::"):]: v for k, v in md.items() if k.startswith("rafal::")}
    direct_md = {k: v for k, v in md.items() if not k.startswith("rafal::")}

    run_config_str = direct_md.get("run_config", "")
    if not run_config_str:
        print(f"  WARNING: no run_config in {os.path.basename(onnx_path)}, skipping")
        return

    cfg = yaml.safe_load(run_config_str)

    print("  Computing training min/max ...")
    x_min, x_max, y_min, y_max = compute_training_minmax(cfg)
    print(f"  x_min={x_min}")
    print(f"  x_max={x_max}")
    print(f"  y_min={y_min}")
    print(f"  y_max={y_max}")

    model_info = extract_model_info(cfg)
    print(f"  model_name={model_info['model_name']}")
    print(f"  training_date={model_info['training_date']}")
    print(f"  training_duration={model_info['training_duration']}")

    # Clean rafal metadata: strip prefix, skip dropped and replaced keys
    clean_rafal = {
        k: v for k, v in rafal_md.items()
        if k not in KEYS_TO_DROP and k not in KEYS_TO_REPLACE
    }

    # Clear all metadata and rebuild
    while len(m.metadata_props) > 0:
        m.metadata_props.pop()

    def add(key: str, value: str) -> None:
        p = m.metadata_props.add()
        p.key   = key
        p.value = str(value)

    # 1. Rafal's analysis metadata (no prefix, cleaned)
    for k, v in clean_rafal.items():
        add(k, v)

    # 2. Fixed model identity fields
    add("model_author",       MODEL_AUTHOR)
    add("model_name",         model_info["model_name"])
    add("model_parameters",   model_info["model_parameters"])
    add("training_date",      model_info["training_date"])
    add("training_duration",  model_info["training_duration"])

    # 3. Training data range (computed from our split)
    add("x_min", json.dumps(x_min))
    add("x_max", json.dumps(x_max))
    add("y_min", json.dumps(y_min))
    add("y_max", json.dumps(y_max))

    # 4. Joaquin's model keys
    for k, v in direct_md.items():
        add(k, v)

    # 5. Preprocessing description
    add("preprocessing", PREPROCESSING_META)

    onnx.save(m, onnx_path)
    print(f"  Saved ({len(m.metadata_props)} metadata keys)")


def main() -> None:
    onnx_files = sorted(
        f for f in os.listdir(MODELS_ONNX_DIR)
        if f.endswith(".onnx") and not f.startswith("._")
    )
    if not onnx_files:
        print("No ONNX files found.")
        sys.exit(1)

    for fname in onnx_files:
        print(f"\n=== {fname} ===")
        update_model(os.path.join(MODELS_ONNX_DIR, fname))

    print("\nDone.")


if __name__ == "__main__":
    main()
