"""ONNX of a fine-tuned run for evaluation (not for publishing): the run's network,
exported as export_onnx_from_run.py does, with the metadata of the model it was
fine-tuned from. Valid only when the run pinned its normalisation to that model
(data.fixed_stats_onnx): the stored standardization must then match, which is checked.
run_config is replaced by the run's own, unless --keep-run-config (then tools that read
the test split from run_config, e.g. small_delta_bias.py, score the network on the
reference model's own test split: the forgetting check).

    python tools/export_eval_onnx.py RUN_DIR REFERENCE.onnx OUT.onnx [--run-idx 1] [--keep-run-config]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from export_onnx_from_run import export_graph, load_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir"); p.add_argument("reference"); p.add_argument("out")
    p.add_argument("--run-idx", type=int, default=1)
    p.add_argument("--keep-run-config", action="store_true")
    a = p.parse_args()
    device = torch.device("cpu")
    cfg, exp, idx = load_run(a.run_dir, a.run_idx, device)

    ref = onnx.load(a.reference)
    md = {q.key: q.value for q in ref.metadata_props}
    st = json.loads(md["standardization"])
    for mine, key in [(exp.prepd_mean_features[0], "features_mean"), (exp.prepd_std_features[0], "features_std"),
                      (exp.prepd_mean[0], "nLLs_mean"), (exp.prepd_std[0], "nLLs_std")]:
        d = np.abs(np.asarray(mine, dtype=float).ravel() - np.asarray(st[key], dtype=float).ravel()).max()
        assert d < 1e-9, f"{key} differs from the reference by {d}: normalisation was not pinned"

    export_graph(exp, cfg.model.net.n_features, device, a.out)
    out = onnx.load(a.out)
    if not a.keep_run_config:
        md["run_config"] = OmegaConf.to_yaml(cfg)
    del out.metadata_props[:]
    for k, v in md.items():
        out.metadata_props.add(key=k, value=v)
    onnx.save(out, a.out)
    print(f"wrote {a.out} from {a.run_dir} (run {idx}), metadata of {os.path.basename(a.reference)}")


if __name__ == "__main__":
    main()
