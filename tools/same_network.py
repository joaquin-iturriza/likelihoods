"""Is a run's checkpoint the network of a published ONNX file? Exports the run's graph and
compares both files' raw outputs on random standardized inputs (the base of a fine-tune
must be the published model).

    python tools/same_network.py RUN_DIR REFERENCE.onnx [--run-idx 0]
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from export_onnx_from_run import export_graph, load_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir"); p.add_argument("reference")
    p.add_argument("--run-idx", type=int, default=0)
    a = p.parse_args()
    cfg, exp, idx = load_run(a.run_dir, a.run_idx, torch.device("cpu"))
    n = int(cfg.model.net.n_features)
    x = np.random.default_rng(0).normal(size=(1000, n)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.onnx")
        export_graph(exp, n, torch.device("cpu"), path)
        mine = ort.InferenceSession(path).run(None, {"features": x})[0]
    ref = ort.InferenceSession(a.reference).run(None, {"features": x})[0]
    d = np.abs(mine - ref).max()
    print(f"{a.run_dir} (run {idx}) vs {os.path.basename(a.reference)}: max |output diff| {d:.2e}")
    sys.exit(0 if d < 1e-4 else 1)


if __name__ == "__main__":
    main()
