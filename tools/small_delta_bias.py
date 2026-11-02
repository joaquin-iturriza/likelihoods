"""Bias of an ONNX model's predicted deltas on its test split, binned in the true
delta: how many test points sit near zero and how far the prediction is off there
(the region a limit at small signal depends on).

    python tools/small_delta_bias.py MODEL.onnx [MODEL.onnx ...]
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_plots import inverse_preprocess_nLLs, load_test_split, run_onnx  # noqa: E402

EDGES = [0, 0.05, 0.25, 0.5, 1, 2, 3, 5, 10, np.inf]
NAMES = ["exp", "obs", "expA", "obsA"]


def bias_table(path):
    md = {p.key: p.value for p in onnx.load(path).metadata_props}
    cfg = yaml.safe_load(md["run_config"])
    std = json.loads(md["standardization"])
    X, truth, _, _ = load_test_split(cfg, std, max_events=None)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    lo, hi = std.get("nLLs_lo"), std.get("nLLs_hi")
    pred = inverse_preprocess_nLLs(run_onnx(sess, X), np.array(std["nLLs_mean"][0]),
                                   np.array(std["nLLs_std"][0]),
                                   pipeline=cfg["data"].get("nLL_trafos") or None,
                                   nll_bounds={"lo": lo, "hi": hi} if lo is not None else {})
    print(f"=== {os.path.basename(path)}  ({len(truth)} test rows)")
    for i in (0, 1):
        t, r = truth[:, i], pred[:, i] - truth[:, i]
        print(f"  {NAMES[i]}: true-delta bin   rows   median(pred-true)  mean(pred-true)")
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            m = (t >= a) & (t < b)
            if m.any():
                print(f"    [{a:5g}, {b:5g})  {m.sum():8d}  {np.median(r[m]):+10.4f}  {r[m].mean():+10.4f}")
        if i == 1:
            m = t < 0
            if m.any():
                print(f"    (< 0)          {m.sum():8d}  {np.median(r[m]):+10.4f}  {r[m].mean():+10.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        bias_table(p)
