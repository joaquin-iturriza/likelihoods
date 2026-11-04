"""Bias of an ONNX model's predicted deltas on its test split, binned in the true
delta: how many test points sit near zero and how far the prediction is off there
(the region a limit at small signal depends on).

    python tools/small_delta_bias.py MODEL.onnx [MODEL.onnx ...]
    python tools/small_delta_bias.py MODEL.onnx --data ROWS.npy   # any [inputs | 8 nLLs] file
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
import olll_transforms  # noqa: E402,F811

EDGES = [0, 0.05, 0.25, 0.5, 1, 2, 3, 5, 10, np.inf]
NAMES = ["exp", "obs", "expA", "obsA"]


def bias_table(path, data=None):
    md = {p.key: p.value for p in onnx.load(path).metadata_props}
    cfg = yaml.safe_load(md["run_config"])
    std = json.loads(md["standardization"])
    if data is None:
        X, truth, _, _ = load_test_split(cfg, std, max_events=None)
    else:
        rows = np.load(data)
        n = rows[:, -8:]
        truth = n[:, 1::2] - n[:, 0::2]
        X, _, _ = olll_transforms.preprocess_features(
            rows[:, :-8].astype(np.float64), trafos=cfg["data"]["trafos"],
            mean=np.asarray(std["features_mean"][0]), std=np.asarray(std["features_std"][0]))
        X = X.astype(np.float32)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    lo, hi = std.get("nLLs_lo"), std.get("nLLs_hi")
    pred = inverse_preprocess_nLLs(run_onnx(sess, X), np.array(std["nLLs_mean"][0]),
                                   np.array(std["nLLs_std"][0]),
                                   pipeline=cfg["data"].get("nLL_trafos") or None,
                                   nll_bounds={"lo": lo, "hi": hi} if lo is not None else {})
    print(f"=== {os.path.basename(path)} on {os.path.basename(data) if data else 'its test split'}"
          f"  ({len(truth)} rows)")
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
    args = sys.argv[1:]
    data = None
    if "--data" in args:
        k = args.index("--data")
        data = args[k + 1]
        del args[k:k + 2]
    for p in args:
        bias_table(p, data)
