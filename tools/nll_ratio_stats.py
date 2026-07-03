"""
nll_ratio_stats.py

For a given ONNX model, run it over the test split and report, for each of the
four nLLs (Expected, Observed, Expected Asimov, Observed Asimov), the mean and
error of the ratio:

        nLL_pred / nLL_truth

The data loading, preprocessing and preprocessing-inversion are reused verbatim
from validation_plots.py so the numbers match what the model actually trained on.

Usage:
    python nll_ratio_stats.py [model.onnx] [--min-truth 1e-3]

Defaults to models_onnx/SUSY-2019-02_Sleptons_700k.onnx.
"""

import os
import json
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import yaml

from validation_plots import (
    load_test_split,
    run_onnx,
    inverse_preprocess_nLLs,
    SUBPLOT_TITLES,   # ["Expected", "Observed", "Expected Asimov", "Observed Asimov"]
)

# BASE_DIR is the directory this script lives in, resolved from __file__ so it
# works regardless of where the eos area is mounted (locally or on lxplus, where
# it is /eos/user/j/joiturri/likelihoods).
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in tools/)
MODELS_DIR   = os.path.join(BASE_DIR, "models_onnx")
DEFAULT_ONNX = os.path.join(MODELS_DIR, "SUSY-2018-16_Sleptons_700k.onnx")


def resolve_model_path(arg: str) -> str:
    """Resolve a model argument so the script runs from any cwd on lxplus.

    Order: as-given (abs or relative to cwd) → relative to models_onnx/ →
    relative to BASE_DIR. First existing match wins.
    """
    candidates = [
        arg,
        os.path.join(MODELS_DIR, arg),
        os.path.join(MODELS_DIR, os.path.basename(arg)),
        os.path.join(BASE_DIR, arg),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find model '{arg}'. Tried:\n  " + "\n  ".join(candidates))


def fmt_val_err(val: float, err: float):
    """Format (value, error) with the error to 2 significant figures and the
    value rounded to the error's last significant digit (standard 'x ± dx').

    Returns (val_str, err_str). Falls back to plain formatting when the error
    is zero / non-finite (e.g. N<=1) so nothing blows up.
    """
    import math
    if not math.isfinite(err) or err == 0:
        return f"{val:.6g}", f"{err:.2g}"
    exp = math.floor(math.log10(abs(err)))   # decade of the leading error digit
    ndec = max(0, -(exp - 1))                # decimals needed for 2 sig figs
    return f"{val:.{ndec}f}", f"{err:.{ndec}f}"


def ratio_stats(truth: np.ndarray, pred: np.ndarray, min_truth: float):
    """Per-column (mean, sem, std, n) of pred/truth, ignoring tiny/non-finite truth."""
    out = []
    for i in range(truth.shape[1]):
        t, p = truth[:, i], pred[:, i]
        mask = np.isfinite(t) & np.isfinite(p) & (np.abs(t) >= min_truth)
        ratio = p[mask] / t[mask]
        ratio = ratio[np.isfinite(ratio)]
        n = len(ratio)
        mean = float(np.mean(ratio)) if n else float("nan")
        std  = float(np.std(ratio, ddof=1)) if n > 1 else float("nan")
        sem  = std / np.sqrt(n) if n > 1 else float("nan")
        out.append((mean, sem, std, n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=DEFAULT_ONNX, help="ONNX model file")
    ap.add_argument("--min-truth", type=float, default=1e-3,
                    help="ignore events with |nLL_truth| below this (avoids /0 blow-up)")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)

    # --- metadata: config + standardization (same as validation_plots.process_model) ---
    m  = onnx.load(model_path)
    md = {p.key: p.value for p in m.metadata_props}
    cfg = yaml.safe_load(md["run_config"])
    standardization = json.loads(md["standardization"])
    nLLs_mean = np.array(standardization["nLLs_mean"][0])
    nLLs_std  = np.array(standardization["nLLs_std"][0])
    nll_pipeline = cfg.get("data", {}).get("nLL_trafos") or None
    nll_lo = standardization.get("nLLs_lo")
    nll_hi = standardization.get("nLLs_hi")
    nll_bounds = {"lo": nll_lo, "hi": nll_hi} if nll_lo is not None else {}

    # --- test split: features (preprocessed) + delta-nLL truth + nLL0 baseline ---
    print(f"Model: {model_path}")
    print("Loading test split ...")
    features_prepd, nLLs_truth_delta, nLL0 = load_test_split(cfg, standardization, max_events=None)
    print(f"Test events: {len(features_prepd)}")

    # --- inference + invert preprocessing → predicted delta-nLLs ---
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    preds_prepd = run_onnx(session, features_prepd)
    nLLs_pred_delta = inverse_preprocess_nLLs(
        preds_prepd, nLLs_mean, nLLs_std, pipeline=nll_pipeline, nll_bounds=nll_bounds)

    # Absolute nLL1 = baseline (mu=0) + predicted/true delta
    nLL1_truth = nLL0 + nLLs_truth_delta
    nLL1_pred  = nLL0 + nLLs_pred_delta

    # --- report ratio stats for both the absolute nLL1 and the delta-nLL ---
    for label, truth, pred in [
        ("nLL1 = nLL0 + delta (absolute)", nLL1_truth, nLL1_pred),
        ("delta-nLL (nLL1 - nLL0)",        nLLs_truth_delta, nLLs_pred_delta),
    ]:
        stats = ratio_stats(truth, pred, args.min_truth)
        print(f"\n=== ratio pred/truth for {label} ===")
        print(f"{'nLL':<18}{'mean ± error (SEM)':>26}{'std':>14}{'N':>10}")
        for name, (mean, sem, std, n) in zip(SUBPLOT_TITLES, stats):
            mean_s, sem_s = fmt_val_err(mean, sem)   # error to 2 sig figs, mean matched
            _, std_s = fmt_val_err(mean, std)        # std also to 2 sig figs
            print(f"{name:<18}{mean_s + ' ± ' + sem_s:>26}{std_s:>14}{n:>10d}")


if __name__ == "__main__":
    main()
