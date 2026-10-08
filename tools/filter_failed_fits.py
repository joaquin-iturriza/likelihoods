"""
Drop rows whose nLL(mu=1) comes from a failed fit in the sampling pipeline.

    python tools/filter_failed_fits.py DATASET [--max-delta 100] [--suffix -fitok]

Reads <DATA_DIR>/DATASET.npy (rows = [yields | 8 nLL columns]) and writes
DATASET<suffix>.npy without the rows where any of the four deltas
nLL(mu=1) - nLL(mu=0) is above --max-delta. The sidecar files go along:
DATASET.json (with x/y_min/max recomputed on the kept rows and a "filtered"
record) and DATASET_columns.txt, when they exist.

Why: R. Maselek's sampler profiles nLL(mu=1) with spey, which does not raise
when the optimizer fails. On the 2018-16 EWkino scans SciPy's SLSQP stops at
its first iteration ("Inequality constraints incompatible") on ~0.5-3% of the
points, and the objective at the starting parameters is stored as the result:
a delta of ~500-3000 where the neighbouring chain points have ~10-20. Refitting
two such rows with pyhf on the HEPData workspace reproduced the stored values to
two decimals at exactly that failure. Each of the four outputs fails on its own,
so a row is dropped when ANY delta is above the cut. On those scans the genuine
deltas end well below 100 and the failed ones start around 500.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import siteconf

NAMES = ["exp", "obs", "expA", "obsA"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--max-delta", type=float, default=100.0)
    ap.add_argument("--suffix", default="-fitok")
    ap.add_argument("--data-dir", default=None, help="default: siteconf.DATA_DIR")
    a = ap.parse_args()

    ddir = a.data_dir or siteconf.DATA_DIR
    src = os.path.join(ddir, a.dataset)
    dst = os.path.join(ddir, a.dataset + a.suffix)
    if os.path.exists(dst + ".npy"):
        sys.exit(f"{dst}.npy exists; not overwriting")

    data = np.load(src + ".npy")
    nll = data[:, -8:]
    delta = nll[:, 1::2] - nll[:, 0::2]
    failed = delta > a.max_delta
    drop = failed.any(axis=1)
    kept = data[~drop]

    print(f"{a.dataset}: {len(data)} rows, dropping {drop.sum()} ({drop.mean():.2%}) with a delta > {a.max_delta:g}")
    for i, n in enumerate(NAMES):
        print(f"  {n:5s} failed {failed[:, i].sum():6d}   kept max {delta[~drop, i].max():9.3f}")
    np.save(dst + ".npy", kept)
    print(f"wrote {dst}.npy  {kept.shape}")

    if os.path.exists(src + "_columns.txt"):
        shutil.copyfile(src + "_columns.txt", dst + "_columns.txt")
    if os.path.exists(src + ".json"):
        meta = json.load(open(src + ".json"))
        x, y = kept[:, :-8], kept[:, -8:]
        meta.update(x_min=x.min(0).tolist(), x_max=x.max(0).tolist(),
                    y_min=y.min(0).tolist(), y_max=y.max(0).tolist())
        meta["filtered"] = {"source": a.dataset, "tool": "tools/filter_failed_fits.py",
                            "max_delta": a.max_delta, "rows_in": int(len(data)),
                            "rows_dropped": int(drop.sum()),
                            "failed_per_output": dict(zip(NAMES, map(int, failed.sum(0))))}
        with open(dst + ".json", "w") as f:
            json.dump(meta, f, indent=4)
        print(f"wrote {dst}.json")


if __name__ == "__main__":
    main()
