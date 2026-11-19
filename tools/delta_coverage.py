"""How far a training set is from covering the small-delta region: for each of the four
nLL deltas [exp, obs, expA, obsA] (mu=1 minus mu=0), the smallest values in the block
the model trained on, and how its rows spread over bins around the 95% CL threshold
(delta = 1.92, i.e. q = 3.84).

    python tools/delta_coverage.py NAME[:TRAIN_FRAC[:SUBSAMPLE]] ...

The train block is taken as experiment.init_data takes it (seed-1234 shuffle,
train = min(SUBSAMPLE, int(N*TRAIN_FRAC)), or the on-disk block of a presplit set).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olll_training_stats import load_ordered  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["exp", "obs", "expA", "obsA"]
EDGES = [-np.inf, 0, 0.05, 0.5, 1, 1.92, 3, 5, np.inf]


def coverage(spec):
    name, *rest = spec.split(":")
    frac = float(rest[0]) if rest else 0.4
    sub = int(rest[1]) if len(rest) > 1 and rest[1] else None
    data, presplit = load_ordered(os.path.join(REPO, "data"), name)
    if presplit is not None:
        n_tr = presplit[0]
    else:
        n_tr = int(len(data) * frac)
        if sub is not None:
            n_tr = min(sub, n_tr)
    nll = data[:n_tr, -8:].astype(np.float64)
    delta = nll[:, 1::2] - nll[:, 0::2]
    full = data[:, -8:].astype(np.float64)
    full_min = (full[:, 1::2] - full[:, 0::2]).min(0)
    print(f"=== {name}: train block {n_tr} of {len(data)} rows")
    head = "".join(f"{f'[{a:g},{b:g})':>11s}" for a, b in zip(EDGES[:-1], EDGES[1:]))
    print(f"  {'':5s} {'min':>8s} {'q0.1%':>8s} {'q1%':>8s} {'q5%':>8s} {'min(all)':>9s}  {head}")
    for k, n in enumerate(NAMES):
        d = delta[:, k]
        q = np.quantile(d, [0.001, 0.01, 0.05])
        frac_bins = np.histogram(d, EDGES)[0] / len(d) * 100
        print(f"  {n:5s} {d.min():8.3f} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} {full_min[k]:9.3f}  "
              + "".join(f"{v:10.2f}%" for v in frac_bins))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        coverage(s)
