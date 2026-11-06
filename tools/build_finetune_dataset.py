"""Fine-tune set: new rows (a converted simplified-model scan, tools/convert_sm_scan.py)
plus a replay sample of the base model's own data, written as a fixed on-disk split.

  * new rows are split by whole mass points (val/test points never trained on);
  * the replay is --old-frac of the base model's train block in train and of its val
    block in val, taken from the base split exactly as experiment.init_data makes it
    (seed-1234 shuffle, train = int(N*f_train), val = int(n_train*f_val/f_train), or the
    on-disk train|val|test blocks of a presplit dataset), so no row the base model was
    tested on enters training;
  * test holds the new test points only: the base model's own test block is the
    forgetting check, evaluated separately.

    python tools/build_finetune_dataset.py --old DATASET --new NEW.npy --out NAME
        [--old-frac 0.1] [--val-points 0.15] [--test-points 0.15] [--train-test-val 0.4 0.5 0.1]

Writes data/NAME.npy (train), data/NAME_val.npy, data/NAME_test.npy.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olll_training_stats import load_ordered  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True, help="the base model's dataset")
    p.add_argument("--new", required=True, help="converted scan (.npy); its _points.npy alongside")
    p.add_argument("--out", required=True)
    p.add_argument("--old-frac", type=float, default=0.10)
    p.add_argument("--val-points", type=float, default=0.15)
    p.add_argument("--test-points", type=float, default=0.15)
    p.add_argument("--train-test-val", type=float, nargs=3, default=[0.4, 0.5, 0.1])
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    data_dir = os.path.join(REPO, "data")
    rng = np.random.default_rng(a.seed)

    old, presplit = load_ordered(data_dir, a.old)
    if presplit is not None:  # on-disk train|val|test: its blocks are the split
        n_tr, n_va = presplit[0], presplit[1]
    else:
        f_tr, _, f_va = a.train_test_val
        n_tr = int(len(old) * f_tr)
        n_va = max(int(n_tr * f_va / f_tr), 1)
    old_tr, old_va = old[:n_tr], old[n_tr:n_tr + n_va]
    rep_tr = old_tr[rng.choice(n_tr, int(round(a.old_frac * n_tr)), replace=False)]
    rep_va = old_va[rng.choice(n_va, int(round(a.old_frac * n_va)), replace=False)]

    new = np.load(a.new)
    point = np.load(a.new.replace(".npy", "_points.npy"))
    assert new.shape[1] == old.shape[1], f"new rows have {new.shape[1]} columns, old {old.shape[1]}"
    pts = rng.permutation(np.unique(point))
    n_te, n_vp = int(round(a.test_points * len(pts))), int(round(a.val_points * len(pts)))
    te, va = np.isin(point, pts[:n_te]), np.isin(point, pts[n_te:n_te + n_vp])
    new_tr, new_va, new_te = new[~te & ~va], new[va], new[te]

    train = np.concatenate([new_tr, rep_tr])
    train = train[rng.permutation(len(train))]
    val = np.concatenate([new_va, rep_va])
    test = new_te
    for name, arr in [(a.out, train), (f"{a.out}_val", val), (f"{a.out}_test", test)]:
        np.save(os.path.join(data_dir, f"{name}.npy"), arr)
    print(f"old {a.old}: train {n_tr}, val {n_va}; replay {len(rep_tr)} train + {len(rep_va)} val rows")
    print(f"new: {len(new)} rows / {len(pts)} points -> train {len(new_tr)}, "
          f"val {len(new_va)} ({n_vp} pts), test {len(new_te)} ({n_te} pts)")
    print(f"wrote {a.out}: train {train.shape}, val {val.shape}, test {test.shape}")


if __name__ == "__main__":
    main()
