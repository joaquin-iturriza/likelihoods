"""Build the combined old+new training set for a full retrain, with a fixed split.

The new TChiWZoff data is a signal-strength *scan*: 101 mass points x 11 steps,
step 0 having zero signal (delta-nLL == 0 exactly). That structure drives every
choice here:

  * val/test hold out **whole mass points**, so they measure generalization to
    unseen points rather than to unseen scan steps of a point already trained on;
  * the new rows are oversampled in the **train block only** — 288k old rows would
    otherwise bury ~700 new ones — which is safe precisely because the split is
    fixed on disk (`experiment.init_data` keeps the train|val|test order and only
    shuffles within train, so duplicates cannot leak into val/test);
  * rows above `--ceiling` are dropped: the top scan steps run to delta-nLL ~ 1.7e5,
    two decades past anything the old data covers, with very little support.

--old picks the generated scan the new data is added to (default: the
wino/bino(-) offshell scan). --split-like DATASET holds out exactly the new-data
mass points that DATASET's val/test hold out, so models built on different old
scans are tested on the same new points (without it the point split depends on
the old scan's size through the shared RNG).

Writes <out>.npy (train), <out>_val.npy, <out>_test.npy into data/.
"""
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "2106.01676-offshell-winobino-minus-300k-fluct20_"
NEW = "2106.01676-offshell-TChiWZoff-jsons2"
STEPS_PER_POINT = 11


def deltas(a):
    """The 4 regression targets: mu=1 minus mu=0, per {exp, obs, expA, obsA}."""
    n = a[:, -8:]
    return np.stack([n[:, 1]-n[:, 0], n[:, 3]-n[:, 2], n[:, 5]-n[:, 4], n[:, 7]-n[:, 6]], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", default=OLD, help="the generated scan the new data is added to")
    p.add_argument("--out", default="2106.01676-winobino-minus-tchiwzoff-retrain")
    p.add_argument("--split-like", default=None,
                   help="dataset whose <name>_val/_test fix which new-data points are held out")
    p.add_argument("--ceiling", type=float, default=1e3,
                   help="drop new rows whose max|delta-nLL| exceeds this")
    p.add_argument("--new-frac", type=float, default=0.10,
                   help="target share of the train block held by new rows")
    p.add_argument("--val-points", type=int, default=15)
    p.add_argument("--test-points", type=int, default=15)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    data_dir = os.path.join(REPO, "data")
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- old data
    old = np.load(os.path.join(data_dir, f"{args.old}.npy"))
    old = old[rng.permutation(len(old))]
    n_tr = int(0.70 * len(old))
    n_va = int(0.15 * len(old))
    old_tr, old_va, old_te = old[:n_tr], old[n_tr:n_tr+n_va], old[n_tr+n_va:]

    # ---------------------------------------------------------------- new data
    new = np.load(os.path.join(data_dir, f"{NEW}.npy"))
    assert len(new) % STEPS_PER_POINT == 0, f"{len(new)} rows is not a whole number of points"
    point = np.arange(len(new)) // STEPS_PER_POINT
    n_points = point.max() + 1

    keep = np.abs(deltas(new)).max(1) <= args.ceiling
    print(f"new: {len(new)} rows / {n_points} points; "
          f"ceiling {args.ceiling:g} keeps {keep.sum()} rows ({keep.mean()*100:.1f}%)")

    perm = rng.permutation(n_points)
    te_pts = set(perm[:args.test_points].tolist())
    va_pts = set(perm[args.test_points:args.test_points+args.val_points].tolist())
    if args.split_like:
        # the points whose rows sit in that dataset's val/test files
        def held(suffix):
            rows = {r.tobytes() for r in np.load(os.path.join(data_dir, f"{args.split_like}_{suffix}.npy"))}
            return {int(point[i]) for i, r in enumerate(new) if r.tobytes() in rows}
        te_pts, va_pts = held("test"), held("val")
        assert not te_pts & va_pts, "a point is in both val and test"
        print(f"split like {args.split_like}: test points {sorted(te_pts)}, val points {sorted(va_pts)}")
    in_te = np.isin(point, list(te_pts))
    in_va = np.isin(point, list(va_pts))
    new_tr = new[keep & ~in_te & ~in_va]
    new_va = new[keep & in_va]
    new_te = new[keep & in_te]
    print(f"new split by point: train {len(new_tr)} rows / "
          f"{n_points - len(va_pts) - len(te_pts)} pts, "
          f"val {len(new_va)}/{len(va_pts)} pts, test {len(new_te)}/{len(te_pts)} pts")

    # ----------------------------------------------- oversample new in TRAIN only
    # solve reps so that reps*len(new_tr) / (len(old_tr) + reps*len(new_tr)) = frac
    f = args.new_frac
    reps = max(1, round(f * len(old_tr) / ((1 - f) * len(new_tr))))
    new_tr_os = np.repeat(new_tr, reps, axis=0)
    train = np.concatenate([old_tr, new_tr_os])
    train = train[rng.permutation(len(train))]

    # val/test: match the same new-row share by *capping the old rows* rather than
    # duplicating the new ones. Left uncapped, val would be 99.7% old data and
    # val_loss — what the sweep selects on — would be blind to the new region.
    # Reporting on the full old set is a separate, unweighted evaluation.
    cap = int(round((1 - f) / f * len(new_va)))
    val = np.concatenate([old_va[:cap], new_va])
    cap = int(round((1 - f) / f * len(new_te)))
    test = np.concatenate([old_te[:cap], new_te])

    print(f"oversampling new train rows x{reps} -> {len(new_tr_os)} "
          f"({len(new_tr_os)/len(train)*100:.1f}% of train)")
    print(f"new share: val {len(new_va)/len(val)*100:.1f}%, test {len(new_te)/len(test)*100:.1f}%")
    print(f"train {train.shape}  val {val.shape}  test {test.shape}")

    for name, arr in [(args.out, train), (f"{args.out}_val", val), (f"{args.out}_test", test)]:
        path = os.path.join(data_dir, f"{name}.npy")
        np.save(path, arr)
        print("wrote", path, arr.shape)

    t = deltas(train)
    print(f"train target range per output: "
          f"{[f'[{t[:, i].min():.4g}, {t[:, i].max():.4g}]' for i in range(4)]}")
    print(f"train rows with an exact zero: {int((t == 0).any(1).sum())}")


if __name__ == "__main__":
    main()
