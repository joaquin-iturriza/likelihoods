#!/usr/bin/env python3
"""Build training sets for ATLAS-SUSY-2018-16 from the collaborator scan tarballs
(`201816sleptons.tar.gz` = full_slep/, `201816chiwzoff.tar.gz` = full_chiwzoff/).

What the tarballs actually contain
----------------------------------
Each `.csv` member is one mass point scanned over a signal-strength ladder of 11
steps, mu = [0, 1e-5, 1e-4, 5e-4, 2e-3, 4e-3, 1e-2, 2e-2, 5e-2, 2e-1, 1] x 100.
Row 0 is signal-free, so its delta-nLL is exactly 0. Columns are
`[ yields... | 8 nLL columns ]`, the same schema as `data/`.

The folders are *simplified-model* scans, not one dataset per analysis: SModelS
dumps whichever analysis it evaluated for that point, so a folder mixes analyses
and the member width identifies which:

    width 46 (38 regions)  ATLAS-SUSY-2018-16 sleptons  <-> 1911.12606-sleptons-*
    width 52 (44 regions)  ATLAS-SUSY-2018-16 EWkinos   <-> 1911.12606-EWKinos-*
    width 39 (31 regions)  ATLAS-SUSY-2019-09 offshell  <-> 2106.01676-offshell-*

so `full_chiwzoff/` is mostly 2019-09, not 2018-16. Only `--channel` selects what
is built; the other widths are reported and skipped.

The per-member header is unusable: it is written from the `nsignals` dict, which
drops zero-signal regions, so its length varies point to point and is always <=
the data width. Columns are therefore mapped POSITIONALLY. That mapping is
verified, not assumed:

  * the ordered region list recovered by topologically merging every point's
    `nsignals` sequence is consistent across all points (0 inversions) and matches
    the deployed model's `bkg_yields` order one-for-one, region for region;
  * row 0's control-region values reproduce the deployed model's `obs_yields`
    exactly (sleptons 367/272/218/799/2053/3106, EWkinos 687/721/592/2247/4327/6164).

The nLL block carries the *truth* values (spey/pyhf on the ATLAS background-only
workspace), i.e. the JSONs' `<analysis>-orig` entries, not the surrogate ONNX
predictions that the unsuffixed entries hold. Note the mu=0 baselines sit
+0.918939 (= 0.5*ln(2*pi)) above those in `data/`, a normalisation change in the
newer spey; it cancels in the mu=1 - mu=0 delta that is regressed, but any ONNX
exported from this data needs its `nLL_*_mu0` metadata refreshed.

Split
-----
val/test hold out **whole mass points**, so they measure generalisation to unseen
points rather than to unseen ladder steps of a point already trained on. Written
as `<out>.npy` / `<out>_val.npy` / `<out>_test.npy`, the fixed-on-disk layout
`experiment.init_data` consumes (it keeps train|val|test order and shuffles only
within train).

Usage:
    python tools/build_susy201816_dataset.py --channel sleptons \
        --out 1911.12606-sleptons-scan-201816
"""
import argparse
import glob
import io
import os
import tarfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# width of a data row -> (channel key, n regions, what it really is)
WIDTHS = {
    46: ("sleptons", 38, "ATLAS-SUSY-2018-16 sleptons"),
    52: ("ewkinos", 44, "ATLAS-SUSY-2018-16 EWkinos"),
    39: ("offshell", 31, "ATLAS-SUSY-2019-09 offshell (NOT 2018-16)"),
}
SOURCES = {
    "sleptons": ["201816sleptons.tar.gz", "201816chiwzoff.tar.gz"],
    "ewkinos": ["201816sleptons.tar.gz", "201816chiwzoff.tar.gz"],
    "offshell": ["201816chiwzoff.tar.gz"],
}


def _members(path):
    """Yield (name, ndarray) for every .csv member, header dropped, None -> NaN."""
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)):
            yield os.path.basename(f)[:-4], open(f, "rb").read()
        return
    with tarfile.open(path) as tar:
        for m in sorted((x for x in tar.getmembers() if x.isfile()
                         and x.name.endswith(".csv")), key=lambda x: x.name):
            yield os.path.basename(m.name)[:-4], tar.extractfile(m).read()


def _parse(raw):
    lines = io.BytesIO(raw).read().decode().splitlines()[1:]   # header is unusable
    return np.array([[np.nan if v == "None" else float(v) for v in ln.split(",")]
                     for ln in lines if ln.strip()])


def deltas(a):
    """The 4 regression targets: mu=1 minus mu=0, per {exp, obs, expA, obsA}."""
    n = a[:, -8:]
    return np.stack([n[:, 1]-n[:, 0], n[:, 3]-n[:, 2],
                     n[:, 5]-n[:, 4], n[:, 7]-n[:, 6]], 1)


def collect(sources, channel):
    """Return {point_name: rows} for the requested channel, plus a width census."""
    want = next(w for w, (c, _, _) in WIDTHS.items() if c == channel)
    points, census, dropped = {}, {}, []
    for src in sources:
        p = src if os.path.isabs(src) else os.path.join(REPO, src)
        if not os.path.exists(p):
            raise SystemExit(f"missing source: {p}")
        for name, raw in _members(p):
            a = _parse(raw)
            if a.ndim != 2 or a.shape[0] == 0:
                dropped.append((name, "unparseable")); continue
            census[a.shape[1]] = census.get(a.shape[1], 0) + 1
            if a.shape[1] != want:
                continue
            bad = ~np.isfinite(a).all(1)
            if bad.all():
                dropped.append((name, "all rows non-finite")); continue
            if bad.any():
                dropped.append((name, f"{int(bad.sum())}/{len(a)} non-finite rows dropped"))
                a = a[~bad]
            if name in points:
                dropped.append((name, "duplicate point name, kept first")); continue
            points[name] = a
    return points, census, dropped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", required=True, choices=sorted(SOURCES))
    p.add_argument("--out", required=True, help="output basename under data/")
    p.add_argument("--source", nargs="*", default=None,
                   help="tarballs or extracted dirs (default: the two 2018-16 tarballs)")
    p.add_argument("--ceiling", type=float, default=None,
                   help="drop rows whose max|delta-nLL| exceeds this")
    p.add_argument("--val-frac", type=float, default=0.15, help="share of mass points")
    p.add_argument("--test-frac", type=float, default=0.15, help="share of mass points")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    points, census, dropped = collect(a.source or SOURCES[a.channel], a.channel)
    want = next(w for w, (c, _, _) in WIDTHS.items() if c == a.channel)

    print("member width census (all sources):")
    for w, n in sorted(census.items()):
        c, nreg, what = WIDTHS.get(w, ("?", w-8, "unknown region set"))
        mark = "  <- selected" if w == want else ""
        print(f"   width {w:3d} ({nreg:2d} regions): {n:4d} points   {what}{mark}")
    if not points:
        raise SystemExit(f"no points of width {want} found for channel {a.channel}")

    names = sorted(points)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(names))
    n_te = max(1, int(round(a.test_frac * len(names))))
    n_va = max(1, int(round(a.val_frac * len(names))))
    te = {names[i] for i in perm[:n_te]}
    va = {names[i] for i in perm[n_te:n_te+n_va]}

    blocks = {"train": [], "val": [], "test": []}
    kept = total = 0
    for n in names:
        arr = points[n]
        total += len(arr)
        if a.ceiling is not None:
            arr = arr[np.abs(deltas(arr)).max(1) <= a.ceiling]
            if not len(arr):
                continue
        kept += len(arr)
        blocks["test" if n in te else "val" if n in va else "train"].append(arr)

    parts = {k: np.concatenate(v) if v else np.empty((0, want)) for k, v in blocks.items()}
    train = parts["train"][rng.permutation(len(parts["train"]))]

    print(f"\nchannel {a.channel}: {len(names)} mass points, {total} rows"
          + (f"; ceiling {a.ceiling:g} keeps {kept} ({kept/total*100:.1f}%)"
             if a.ceiling is not None else ""))
    print(f"split by mass point: train {len(names)-n_va-n_te} pts / {len(train)} rows | "
          f"val {n_va} pts / {len(parts['val'])} rows | "
          f"test {n_te} pts / {len(parts['test'])} rows")
    if dropped:
        print(f"notes ({len(dropped)}):")
        for n, why in dropped[:20]:
            print(f"   {n}: {why}")
        if len(dropped) > 20:
            print(f"   ... {len(dropped)-20} more")

    data_dir = os.path.join(REPO, "data")
    for tag, arr in [(a.out, train), (f"{a.out}_val", parts["val"]),
                     (f"{a.out}_test", parts["test"])]:
        path = os.path.join(data_dir, f"{tag}.npy")
        np.save(path, arr)
        print("wrote", path, arr.shape)

    t = deltas(train)
    print("train target range per output: "
          + str([f"[{t[:, i].min():.4g}, {t[:, i].max():.4g}]" for i in range(4)]))
    print(f"train rows with an exact zero target: {int((t == 0).any(1).sum())} / {len(t)}")


if __name__ == "__main__":
    main()
