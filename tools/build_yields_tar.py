#!/usr/bin/env python3
"""Convert a collaborator "jsons" yields tar (walten/smodels export) into a
training-ready CSV+NPY matching the existing data/ schema.

The per-point .csv members inside the tar have a *broken header*: it under- or
over-names the region columns (systematic off-by-N vs the data), so the header
cannot be trusted. What IS reliable and verified:

  * the last 8 data fields of every row are the nLL block
    [nLL_exp_mu0, nLL_exp_mu1, nLL_obs_mu0, nLL_obs_mu1,
     nLLA_exp_mu0, nLLA_exp_mu1, nLLA_obs_mu0, nLLA_obs_mu1]
  * the leading data fields are the region yields in the *canonical* order used
    by the deployed training data (CR_0J_WZ, CR_nJ_WZ, SRhigh_*, SRlow_*).

So we ignore each member's header and map columns POSITIONALLY onto the canonical
41-column schema taken from an existing training CSV. Validation (per-column
medians and the mu=0 baselines) confirms this alignment against the shipped
2106.01676 offshell data.

Files that do not have exactly 41 data columns carry a different region set
(extra onshell SRee/SRmm regions) and are dropped with a report, as are members
whose packaging is corrupt (a stray gzip-tar blob).

Each point contributes 11 rows: the same mass point with its signal scaled by an
increasing normalisation (row 0 ~ background-only, last row ~ large signal).

Usage:
    python tools/build_yields_tar.py yields/jsons2.tar \
        --out data/2106.01676-offshell-TChiWZoff-jsons2.csv
"""
import argparse, gzip, io, os, tarfile
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CANON = os.path.join(
    REPO, "data", "2106.01676-offshell-winobino-minus-300k-fluct20_.csv")
NLL = ["nLL_exp_mu0", "nLL_exp_mu1", "nLL_obs_mu0", "nLL_obs_mu1",
       "nLLA_exp_mu0", "nLLA_exp_mu1", "nLLA_obs_mu0", "nLLA_obs_mu1"]


def _read_member(tar, m):
    """Return a header-less DataFrame of a tar member, transparently unwrapping
    the occasional gzip-tar blob some members are mistakenly packed as."""
    raw = tar.extractfile(m).read()
    if raw[:2] == b"\x1f\x8b":                      # gzip -> inner tar -> csv
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as inner:
            im = inner.getmembers()[0]
            if os.path.basename(im.name) != os.path.basename(m.name):
                return None, f"corrupt gzip-tar (inner={im.name})"
            raw = inner.extractfile(im).read()
    return pd.read_csv(io.BytesIO(raw), header=None, skiprows=1), None


def build(tar_path, out_csv, canon_csv=DEFAULT_CANON):
    canon = list(pd.read_csv(canon_csv, nrows=0).columns)   # 41 cols
    assert canon[-8:] == NLL and len(canon) == 41, canon

    rows, used, dropped = [], [], []
    with tarfile.open(tar_path) as tar:
        for m in sorted((x for x in tar.getmembers()
                         if x.isfile() and x.name.endswith(".csv")),
                        key=lambda x: x.name):
            name = os.path.basename(m.name)[:-4]
            df, err = _read_member(tar, m)
            if err:
                dropped.append((name, err)); continue
            if df.shape[1] != 41:
                dropped.append((name,
                    f"{df.shape[1]} data cols (non-canonical region set)")); continue
            df.columns = canon                          # positional -> canonical
            rows.append(df); used.append(name)

    comb = pd.concat(rows, ignore_index=True)[canon]
    comb.to_csv(out_csv, index=False)
    np.save(os.path.splitext(out_csv)[0] + ".npy", comb.to_numpy())

    print(f"points used : {len(used)}")
    print(f"rows        : {len(comb)}  ({len(comb)//max(len(used),1)} per point)")
    print(f"cols        : {comb.shape[1]}  (matches {os.path.basename(canon_csv)})")
    print(f"NaN / Inf   : {bool(comb.isna().any().any())} / "
          f"{bool(np.isinf(comb.to_numpy()).any())}")
    if dropped:
        print(f"dropped ({len(dropped)}):")
        for n, why in dropped:
            print(f"   {n}: {why}")
    print(f"wrote {out_csv}")
    print(f"wrote {os.path.splitext(out_csv)[0]}.npy")
    return comb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tar", help="collaborator yields tar (e.g. yields/jsons2.tar)")
    ap.add_argument("--out", required=True, help="output CSV path under data/")
    ap.add_argument("--canon", default=DEFAULT_CANON,
                    help="existing training CSV to take the 41-col schema from")
    a = ap.parse_args()
    build(a.tar, a.out, a.canon)
