"""Deduplicate a collaborator CSV and write the training-ready .npy.

Same output contract as conv_csv_npy.py ([ yields... | 8 nLL columns ], float64),
but drops exactly-repeated rows first. The 2018-16 boundary scans arrive with
~18% duplicated rows (including the signal-free anchor repeated dozens of times),
which would otherwise straddle the random train/val/test split.

Usage:  python data/dedup_csv_npy.py <in.csv> <out.npy> [<in.csv> <out.npy> ...]
"""
import os
import sys

import numpy as np
import pandas as pd

NLL_COLS = ["nLL_exp_mu0", "nLL_exp_mu1", "nLL_obs_mu0", "nLL_obs_mu1",
            "nLLA_exp_mu0", "nLLA_exp_mu1", "nLLA_obs_mu0", "nLLA_obs_mu1"]


def convert(csv_path, npy_path):
    df = pd.read_csv(csv_path)
    assert list(df.columns[-8:]) == NLL_COLS, (
        f"{csv_path}: last 8 columns are {list(df.columns[-8:])}, expected {NLL_COLS}"
    )
    n_raw = len(df)
    df = df.drop_duplicates(ignore_index=True)
    data = df.to_numpy(dtype=np.float64)
    assert np.isfinite(data).all(), f"{csv_path}: non-finite values after dedup"

    np.save(npy_path, data)
    print(f"{os.path.basename(csv_path)}: {n_raw} -> {len(df)} rows "
          f"({n_raw - len(df)} duplicates dropped, {100*(n_raw-len(df))/n_raw:.1f}%), "
          f"{data.shape[1] - 8} features -> {npy_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    for csv_path, npy_path in zip(args[::2], args[1::2]):
        convert(csv_path, npy_path)
