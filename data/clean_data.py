import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------
# User inputs
# ------------------------------------------------------------
csv_path = "1911.06660-leakage-10_.csv"
npy_path = "1911.06660-leakage-10_.npy"

EPS = 1e-6

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = pd.read_csv(csv_path)
data_raw = np.load(npy_path, allow_pickle=True)

assert len(df) == len(data_raw), "CSV and NPY must have same number of rows"

# ------------------------------------------------------------
# 2. Compute nLLs EXACTLY as in training
# ------------------------------------------------------------
nLLs = data_raw[:, -8:].copy()

# baseline subtraction
for i in range(4):
    nLLs[:, 2 * i + 1] -= nLLs[:, 2 * i]

# keep only the differences
nLLs = nLLs[:, 1::2]   # shape (N, 4)

# ------------------------------------------------------------
# 3. Build mask: keep rows where ALL nLLs are outside [-eps, eps]
# ------------------------------------------------------------
bad_mask = np.any(
    (nLLs > -EPS) & (nLLs < EPS),
    axis=1
)

good_mask = ~bad_mask

n_removed = np.sum(bad_mask)
n_total = len(nLLs)

# ------------------------------------------------------------
# 4. Apply mask
# ------------------------------------------------------------
df_cleaned = df.loc[good_mask].reset_index(drop=True)
data_raw_cleaned = data_raw[good_mask]

# ------------------------------------------------------------
# 5. Save cleaned files
# ------------------------------------------------------------
csv_out = os.path.splitext(csv_path)[0] + "_cleaned.csv"
npy_out = os.path.splitext(npy_path)[0] + "_cleaned.npy"

df_cleaned.to_csv(csv_out, index=False)
np.save(npy_out, data_raw_cleaned)

# ------------------------------------------------------------
# 6. Report
# ------------------------------------------------------------
print(f"Total datapoints     : {n_total}")
print(f"Removed datapoints   : {n_removed}")
print(f"Remaining datapoints : {len(df_cleaned)}")
print(f"Saved:")
print(f"  {csv_out}")
print(f"  {npy_out}")
