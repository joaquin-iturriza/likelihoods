import numpy as np
import pandas as pd
import os

# List of CSV filenames
csv_files = ["1909.09226-leakage-10_.csv", "1911.06660-leakage-10_.csv"
             , "1911.12606-EWKinos-1M-z4-nll400-delta200.csv", "1911.12606-sleptons-700k-fluct30%-nll300--delta300-z3.csv",
             '2106.01676-offshell-higgsino-300k-fluct20_.csv','1911.12606-sleptons-200k-fluct20_.csv',
             '2106.01676-offshell-winobino-minus-300k-fluct20_.csv','2106.01676-offshell-winobino-plus-fluct20_-300k.csv',
             "2106.01676-onshell-winobino-fluct25_-300k.csv","1911.12606-EWKinos-1M-fluct100_new-trim.csv","1911.12606-sleptons-700k-fluct30_ll300--delta300-z3.csv"]

for csv_file in csv_files:
    # Read CSV using pandas
    df = pd.read_csv(csv_file)

    # Convert to NumPy array
    data = df.to_numpy()

    # Save as .npy
    npy_filename = os.path.splitext(csv_file)[0] + ".npy"
    np.save(npy_filename, data)

    print(f"Saved {npy_filename}")