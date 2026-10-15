"""Model-free check of a likelihood dataset's labels, and the dataset without the rows it flags.

A row is flagged when, for any of the four nLL deltas [exp, obs, expA, obsA], its delta differs
from the median delta of its K nearest neighbours in standardised yield space by more than THR.
On R. Maselek's 2018-16 scans these are mu=1 fits that did not converge: the nLL at mu=1 is too
high by 2-5, and a network trained on the file agrees with the neighbours, not with the label.

    python tools/label_check.py DATA.csv|DATA.npy OUT_PREFIX [--k 10] [--thr 1.0] [--drop OUT.npy]

Prints the flag rates and their profile along the file's row order (a merged file keeps its
sources in blocks), saves OUT_PREFIX.npz (flag [n,4], residual [n,4], an 8-byte hash per row
of its yields to match rows between files). With --drop, writes the table without the flagged
rows (any output), rows and columns otherwise in the input's order, as a training-ready .npy.
"""
import argparse, hashlib, time
import numpy as np
from sklearn.neighbors import NearestNeighbors

OUT = ["exp", "obs", "expA", "obsA"]
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out_prefix")
ap.add_argument("--k", type=int, default=10); ap.add_argument("--thr", type=float, default=1.0)
ap.add_argument("--drop", default=None)
a = ap.parse_args()

t = np.load(a.data) if a.data.endswith(".npy") else np.loadtxt(a.data, delimiter=",", skiprows=1)
X, L = t[:, :-8], t[:, -8:]
D = L[:, 1::2] - L[:, 0::2]
n = len(t)
print(f"{a.data}: {n} rows, {X.shape[1]} yields", flush=True)
Xs = (X - X.mean(0)) / X.std(0)
t0 = time.time()
dist, idx = NearestNeighbors(n_neighbors=a.k + 1, n_jobs=-1).fit(Xs).kneighbors(Xs)
print(f"neighbours: {time.time()-t0:.0f} s; median distance to nearest {np.median(dist[:, 1]):.3f}", flush=True)
res = D - np.median(D[idx[:, 1:]], axis=1)            # own delta minus the neighbours' median
flag = np.abs(res) > a.thr
print(f"flag: |delta - median of {a.k} neighbours| > {a.thr}")
for i in range(4):
    print(f"  {OUT[i]:4s} flagged {flag[:, i].sum():7d} ({100*flag[:, i].mean():.3f}%)  of which above: "
          f"{(res[flag[:, i], i] > 0).mean()*100:.0f}%   |residual| median all rows {np.median(np.abs(res[:, i])):.3f}")
anyf = flag.any(1)
print(f"  any output flagged: {anyf.sum()} ({100*anyf.mean():.3f}%)")
B = 40
print(f"flag rate (any output, %) along the file's row order, {B} blocks of {n//B} rows:")
print("  " + " ".join(f"{100*anyf[b].mean():.2f}" for b in np.array_split(np.arange(n), B)))
h = np.frombuffer(b"".join(hashlib.blake2b(r.tobytes(), digest_size=8).digest()
                           for r in np.ascontiguousarray(X)), dtype=np.uint64)
np.savez_compressed(a.out_prefix + ".npz", flag=flag, res=res.astype(np.float32), hash=h)
print("saved", a.out_prefix + ".npz")
if a.drop:
    np.save(a.drop, t[~anyf])
    print(f"wrote {a.drop}: {(~anyf).sum()} rows ({anyf.sum()} flagged rows dropped)")
