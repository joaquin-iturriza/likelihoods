"""
Write the exact train/val/test splits the release models were trained on, and zip them.

    python tools/export_release_splits.py [--out-dir DIR]

Runs where the data is (lxplus: `site run lxplus likelihoods -- python
tools/export_release_splits.py`). For each model of tools/olll_release.py it
reproduces experiment.init_data/_init_dataloader for its run config: seed-1234
shuffle of the whole array, then train = first int(N*f_train) rows, val = the
next int(n_train * f_val/f_train), test = the rest ([train | val | test] with
train_test_val = [f_train, f_test, f_val]). Rows are raw: [yields... | 8 nLL
columns], as in data/.

Each train split is checked against the model it trained: re-fitting the
feature standardization on it (after the run's feature trafos) must reproduce
the features_mean/std stored in runs/release_v2/<name>/models/model_with_metadata.onnx
(fitted on the training block only). A split that does not is refused.

Writes <out-dir>/<dataset>_{train,val,test}.npy, <dataset>_columns.txt, a
README.txt, and <out-dir>.zip.
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np
import onnx
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from olll_release import RELEASE, RUNS  # noqa: E402
from preprocessing_nnAdapter import _log_with_negatives  # noqa: E402

NLL_COLS = ["nLL_exp_mu0", "nLL_exp_mu1", "nLL_obs_mu0", "nLL_obs_mu1",
            "nLLA_exp_mu0", "nLLA_exp_mu1", "nLLA_obs_mu0", "nLLA_obs_mu1"]


def split(cfg, data_dir):
    d = cfg["data"]
    ds = d["dataset"][0]
    assert not os.path.exists(os.path.join(data_dir, f"{ds}_val.npy")), \
        f"{ds}: on-disk split, not handled here"
    data = np.load(os.path.join(data_dir, f"{ds}.npy"), allow_pickle=True)
    np.random.seed(1234)
    np.random.shuffle(data)
    f_train, _, f_val = d["train_test_val"]
    n = len(data)
    n_train = int(n * f_train)
    if d.get("subsample") is not None:
        n_train = min(int(d["subsample"]), n_train)
    n_val = max(int(n_train * f_val / f_train), 1)
    return ds, data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def check_standardization(cfg, train, onnx_path):
    st = json.loads({p.key: p.value for p in onnx.load(onnx_path).metadata_props}["standardization"])
    x = train[:, :-8].astype(np.float64)
    for fns in (cfg["data"].get("trafos") or {}).values():
        for fn in fns:
            if fn == "log_w_negatives":
                x = _log_with_negatives(x)
            elif fn != "standardization":
                raise ValueError(f"unhandled feature trafo {fn}")
    mean, std = x.mean(axis=0), x.std(axis=0)
    dm = np.max(np.abs(mean - np.asarray(st["features_mean"][0])))
    ds = np.max(np.abs(std / np.asarray(st["features_std"][0]) - 1))
    return dm, ds


def columns(data_dir, ds, n_cols):
    csv = os.path.join(data_dir, f"{ds}.csv")
    if os.path.exists(csv):
        with open(csv) as f:
            head = f.readline().strip().split(",")
        if len(head) == n_cols and head[-8:] == NLL_COLS:
            return head
    return [f"yield_{i}" for i in range(n_cols - 8)] + NLL_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "scratch", "release_v2_splits"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    readme = ["Train/val/test splits used to train the release_v2 models (one per ATLAS analysis channel).",
              "Each .npy is float64 rows [yields... | 8 nLL columns]; column names in <dataset>_columns.txt.",
              "Split: seed-1234 shuffle, then [train | val | test]; train = int(N*f_train),",
              "val = int(n_train*f_val/f_train), test = the rest. Verified: the train split",
              "reproduces each model's stored feature standardization.", ""]
    written = []
    for name, *_ in RELEASE:
        run = os.path.join(RUNS, name)
        cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
        data_dir = cfg["data"]["data_path"]
        ds, tr, va, te = split(cfg, data_dir)
        dm, dsd = check_standardization(cfg, tr, os.path.join(run, "models", "model_with_metadata.onnx"))
        assert dm < 1e-6 and dsd < 1e-6, f"{name}: train split does not reproduce the model's standardization ({dm:.3g}, {dsd:.3g})"
        for tag, arr in (("train", tr), ("val", va), ("test", te)):
            p = os.path.join(args.out_dir, f"{ds}_{tag}.npy")
            np.save(p, arr)
            written.append(p)
        cp = os.path.join(args.out_dir, f"{ds}_columns.txt")
        with open(cp, "w") as f:
            f.write("\n".join(columns(data_dir, ds, tr.shape[1])) + "\n")
        written.append(cp)
        line = (f"{name}: {ds}  train {len(tr)}  val {len(va)}  test {len(te)}  "
                f"(train_test_val={cfg['data']['train_test_val']}; standardization check max |d mean| {dm:.1e}, max |std ratio - 1| {dsd:.1e})")
        readme.append(line)
        print(line, flush=True)
    rp = os.path.join(args.out_dir, "README.txt")
    with open(rp, "w") as f:
        f.write("\n".join(readme) + "\n")
    written.append(rp)
    zp = args.out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in written:
            z.write(p, arcname=os.path.join(os.path.basename(args.out_dir), os.path.basename(p)))
    print(f"wrote {zp} ({os.path.getsize(zp) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
