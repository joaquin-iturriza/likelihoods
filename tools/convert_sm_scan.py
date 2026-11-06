"""Convert a simplified-model signal-strength scan (W. Waltenberger's CSVs: one row per
mass point and signal strength, SR yield columns, the 8 nLL columns, a trailing `mu`)
into this project's row format for one ONNX model: [model inputs in order | 8 nLL columns].

The yield columns hold c + mu*s per mass point, where the mu=0 value c is not part of
the likelihood the labels were computed with: the labels match background + mu*s in the
SRs and the observed counts (+ mu*s where the CSV has CR columns) in the CRs (checked
against the published models, which reproduce them to 1-3% where delta > 5). So each SR becomes bkg + (column - column at
mu=0) and each CR its observed count (+ the column minus its mu=0 value where the CSV
has CR columns). A header short of its rows (SR names missing, the columns otherwise in
model order) is repaired by inserting the missing SRs in model order.

    python tools/convert_sm_scan.py SCAN.csv MODEL.onnx OUT.npy [--max-delta 1000] [--in-box]

--in-box keeps only rows inside the model's training box (inputs within x_min/x_max,
deltas at most y_max; no lower bound on the deltas, the small-signal rows are the ones
wanted and their deltas sit at or just below the training minimum): a model with linearly standardised inputs is hundreds of
sigma out on the scan's large-mu rows, where its loss is not finite.

Also writes OUT_points.npy: the mass-point index of every row, for point-wise splits.
"""
import argparse
import csv
import json

import numpy as np
import onnx

NLL = ["nLL_exp_mu0", "nLL_exp_mu1", "nLL_obs_mu0", "nLL_obs_mu1",
       "nLLA_exp_mu0", "nLLA_exp_mu1", "nLLA_obs_mu0", "nLLA_obs_mu1"]


def convert(scan, model, max_delta, in_box=False):
    md = {p.key: p.value for p in onnx.load(model).metadata_props}
    bkg = dict(json.loads(md["bkg_yields"]))
    obs = dict(json.loads(md["obs_yields"]))
    channels = json.loads(md["channels"])
    if isinstance(channels, list):
        channels = {k: v for d in channels for k, v in d.items()}
    removed = set(json.loads(md.get("remove_channels") or "[]"))
    order = [n for n, _ in json.loads(md["bkg_yields"]) if n not in removed and n.rsplit("-", 1)[0] not in removed]
    if md.get("removeCRsVRs", "false").strip('"').lower() == "true":
        order = [n for n in order if channels.get(n.rsplit("-", 1)[0]) == "SR"]
    assert len(order) == len(json.loads(md["x_min"])), "input order does not match the model"

    rows = list(csv.reader(open(scan)))
    head, body = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    i = head.index("nLL_exp_mu0")
    assert head[i:i+8] == NLL and head[-1] == "mu", head[i:]
    names = [c.replace("[", "-").replace("]", "") for c in head[:i]]
    n_yields = body.shape[1] - 9
    srs = [n for n in order if channels.get(n.rsplit("-", 1)[0]) == "SR"]
    if len(names) < n_yields:
        missing = [n for n in srs if n not in names]
        assert len(missing) == n_yields - len(names), f"cannot repair header: {missing}"
        assert names == [n for n in order if n in names], "cannot repair header: names not in model order"
        names = [n for n in order if n in names or n in missing]
        print(f"header repaired: {missing} inserted in model order")
    crs = [n for n in names if channels.get(n.rsplit("-", 1)[0]) != "SR"]
    assert len(names) == n_yields and set(names) - set(crs) == set(srs) and set(crs) <= set(order), \
        "yield columns do not match the model"

    y, nll, mu = body[:, :n_yields], body[:, n_yields:n_yields+8], body[:, -1]
    first = np.maximum.accumulate(np.where(mu == 0, np.arange(len(mu)), 0))
    assert mu[0] == 0, "the scan must start at mu=0"
    signal = y - y[first]
    point = np.cumsum(mu == 0) - 1

    col = {n: k for k, n in enumerate(names)}
    for n in crs:  # CR columns, where given, are the observed count plus the signal leakage
        assert np.allclose(y[mu == 0, col[n]], obs[n]), f"{n}: CR column at mu=0 is not the observed count"
    base = {n: obs[n] if n in crs else bkg[n] for n in names}
    X = np.stack([base[n] + signal[:, col[n]] if n in col else np.full(len(y), obs[n]) for n in order], 1)
    delta = nll[:, 1::2] - nll[:, 0::2]
    keep = np.abs(delta).max(1) <= max_delta
    if in_box:
        box = (np.all((X >= json.loads(md["x_min"])) & (X <= json.loads(md["x_max"])), 1)
               & np.all(delta <= json.loads(md["y_max"]), 1))
        print(f"training box keeps {box.sum()} of {len(box)} rows "
              f"(by mu: {', '.join(f'{m:g}: {box[mu == m].sum()}/{(mu == m).sum()}' for m in np.unique(mu))})")
        keep &= box
    mu0 = np.array([float(json.loads(md[k])) for k in ("nLL_exp_mu0", "nLL_obs_mu0", "nLLA_exp_mu0", "nLLA_obs_mu0")])
    off = np.abs(nll[:, 0::2] - mu0).max(0)
    print(f"{len(y)} rows / {point.max()+1} points; |delta| <= {max_delta:g} keeps {keep.sum()}; "
          f"mu=0 baselines vs metadata max |diff| {np.round(off, 5).tolist()}")
    return np.c_[X, nll][keep], point[keep]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scan"); p.add_argument("model"); p.add_argument("out")
    p.add_argument("--max-delta", type=float, default=1e3)
    p.add_argument("--in-box", action="store_true", help="keep only rows inside the model's training box")
    a = p.parse_args()
    data, point = convert(a.scan, a.model, a.max_delta, a.in_box)
    np.save(a.out, data)
    np.save(a.out.replace(".npy", "_points.npy"), point)
    print("wrote", a.out, data.shape)
