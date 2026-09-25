"""
Republish an existing ONNX model with clean OLLL v0.1 metadata.

    python tools/olll_publish.py IN.onnx OUT.onnx --analysis ATLAS-SUSY-2018-16 \
        --label EWKinos --stats stats/<dataset>.json

--stats is the JSON printed by tools/olll_training_stats.py for the model's
training dataset (it runs where the data lives; see that file).

Nothing is taken on trust from the input file's metadata:
  1. Every (run_config, standardization) pairing present in the file is run
     through the network on held-out rows of the training dataset; the one
     that reproduces the truth is the one that belongs to these weights, and it
     must win clearly. A file whose metadata cannot reproduce its own data is
     refused.
  2. The architecture in that run_config must match the weight shapes in the
     graph.
  3. Bounds and mu=0 baselines come from the training data (--stats); stored
     baselines that disagree are reported.
  4. Metadata is rebuilt by olll_metadata.build_metadata; the graph must come
     out byte-identical.
  5. The written file is decoded again from its own metadata alone and must
     pass the same closure test, then S. Kraml's validator (tools/olll_validate.py)
     with no errors.
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as ort
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import olll_metadata as om  # noqa: E402
from preprocessing_nnAdapter import preprocess_features, undo_preprocess_nLLs  # noqa: E402

# closure tolerance: median relative error on nLL(mu=1) over held-out rows
CLOSURE_TOL = 1e-2
# a competing metadata pairing must be at least this much worse to rule it out
WIN_FACTOR = 5.0
MU0_TOL = 1e-3

REFERENCE_KEYS = ["channels", "bkgfiles", "patchsets", "obs_yields", "bkg_yields",
                  "bkg_unc", "removeCRsVRs", "remove_channels", "analysis_altname",
                  "analysis"] + om.MAX_KEYS + om.GENERATION_KEYS + list(om.LEGACY_SPELLINGS)


def multimap(model):
    out = OrderedDict()
    for p in model.metadata_props:
        out.setdefault(p.key.replace("rafal::", ""), []).append(p.value)
    return out


def predict_nll1(sess, cfg, std, mu0, X):
    """nLL(mu=1) for rows X, decoding with (cfg, std, mu0) the way the adapters do."""
    d = cfg["data"]
    Xp, _, _ = preprocess_features(X, trafos=d["trafos"],
                                   mean=np.asarray(std["features_mean"][0]),
                                   std=np.asarray(std["features_std"][0]))
    out = sess.run(None, {"features": Xp.astype(np.float32)})[0]
    deltas = np.array([undo_preprocess_nLLs(o[:4].astype(np.float64),
                                            mean=np.asarray(std["nLLs_mean"][0]),
                                            std=np.asarray(std["nLLs_std"][0]),
                                            trafos=d["nLL_trafos"]) for o in out])
    return np.asarray(mu0)[None, :] + deltas


def closure(sess, cfg, std, mu0, sample):
    X, T = sample[:, :-8], sample[:, -8:]
    t1 = T[:, 1::2]
    p1 = predict_nll1(sess, cfg, std, mu0, X)
    return np.median(np.abs(p1 - t1) / np.abs(t1), axis=0)


def graph_architecture(model):
    """(n_inputs, width, depth, n_outputs) from the Gemm weight shapes."""
    shapes = [tuple(i.dims) for i in model.graph.initializer if len(i.dims) == 2]
    widths = [s[0] for s in shapes[:-1]]
    assert len(set(widths)) == 1, f"non-uniform hidden widths {shapes}"
    return shapes[0][1], widths[0], len(shapes) - 1, shapes[-1][0]


def duration_hms(text):
    """'1 hours 1 minutes 8 seconds' -> '01:01:08'."""
    m = re.match(r"^\s*(\d+) hours (\d+) minutes (\d+) seconds\s*$", str(text).strip('"'))
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3)):02d}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--analysis", required=True, choices=sorted(om.ANALYSES))
    ap.add_argument("--label", default=None, help="model within the analysis, e.g. EWKinos")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--training-date", default=None,
                    help="YYYY-MM-DD, for runs whose run_config has no timestamp (read it from the run log)")
    ap.add_argument("--training-duration", default=None, help="HH:MM:SS, likewise")
    ap.add_argument("--drop-generation-key", action="append", default=[],
                    help="a recorded generation setting that does not hold for the training "
                         "dataset (e.g. a filter record for a filter it never went through)")
    ap.add_argument("--filtering", default=None,
                    help="how the training dataset was derived from the generated scan "
                         "(our filtering); omit if used as generated")
    args = ap.parse_args()

    model = onnx.load(args.inp)
    graph_before = model.graph.SerializeToString()
    meta = multimap(model)
    stats = json.load(open(args.stats))
    sample = np.asarray(stats["sample"], dtype=np.float64)
    sess = ort.InferenceSession(args.inp)
    log = []

    # -- reference (statistical-model) fields: must be unambiguous as they are
    reference = {}
    for key in REFERENCE_KEYS:
        vals = set(meta.get(key, []))
        assert len(vals) <= 1, f"{key}: {len(vals)} conflicting reference values"
        if vals:
            reference[key] = json.loads(vals.pop())
    alt = reference.get("analysis_altname")
    assert alt in (None, args.analysis), f"file says {alt}, --analysis says {args.analysis}"
    if "analysis" in reference:
        assert reference["analysis"] == om.ANALYSES[args.analysis]["arxiv"], reference["analysis"]

    # -- which run_config/standardization belong to these weights
    rcs, sts = meta.get("run_config", []), meta.get("standardization", [])
    mu0 = stats["mu0"]
    scored = []
    for i, rc in enumerate(rcs):
        cfg = yaml.safe_load(rc)
        for j, st in enumerate(sts):
            err = closure(sess, cfg, json.loads(st), mu0, sample)
            scored.append((float(np.max(err)), i, j, err))
            log.append(f"closure run_config#{i} x standardization#{j}: "
                       f"median rel err nLL(mu=1) {np.round(err, 6).tolist()}")
    scored.sort(key=lambda s: s[0])
    best, i_rc, i_st, best_err = scored[0]
    assert best < CLOSURE_TOL, f"no metadata pairing reproduces the data (best {best:.3g})"
    distinct_rivals = [s for s in scored[1:] if not (rcs[s[1]] == rcs[i_rc] and sts[s[2]] == sts[i_st])]
    for s in distinct_rivals:
        assert s[0] > WIN_FACTOR * best, f"ambiguous: pairing {s[1:3]} scores {s[0]:.3g} vs best {best:.3g}"
    run_config, std = rcs[i_rc], json.loads(sts[i_st])
    cfg = yaml.safe_load(run_config)
    if len(scored) > 1:
        log.append(f"chose run_config#{i_rc} x standardization#{i_st} "
                   f"(run_dir {cfg.get('run_dir') or cfg.get('run_name')})")

    # -- the architecture described must be the one in the graph
    n_in, width, depth, n_out = graph_architecture(model)
    net = cfg["model"]["net"]
    assert (n_in, width, depth, n_out) == (net["n_features"], net["hidden_channels"],
                                          net["hidden_layers"], 8), \
        f"graph {(n_in, width, depth, n_out)} vs run_config {net}"
    assert stats["n_features"] == n_in, (stats["n_features"], n_in)
    assert stats["dataset"] in cfg["data"]["dataset"], (stats["dataset"], cfg["data"]["dataset"])

    # -- baselines: data is authoritative; report disagreement with the file
    for key, val in zip(om.MU0_KEYS, mu0):
        stored = {float(v) for v in meta.get(key, [])}
        if any(abs(s - val) > MU0_TOL for s in stored):
            log.append(f"NOTE {key}: stored {sorted(stored)} -> {val} (from training data)")

    # -- identity: only what the chosen run can vouch for
    date = om.training_date_from_run_name(cfg.get("run_name"))
    duration = None
    if date:
        for d_val, dur in zip(meta.get("training_date", []), meta.get("training_duration", [])):
            if d_val.strip('"').startswith(date):
                duration = duration_hms(dur)
    elif args.training_date:
        date, duration = args.training_date, args.training_duration
        log.append(f"training_date/duration from the command line: {date} {duration}")
    else:
        log.append("training_date/duration omitted: the chosen run_config has no run timestamp")
    tag = f"_{args.label}" if args.label else ""
    model_name = f"{args.analysis}{tag}_MuMLP-{depth}x{width}"

    bounds = {k: stats[k] for k in ("x_min", "x_max", "y_min", "y_max")}
    if stats.get("n_sentinel_train_rows"):
        log.append(f"bounds exclude {stats['n_sentinel_train_rows']} failed-fit rows (|nLL|>=1e9)")

    generation = om.normalize_generation(reference)
    for k in args.drop_generation_key:
        assert k in generation, f"--drop-generation-key {k}: not in the file"
        log.append(f"dropped generation key {k}={json.dumps(generation.pop(k))} (does not hold for the training dataset)")
    training_dataset = {"name": stats["dataset"], "n_rows": stats["n_rows"],
                        "filtering": args.filtering or "none recorded in the training code"}
    new_meta = om.build_metadata(
        analysis_id=args.analysis, model_name=model_name, run_config=run_config,
        standardization=std, reference=reference, bounds=bounds, mu0=mu0,
        generation=generation, training_dataset=training_dataset,
        training_date=date, training_duration=duration)
    dropped = sorted(set(meta) - set(new_meta) - set(om.LEGACY_SPELLINGS))
    log.append(f"not carried over: {dropped}")
    om.write_metadata(model, new_meta)
    assert model.graph.SerializeToString() == graph_before, "graph changed"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    onnx.save(model, args.out)

    # -- the written file, decoded from its own metadata only
    written = onnx.load(args.out)
    wm = {p.key: p.value for p in written.metadata_props}
    err = closure(ort.InferenceSession(args.out), yaml.safe_load(wm["run_config"]),
                  json.loads(wm["standardization"]),
                  [float(wm[k]) for k in om.MU0_KEYS], sample)
    assert np.max(err) < CLOSURE_TOL, f"written file fails closure: {err}"
    log.append(f"written file closure: median rel err nLL(mu=1) {np.round(err, 6).tolist()}")

    import olll_validate
    report = olll_validate.Report(args.out)
    data, m2 = olll_validate.load_input(__import__("pathlib").Path(args.out), report)
    olll_validate.validate(data, report, model=m2)
    res = report.as_dict()
    log.append(f"olll_validate: {res['status']} ({res['errors']} errors, {res['warnings']} warnings)")
    for issue in res["issues"]:
        log.append(f"  [{issue['severity']}] {issue['code']}: {issue['message']}")

    print(f"== {args.out}")
    for line in log:
        print("  " + line)
    sys.exit(1 if res["errors"] else 0)


if __name__ == "__main__":
    main()
