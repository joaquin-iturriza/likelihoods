"""OLLL v0.1 file of a fine-tuned run (config/smscan_ft): the run's network with the
metadata of the published model it was fine-tuned from, updated where the fine-tune
changes it.

    python tools/olll_finetune_publish.py RUN_DIR REFERENCE.onnx OUT.onnx --label LABEL
        --filtering TEXT [--run-idx 1]

Kept from REFERENCE: source, statistical model, channels, yields, preprocessing,
standardization (the run must have pinned it: data.fixed_stats_onnx, checked), mu=0
baselines, *_max and the generation settings of the base scan.
Replaced: model_name, model_parameters, training_date/duration (run log), run_config
(the run's), training_dataset (the fine-tune set, with --filtering saying how it was
made), and x/y_min/max (union of REFERENCE's and the fine-tune training block's).

Written only if the file reproduces held-out rows of both the fine-tune test points and
the base model's own test block (median relative error on nLL(mu=1) below 1%), passes
tools/olll_validate.py and loads in hep_olll.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import olll_metadata as om  # noqa: E402
from export_onnx_from_run import export_graph, load_run  # noqa: E402
from olll_publish import CLOSURE_TOL, closure, graph_architecture  # noqa: E402
from olll_training_stats import training_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir"); p.add_argument("reference"); p.add_argument("out")
    p.add_argument("--label", required=True, help="model_name label, e.g. Onshell-WWdata")
    p.add_argument("--filtering", required=True, help="how the fine-tune set was built")
    p.add_argument("--run-idx", type=int, default=1)
    a = p.parse_args()
    device = torch.device("cpu")
    cfg, exp, idx = load_run(a.run_dir, a.run_idx, device)
    cfgd = OmegaConf.to_container(cfg, resolve=True)
    log = []

    ref = onnx.load(a.reference)
    md = OrderedDict((q.key, q.value) for q in ref.metadata_props)
    st = json.loads(md["standardization"])
    for mine, key in [(exp.prepd_mean_features[0], "features_mean"), (exp.prepd_std_features[0], "features_std"),
                      (exp.prepd_mean[0], "nLLs_mean"), (exp.prepd_std[0], "nLLs_std")]:
        d = np.abs(np.asarray(mine, dtype=float).ravel() - np.asarray(st[key], dtype=float).ravel()).max()
        assert d < 1e-9, f"{key} differs from the reference by {d}: normalisation was not pinned"
    ref_cfg = yaml.safe_load(md["run_config"])
    for k in ("trafos", "nLL_trafos"):
        assert cfgd["data"][k] == ref_cfg["data"][k], f"data.{k} differs from the reference"

    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "graph.onnx")
        export_graph(exp, cfg.model.net.n_features, device, g)
        model = onnx.load(g)
    assert graph_architecture(model) == graph_architecture(ref), \
        f"graph {graph_architecture(model)} vs reference {graph_architecture(ref)}"

    (dataset,) = cfgd["data"]["dataset"]
    data_dir = cfgd["data"]["data_path"]
    ft = training_stats(data_dir, dataset, None)
    assert ft["presplit"] is not None, f"{dataset}: the fine-tune set is expected on disk as train|val|test"
    base_name = json.loads(md["training_dataset"])["name"]
    base = training_stats(data_dir, base_name, ref_cfg["data"]["train_test_val"][0], ref_cfg["data"].get("subsample"))
    ref_mu0 = [float(md[k]) for k in om.MU0_KEYS]
    assert np.allclose(ft["mu0"], ref_mu0, atol=1e-3), f"mu=0 baselines: fine-tune data {ft['mu0']} vs {ref_mu0}"

    date, duration = om.run_log_times(os.path.join(a.run_dir, f"out_{idx}.log"))
    md["model_name"] = om.model_name(json.loads(md["source"])["analysis_id"], a.label,
                                     len(json.loads(md["model_parameters"])["hidden_layers"]),
                                     json.loads(md["model_parameters"])["hidden_layers"][0])
    md["model_parameters"] = json.dumps(om.model_parameters(cfgd))
    md["training_date"], md["training_duration"] = date, duration
    md["run_config"] = OmegaConf.to_yaml(cfg)
    for key, f in [("x_min", np.minimum), ("x_max", np.maximum), ("y_min", np.minimum), ("y_max", np.maximum)]:
        md[key] = json.dumps([float(v) for v in f(json.loads(md[key]), ft[key])])
    md["training_dataset"] = json.dumps(OrderedDict([("name", dataset), ("n_rows", ft["n_rows"]),
                                                     ("filtering", a.filtering)]))
    om.write_metadata(model, md)

    with tempfile.TemporaryDirectory() as tmp:
        cand = os.path.join(tmp, "model.onnx")
        onnx.save(model, cand)
        wm = {q.key: q.value for q in onnx.load(cand).metadata_props}
        sess = ort.InferenceSession(cand)
        for name, stats in [("fine-tune test points", ft), (f"base test block ({base_name})", base)]:
            err = closure(sess, yaml.safe_load(wm["run_config"]), json.loads(wm["standardization"]),
                          [float(wm[k]) for k in om.MU0_KEYS], np.asarray(stats["sample"], dtype=np.float64))
            log.append(f"closure, {name}, {len(stats['sample'])} rows: median rel err nLL(mu=1) "
                       f"{np.round(err, 6).tolist()}")
            if np.max(err) >= CLOSURE_TOL:
                sys.exit("\n".join(log) + f"\nthe file does not reproduce held-out rows (tolerance {CLOSURE_TOL})")

        import olll_validate
        from pathlib import Path
        report = olll_validate.Report(cand)
        data, m2 = olll_validate.load_input(Path(cand), report)
        olll_validate.validate(data, report, model=m2)
        res = report.as_dict()
        log.append(f"olll_validate: {res['status']} ({res['errors']} errors, {res['warnings']} warnings)")
        log.extend(f"  [{i['severity']}] {i['code']}: {i['message']}" for i in res["issues"])
        if res["errors"]:
            sys.exit("\n".join(log))

        from hep_olll.nnAdapter import NNAdapter
        try:
            import jsonschema  # noqa: F401
            check_meta = True
        except ImportError:
            check_meta = False
            log.append("hep_olll metadata validator skipped: jsonschema not installed")
        adapter = NNAdapter(cand, validate_metadata=check_meta)
        adapter.predict({s: 0.0 for s in adapter.srOrder})
        log.append(f"hep_olll: loads and predicts ({len(adapter.srOrder)} inputs)")

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        onnx.save(model, a.out)

    print(f"== {a.out} ({md['model_name']}, trained {date}, {duration})")
    for line in log:
        print("  " + line)


if __name__ == "__main__":
    main()
