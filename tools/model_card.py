"""Print what a paper needs to state about each ONNX model: architecture, training
settings, preprocessing and metadata keys, read from the file itself.

    python tools/model_card.py MODEL.onnx [MODEL.onnx ...]
"""
import json
import sys

import onnx
import yaml


def _get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


TRAINING_KEYS = [
    "data.dataset", "data.train_test_val", "data.trafos", "data.nLL_trafos", "data.asinh_scale",
    "model._target_", "model.net._target_", "model.net.hidden_channels", "model.net.hidden_layers",
    "model.net.activation", "model.net.dropout_prob", "model.net.use_batchnorm",
    "training.loss", "training.output_weights", "training.iterations", "training.batchsize",
    "training.optimizer", "training.lr", "training.weight_decay", "training.regularization",
    "training.regularization_lambda", "training.scheduler", "training.cosanneal_warmup_frac",
    "training.cosanneal_warmup_steps", "training.cosanneal_eta_min", "training.clip_grad_norm",
    "training.validate_every_n_steps", "training.es_load_best_model", "training.dtype",
]


def card(path):
    md = {p.key: p.value for p in onnx.load(path).metadata_props}
    cfg = yaml.safe_load(md.get("run_config", "")) or {}
    print(f"=== {path}")
    for k in TRAINING_KEYS:
        v = _get(cfg, k)
        if v is not None:
            print(f"  {k}: {v}")
    for k in ("preprocessing", "model_name", "training_date", "training_duration", "removeCRsVRs"):
        if k in md:
            print(f"  meta {k}: {md[k][:400]}")
    std = json.loads(md["standardization"]) if "standardization" in md else {}
    print(f"  meta standardization keys: {sorted(std)}")
    print(f"  meta keys: {sorted(md)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        card(p)
