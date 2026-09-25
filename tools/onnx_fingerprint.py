"""
Identify ONNX models by their network, independent of metadata.

    python tools/onnx_fingerprint.py FILE_OR_DIR [...]

Prints one line per file: sha256 of the serialized graph (weights + ops, no
metadata), input/output shapes, the run_dir/run_name of every embedded
run_config, and model_author values. Two files with the same graph hash carry
the same network, whatever their metadata says.
"""

import glob
import hashlib
import os
import sys

import onnx
import yaml


def files(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(glob.glob(os.path.join(a, "**", "*.onnx"), recursive=True))
        else:
            yield a


def main():
    for f in files(sys.argv[1:]):
        try:
            m = onnx.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f}\tERROR {exc}")
            continue
        h = hashlib.sha256(m.graph.SerializeToString()).hexdigest()[:16]
        dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
        io = ",".join(f"{v.name}{dims(v)}" for v in list(m.graph.input) + list(m.graph.output))
        runs, authors = [], []
        for p in m.metadata_props:
            if p.key.endswith("run_config"):
                try:
                    c = yaml.safe_load(p.value)
                    runs.append(str(c.get("run_dir") or c.get("run_name")))
                except Exception:  # noqa: BLE001
                    runs.append("?")
            if p.key.endswith("model_author"):
                authors.append(p.value.strip('"'))
        print(f"{f}\t{h}\t{io}\truns={runs}\tauthors={authors}")


if __name__ == "__main__":
    main()
