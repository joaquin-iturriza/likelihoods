#!/usr/bin/env python3
"""Regenerate the nine released models from their own stored run_config.

Each published ONNX carries the fully resolved Hydra config of the run that
produced it. Replaying that config verbatim reproduces the model exactly, so the
only behavioural difference is the fix in experiment.init_data: the
standardisation statistics are now fitted on the training block instead of the
whole dataset.

Writes one Hydra config per model into config/release/, and one .sh/.sub pair
per model into the AFS Condor tree. Generating is free; submitting is not.
"""
import json
import os

import onnx
import yaml

EOS = "/eos/user/j/joiturri/likelihoods"
AFS = "/afs/cern.ch/user/j/joiturri/likelihoods"          # path as lxplus sees it
_afs_mnt = os.path.expanduser("~/mnt/afs/likelihoods")
# Same bytes either way: the sshfs mount when running locally, the real path
# when running on lxplus (where the mount does not exist).
AFS_LOCAL = _afs_mnt if os.path.isdir(_afs_mnt) else AFS
VENV = "/eos/user/j/joiturri/jitu/amplitude_DSI/amplitudes_env/bin/activate"
EXP_NAME = "release_v2"
FLAVOUR = "tomorrow"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(REPO, "config", "release")

def deep_merge(base, over):
    """base updated by over, recursively. Keys only in base survive."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


MODELS = [
    "SUSY-2018-04",
    "SUSY-2019-08",
    "SUSY-2018-32",
    "SUSY-2018-16_Sleptons",
    "SUSY-2018-16_EWkinos_obs40cut",
    "SUSY-2019-09_Onshell_Winobino",
    "SUSY-2019-09_Offshell_Winobino_Plus",
    "SUSY-2019-09_Offshell_Winobino_Minus_asinh",
    "SUSY-2019-09_Offshell_Higgsinos",
]


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    for d in ("jobs", "subs", "output", "error", "log"):
        os.makedirs(os.path.join(AFS_LOCAL, d), exist_ok=True)

    submits = []
    for name in MODELS:
        onnx_path = os.path.join(REPO, "models_onnx", f"{name}.onnx")
        m = onnx.load(onnx_path, load_external_data=False)
        md = {p.key: p.value for p in m.metadata_props}
        cfg = yaml.safe_load(md["run_config"])

        # The older runs predate several config fields the current code reads
        # (cosanneal_warmup_frac, mae_to_het_transition_frac, the DyHPO keys).
        # Replaying them verbatim raises ConfigAttributeError on the first
        # missing key, so merge each stored config over the current defaults:
        # every key the code expects exists, and the stored value always wins.
        defaults = yaml.safe_load(open(os.path.join(REPO, "config", "default.yaml")))
        defaults.pop("defaults", None)
        cfg = deep_merge(defaults, cfg)

        # Point the run at a fresh, predictable output directory and strip the
        # bookkeeping of the original run.
        cfg["exp_name"] = EXP_NAME
        cfg["run_name"] = name
        cfg["run_dir"] = None
        cfg["run_idx"] = 0
        cfg["base_dir"] = EOS
        cfg["jobid"] = None
        cfg["warm_start_idx"] = None
        cfg.pop("warm_start_dir", None)
        cfg["data"]["data_path"] = os.path.join(EOS, "data")
        for k in ("is_dyhpo_run", "result_path", "increment_steps"):
            cfg["training"].pop(k, None)
        cfg["training"]["save_intermediate"] = False
        cfg["training"]["save_preds_intermediate"] = False

        cfg_path = os.path.join(CFG_DIR, f"{name}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

        sh_path = os.path.join(AFS_LOCAL, "jobs", f"release_{name}.sh")
        with open(sh_path, "w") as f:
            f.write(f"""#!/bin/bash
set -e
cd {EOS}
test -f run.py || {{ echo "EOS not mounted"; exit 1; }}
source {VENV}
python run.py --config-path config/release --config-name {name}
""")
        os.chmod(sh_path, 0o755)

        sub_path = os.path.join(AFS_LOCAL, "subs", f"release_{name}.sub")
        jn = f"release_{name}"
        with open(sub_path, "w") as f:
            f.write(f"""executable            = {AFS}/jobs/{jn}.sh
arguments             = $(ClusterId) $(ProcId)
output                = {AFS}/output/{jn}.$(ClusterId).out
error                 = {AFS}/error/{jn}.$(ClusterId).err
log                   = {AFS}/log/{jn}.$(ClusterId).log
request_gpus          = 1
requirements          = !regexp("MIG", TARGET.GPUs_DeviceName)
+JobFlavour           = "{FLAVOUR}"
queue
""")
        submits.append(sub_path)
        it = cfg["training"]["iterations"]
        print(f"  {name:44s} steps={it:>7}  ds={cfg['data']['dataset'][0]}")

    print(f"\n{len(submits)} jobs written to {AFS}/subs/")
    print(f"configs written to {CFG_DIR}/")


if __name__ == "__main__":
    main()
