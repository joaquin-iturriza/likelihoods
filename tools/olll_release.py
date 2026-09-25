"""
Publish the release models (runs/release_v2) with OLLL v0.1 metadata.

    python tools/olll_release.py [--only NAME ...]

Runs on the site holding the runs and the data (lxplus):
    site run lxplus likelihoods -- python tools/olll_release.py

For each model: IN = runs/release_v2/<name>/models/model_with_metadata.onnx,
OUT = models_onnx/olll_v0.1/<name>.onnx, statistics from
scratch/olll_stats/<dataset>.json (tools/olll_training_stats.py), training
date/duration from the run log, and statistical-model/generation metadata from
whichever of the generation pipeline's files for that analysis matches the
training data (tools/olll_publish.py --reference-onnx). Stops at the first
model that fails.
"""

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "release_v2")
OUT = os.path.join(ROOT, "models_onnx", "olll_v0.1")
STATS = os.path.join(ROOT, "scratch", "olll_stats")
GENERATION_MODELS = "/eos/home-j/joiturri/Instance1_fr/ML_LHClikelihoods/models"

CLEANED = "rows with any |delta nLL| < 1e-6 removed (data/clean_data.py)"
# the CSV as generated, converted row for row (data/conv_csv_npy.py)
NONE = ["--filtering", "none"]
OBS40 = "rows with delta nLL_obs > 40 removed from the filtered scan (filtering_applied)"

# name, analysis, label, training dataset, arXiv of the generation files, extra args
RELEASE = [
    ("SUSY-2018-04", "ATLAS-SUSY-2018-04", None,
     "1911.06660-leakage-10__cleaned", "1911.06660", ["--filtering", CLEANED]),
    ("SUSY-2018-16_EWkinos_obs40cut", "ATLAS-SUSY-2018-16", "EWKinos",
     "1911.12606-EWKinos-1M-z4-nll400-delta200-obs40cut", "1911.12606", ["--filtering", OBS40]),
    ("SUSY-2018-16_Sleptons", "ATLAS-SUSY-2018-16", "Sleptons",
     "1911.12606-sleptons-200k-fluct20_", "1911.12606", NONE),
    ("SUSY-2018-32", "ATLAS-SUSY-2018-32", None,
     "1908.08215-400k-fluct20_", "1908.08215", NONE),
    ("SUSY-2019-08", "ATLAS-SUSY-2019-08", None,
     "1909.09226-leakage-10_", "1909.09226", NONE),
    ("SUSY-2019-09_Onshell_Winobino", "ATLAS-SUSY-2019-09", "Onshell-WinoBino",
     "2106.01676-onshell-winobino-fluct25_-300k", "2106.01676",
     # the generation record's z=5 filter (268800 -> 268321) was not applied to
     # this dataset; ours removed only the two failed fits (268800 -> 268798,
     # checked against the original kept as .npy.bak)
     ["--drop-generation-key", "filtering_applied", "--drop-generation-key", "total_points",
      "--filtering", "2 rows with a failed observed fit removed "
                     "(nLL_obs(mu=1) = 1e10 placeholder): 268800 -> 268798"]),
    ("SUSY-2019-09_Offshell_Winobino_Plus", "ATLAS-SUSY-2019-09", "Offshell-WinoBino-plus",
     "2106.01676-offshell-winobino-plus-fluct20_-300k", "2106.01676", NONE),
    ("SUSY-2019-09_Offshell_Winobino_Minus_asinh", "ATLAS-SUSY-2019-09", "Offshell-WinoBino-minus",
     "2106.01676-offshell-winobino-minus-300k-fluct20_", "2106.01676", NONE),
    ("SUSY-2019-09_Offshell_Higgsinos", "ATLAS-SUSY-2019-09", "Offshell-Higgsino",
     "2106.01676-offshell-higgsino-300k-fluct20_", "2106.01676", NONE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for name, analysis, label, dataset, arxiv, extra in RELEASE:
        if args.only and name not in args.only:
            continue
        run = os.path.join(RUNS, name)
        refs = sorted(f for f in glob.glob(os.path.join(GENERATION_MODELS, f"{arxiv}-*", "*.onnx"))
                      if not os.path.basename(f).startswith("._"))
        cmd = [sys.executable, os.path.join(ROOT, "tools", "olll_publish.py"),
               os.path.join(run, "models", "model_with_metadata.onnx"),
               os.path.join(OUT, f"{name}.onnx"),
               "--analysis", analysis,
               "--stats", os.path.join(STATS, f"{dataset}.json"),
               "--run-log", os.path.join(run, "out_0.log")]
        if label:
            cmd += ["--label", label]
        for r in refs:
            cmd += ["--reference-onnx", r]
        cmd += extra
        rc = subprocess.call(cmd)
        if rc:
            sys.exit(f"{name}: olll_publish failed ({rc})")


if __name__ == "__main__":
    main()
