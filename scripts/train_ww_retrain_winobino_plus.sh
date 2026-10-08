#!/bin/bash
#SBATCH --job-name=ww_retrain_winobino_plus
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Wino/bino(+) offshell scan + WW's TChiWZoff points, trained exactly like the
# wino/bino(-) + WW retrain (config/ww_retrain/higgsino_tchiwzoff.yaml).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
test -f "data/2106.01676-winobino-plus-tchiwzoff-retrain_test.npy" || { echo "dataset missing on $CCORCH_SITE"; exit 1; }
python run.py --config-path config/ww_retrain --config-name winobino_plus_tchiwzoff
