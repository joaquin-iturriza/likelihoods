#!/bin/bash
#SBATCH --job-name=ww_retrain
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# One offshell + WW retrain from config/ww_retrain/: scripts/train_ww_retrain.sh <config-name>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
CFG="${1:?usage: train_ww_retrain.sh <config-name>}"
DS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['dataset'][0])" "config/ww_retrain/$CFG.yaml") || exit 1
test -f "data/${DS}_test.npy" || { echo "dataset $DS missing on $CCORCH_SITE"; exit 1; }
python run.py --config-path config/ww_retrain --config-name "$CFG"
