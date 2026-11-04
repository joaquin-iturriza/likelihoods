#!/bin/bash
#SBATCH --job-name=smscan_ft
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# One simplified-model-scan fine-tune from config/smscan_ft/: scripts/train_smscan_ft.sh <config-name>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
CFG="${1:?usage: train_smscan_ft.sh <config-name>}"
DS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['dataset'][0])" "config/smscan_ft/$CFG.yaml") || exit 1
test -f "data/${DS}_test.npy" || { echo "dataset $DS missing on $CCORCH_SITE"; exit 1; }
python run.py --config-path config/smscan_ft --config-name "$CFG"
