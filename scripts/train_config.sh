#!/bin/bash
#SBATCH --job-name=train_config
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# One training run from a config file: scripts/train_config.sh <config-dir> <config-name>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
DIR="${1:?usage: train_config.sh <config-dir> <config-name>}"
CFG="${2:?usage: train_config.sh <config-dir> <config-name>}"
DS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['dataset'][0])" "$DIR/$CFG.yaml") || exit 1
test -f "data/$DS.npy" || { echo "dataset $DS missing on $CCORCH_SITE"; exit 1; }
python run.py --config-path "$DIR" --config-name "$CFG"
