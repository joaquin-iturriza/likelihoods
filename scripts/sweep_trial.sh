#!/bin/bash
#SBATCH --job-name=sweep_trial
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# One trial of an initialised DyHPO sweep, submitted through the site tool:
#   site submit <site> likelihoods scripts/sweep_trial.sh -- <sweep_name> <trial_idx>
# The sweep is initialised first by sweep/generate_sweep.py (site run), which saves
# the resolved config to <SUBMIT_DIR>/sweeps/<sweep_name>/sweep_config.yaml.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
cfg="$SUBMIT_DIR/sweeps/$1/sweep_config.yaml"
test -f "$cfg" || { echo "no $cfg: initialise the sweep with sweep/generate_sweep.py first"; exit 1; }
python sweep/run_trial.py --sweep-config "$cfg" --trial-idx "$2"
