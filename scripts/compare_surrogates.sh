#!/bin/bash
#SBATCH --job-name=cmp_surrogates
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# CPU job, no GPU: the sampler test of tools/compare_surrogates.py.
#   site submit <site> likelihoods scripts/compare_surrogates.sh -- SHAPE NOISE POOL WAVES SEEDS
# e.g. -- real 0.03 2000 10,5,5,5,5,5,5,5,5 0,1,2
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
SHAPE="$1" NOISE="$2" POOL="$3" WAVES="$4" SEEDS="$5" python tools/compare_surrogates.py
