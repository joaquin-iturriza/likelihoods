#!/bin/bash
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python $PROJECT_DIR/run.py model=mlp \
    training.lr=2e-4 training.loss=MSE training.iterations=10000000 \
    data.dataset=[1908.08215-400k-fluct20_] training.scheduler=CosineAnnealingLR training.save_preds_intermediate=true\
    model.net.hidden_channels=2048 model.net.hidden_layers=5\
    training.es_patience=100000000 training.save_intermediate=true exp_name=mlp_1908_2k_hn_5_hl \