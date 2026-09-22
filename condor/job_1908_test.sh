#!/bin/bash
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
env > /tmp/job_env_$(date +%s).txt 2>&1
python $PROJECT_DIR/run.py model=mlp training.lr=1e-05 training.loss=HETEROSC training.batchsize=2048 training.iterations=200                     data.dataset=[1908.08215-400k-fluct20_] training.scheduler=CosineAnnealingLR training.cosanneal_warmup_steps=4000 training.get_ID=true                     training.save_preds_intermediate=true model.net.hidden_channels=256 model.net.hidden_layers=5                     training.regularization=L2 training.regularization_lambda=0.0001                     training.es_patience=100000000 exp_name=TTTTTTTTTEEEEEEESTSAD
                            