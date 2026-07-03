#!/bin/bash
cd /eos/user/j/joiturri/jitu/amplitude_DSI
source amplitudes_env/bin/activate
cd /eos/user/j/joiturri/likelihoods
python /eos/user/j/joiturri/likelihoods/run.py model=mlp \
    training.lr=2e-4 training.loss=MSE training.iterations=1000 \
    data.dataset=[1909.09226-leakage-10_] training.scheduler=CosineAnnealingLR training.save_preds_intermediate=true\
    model.net.hidden_channels=4096 model.net.hidden_layers=3\
    training.es_patience=100000000 training.save_intermediate=true exp_name=mlp_1909_test \