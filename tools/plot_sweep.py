#!/usr/bin/env python3
"""Re-generate plots for all completed trials in a sweep (no retraining).

Usage:
    python plot_sweep.py 1911_ewkinos_014
"""
import json, sys, os
# script lives in tools/; put the repo root on sys.path so experiment/misc import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from omegaconf import OmegaConf, open_dict
from experiment import nLLsExperiment
from misc import get_device

sweep_name = sys.argv[1]
afs_sweep  = f'/afs/cern.ch/user/j/joiturri/likelihoods/sweeps/{sweep_name}'

with open(f'{afs_sweep}/checkpoint_index.json') as f:
    ckpt_index = json.load(f)

device = get_device()

for hp_idx_str, entry in ckpt_index.items():
    run_dir = entry['run_dir']
    run_idx = entry['run_idx']

    cfg_path = os.path.join(run_dir, f'config_{run_idx}.yaml')
    if not os.path.exists(cfg_path):
        print(f'[skip] hp_{int(hp_idx_str):04d}: config not found at {cfg_path}')
        continue

    plot_path = os.path.join(run_dir, f'plots_{run_idx}')
    if os.path.exists(plot_path):
        print(f'[skip] hp_{int(hp_idx_str):04d}: plots already exist at {plot_path}')
        continue

    print(f'[plot] hp_{int(hp_idx_str):04d}  run_dir={run_dir}  run_idx={run_idx}')

    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.train         = False
        cfg.evaluate      = True
        cfg.plot          = True
        cfg.warm_start_idx = run_idx
        cfg.warm_start_dir = run_dir   # load from same trial, not pretrain dir
        cfg.run_dir        = run_dir
        cfg.save_source    = False

    exp = nLLsExperiment(cfg, device=device)
    exp()
