#!/usr/bin/env python3
"""
Generate (and optionally submit) condor jobs that re-run evaluate+plot for
every completed trial in a sweep, without re-training.

Each job loads the saved best model checkpoint and calls evaluate() + plot(),
producing histograms.pdf (nLL distributions, sigma, pull) for that trial.

Usage:
    python generate_plot_jobs.py --sweep-name 1911_ewkinos_014
    python generate_plot_jobs.py --sweep-name 1911_ewkinos_014 --submit
"""

import argparse
import json
import os
import pickle
import stat
import subprocess
import sys
import yaml

AFS_SWEEPS = '/afs/cern.ch/user/j/joiturri/likelihoods/sweeps'
EOS_PROJECT = '/eos/user/j/joiturri/likelihoods'
PYTHON_BIN  = '/eos/user/j/joiturri/jitu/amplitude_DSI/amplitudes_env/bin/python'
PYTHON_ENV  = '/eos/user/j/joiturri/jitu/amplitude_DSI/amplitudes_env/bin/activate'


def fmt(v):
    if v is None:
        return 'null'
    if isinstance(v, float):
        return f'{v:.6e}'
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-name', required=True)
    parser.add_argument('--submit', action='store_true',
                        help='Submit jobs to condor after generating scripts')
    args = parser.parse_args()

    afs_dir = os.path.join(AFS_SWEEPS, args.sweep_name)
    run_script = os.path.join(EOS_PROJECT, 'run.py')

    with open(os.path.join(afs_dir, 'sweep_config.yaml')) as f:
        sweep_cfg = yaml.safe_load(f)

    with open(os.path.join(afs_dir, 'checkpoint_index.json')) as f:
        ckpt_index = json.load(f)

    with open(os.path.join(afs_dir, 'dyhpo_state.pkl'), 'rb') as f:
        dyhpo = pickle.load(f)

    plot_dir = os.path.join(afs_dir, 'plot_jobs')
    os.makedirs(plot_dir, exist_ok=True)

    cluster_cfg = sweep_cfg.get('cluster', {})
    gpus_min_mem = cluster_cfg.get('gpus_minimum_memory', 4000)
    requirements = cluster_cfg.get('requirements', '')
    priority = cluster_cfg.get('priority', 10)

    sh_paths = []
    for hp_idx_str, entry in ckpt_index.items():
        hp_idx  = int(hp_idx_str)
        run_dir = entry['run_dir']
        run_idx = entry['run_idx']   # index of the last saved checkpoint

        hp_params = dyhpo['candidates_raw'][hp_idx]

        cmd = [PYTHON_BIN, run_script]

        # Fixed params from sweep config (skip training.iterations — we set it below)
        skip = {'training.iterations'}
        for key, val in sweep_cfg.get('fixed_params', {}).items():
            if key not in skip:
                cmd.append(f'{key}={fmt(val)}')

        # HP params for this candidate
        for key, val in hp_params.items():
            cmd.append(f'{key}={fmt(val)}')

        # Plot-only overrides
        cmd += [
            'train=false',
            'evaluate=true',
            'plot=true',
            f'base_dir={EOS_PROJECT}',
            f'exp_name={args.sweep_name}',
            f'run_dir={run_dir}',
            f'warm_start_idx={run_idx}',
            f'warm_start_dir={run_dir}',
            'training.iterations=1',        # unused with train=false; satisfies config
            'training.is_dyhpo_run=false',
            'save_source=false',
        ]

        sh = os.path.join(plot_dir, f'plot_hp{hp_idx:04d}.sh')
        with open(sh, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'source {PYTHON_ENV}\n')
            f.write(' '.join(cmd) + '\n')
        os.chmod(sh, os.stat(sh).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        sh_paths.append((hp_idx, sh))

    # Condor submit file
    sub = os.path.join(plot_dir, 'plot_jobs.sub')
    with open(sub, 'w') as f:
        f.write(f'executable             = /bin/bash\n')
        f.write(f'request_GPUs           = 1\n')
        f.write(f'request_memory         = 16000\n')
        f.write(f'+JobFlavour            = "tomorrow"\n')
        if requirements:
            f.write(f'requirements           = {requirements}\n')
        if gpus_min_mem:
            f.write(f'+GPUs_Minimum_Memory   = {gpus_min_mem}\n')
        f.write(f'priority               = {priority}\n')
        f.write('\n')
        for hp_idx, sh in sh_paths:
            f.write(f'arguments = {sh}\n')
            f.write(f'output    = {plot_dir}/plot_hp{hp_idx:04d}.out\n')
            f.write(f'error     = {plot_dir}/plot_hp{hp_idx:04d}.err\n')
            f.write(f'log       = {plot_dir}/condor.log\n')
            f.write('queue\n\n')

    print(f'Generated {len(sh_paths)} plot jobs → {plot_dir}')
    print(f'Submit file: {sub}')

    if args.submit:
        result = subprocess.run(['condor_submit', sub])
        sys.exit(result.returncode)
    else:
        print('Run with --submit to submit to condor.')


if __name__ == '__main__':
    main()
