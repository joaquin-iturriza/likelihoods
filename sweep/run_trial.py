#!/usr/bin/env python3
"""
run_trial.py  —  Per-job entrypoint for DyHPO-based multi-fidelity HPO.

Each HTCondor job runs this script once. It:
  1. Locks the shared DyHPO state on AFS, calls sampler.suggest(), unlocks.
  2. Checks the checkpoint index to decide whether to warm-start from a previous
     fidelity level for this HP configuration.
  3. Runs `python run.py` with the suggested HP params and fidelity overrides.
  4. Locks the state again, calls sampler.observe() with the result, unlocks.
  5. Registers the new training checkpoint in the checkpoint index.

Fidelity dimension:
  - t_steps : training steps (training.iterations = increment from previous level)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Make sweep/ importable from the project root
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import subprocess
from sweep.dyhpo_sampler import DyHPOSampler
from sweep.checkpoint_index import CheckpointIndex


RETRY_ATTEMPTS = 3
RETRY_DELAY_S  = 30


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_hpo_objective(result: dict, scaling_law_cfg: dict) -> float | None:
    """
    Compute the geometric mean of per-dataset transfer ratios as the HPO objective.

    transfer_ratio_ds = val_loss_joint_ds / (C_ds · compute_ds^{-α_ds})

    Values < 1 mean joint training beats the solo scaling law prediction (positive
    transfer).  Returns None if scaling law params are unavailable.
    """
    proc_val_losses = result.get("proc_val_losses", {})
    compute_ds      = result.get("compute_ds", {})

    if not proc_val_losses or not compute_ds:
        return None

    # Load fitted params if available
    sl_params  = {}
    params_file = scaling_law_cfg.get("params_file")
    if params_file and os.path.exists(params_file):
        with open(params_file) as f:
            sl_params = json.load(f)

    alpha_priors = scaling_law_cfg.get("alpha_prior", {})

    ratios = []
    for ds_name, val_loss in proc_val_losses.items():
        compute = compute_ds.get(ds_name, 0)
        if compute <= 0 or val_loss <= 0:
            continue

        p     = sl_params.get(ds_name, {})
        alpha = p.get("alpha", alpha_priors.get(ds_name, 0.5))
        C     = p.get("C")   # None if scaling_law.py hasn't been run yet

        if C is None:
            # Without C we can't compute the transfer ratio — skip this dataset
            continue

        expected = C * (compute ** (-alpha))
        if expected > 0:
            ratios.append(val_loss / expected)

    if not ratios:
        return None

    return float(np.exp(np.mean(np.log(np.clip(ratios, 1e-10, None)))))


def format_value(v):
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def build_command(cfg, hp_params, run_dir, run_idx, result_path, t_steps,
                  warm_start_idx=None, increment_steps=None, warm_start_dir=None):
    project_dir = cfg["paths"]["project_dir"]
    run_script  = os.path.join(project_dir, "run.py")
    sweep_name  = cfg["sweep_name"]

    cmd = [sys.executable, run_script]

    # Fixed params (excluding training.iterations — set below)
    skip_keys = {"training.iterations"}
    for key, val in cfg.get("fixed_params", {}).items():
        if key not in skip_keys:
            cmd.append(f"{key}={format_value(val)}")

    # Suggested HP params
    for key, val in hp_params.items():
        cmd.append(f"{key}={format_value(val)}")

    cmd.append(f"training.iterations={increment_steps}")
    cmd.append(f"training.is_dyhpo_run=true")

    # Run dir and index for output organisation
    cmd.append(f"base_dir={project_dir}")
    cmd.append(f"run_dir={run_dir}")
    cmd.append(f"exp_name={sweep_name}")

    # Always pass increment_steps so scheduler logic has it explicitly at every fidelity level
    cmd.append(f"training.increment_steps={increment_steps}")

    if warm_start_idx is not None:
        # Resume from previous fidelity level checkpoint (or pretrain warm start)
        cmd.append(f"warm_start_idx={warm_start_idx}")
    else:
        # Cold start
        cmd.append("warm_start_idx=null")

    if warm_start_dir is not None:
        cmd.append(f"warm_start_dir={warm_start_dir}")

    cmd.append(f"training.result_path={result_path}")

    return cmd


def _write_summary(cfg, sampler, eos_dir):
    results = sampler.all_results()
    best    = sampler.best_result()

    lines = [
        f"Sweep: {cfg['sweep_name']}",
        f"Evaluations completed: {len(results)}",
    ]
    if best:
        params, loss = best
        lines += [f"Best val_loss: {loss:.6f}", "Best params:"]
        for k, v in params.items():
            lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("Top-10 results (across all fidelity levels):")
    for r in results[:10]:
        ps = "  ".join(
            f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
            for k, v in r['params'].items()
        )
        lines.append(
            f"  hp_{r['hp_idx']:04d}  val_loss={r['val_loss']:.6f}"
            f"  t_steps={r['t_steps']}  {ps}"
        )

    os.makedirs(eos_dir, exist_ok=True)
    with open(os.path.join(eos_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run one DyHPO trial on a compute node")
    parser.add_argument("--sweep-config",  required=True)
    parser.add_argument("--trial-idx",     type=int, required=True,
                        help="Submission-time index (for logging only)")
    parser.add_argument("--t-steps-cap",   type=int, default=None,
                        help="Cap the suggested t_steps to this value (for low-fidelity jobs)")
    parser.add_argument("--fixed-hp-idx",  type=int, default=None,
                        help="Skip DyHPO suggest and run this specific HP config")
    parser.add_argument("--fixed-t-steps", type=int, default=None,
                        help="Fixed t_steps for finalization jobs (requires --fixed-hp-idx)")
    args = parser.parse_args()

    cfg        = load_config(args.sweep_config)
    sweep_name = cfg["sweep_name"]
    project_dir= cfg["paths"]["project_dir"]

    afs_dir    = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos_dir    = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    results_dir= os.path.join(eos_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    state_path  = os.path.join(afs_dir, "dyhpo_state.pkl")
    ckpt_index_path = os.path.join(afs_dir, "checkpoint_index.json")

    print(f"[run_trial] Sweep: {sweep_name}  |  trial_idx: {args.trial_idx}")

    # ---------------------------------------------------------------
    # 1. Ask DyHPO for the next (hp_config, fidelity) to evaluate
    #    — or use fixed values for finalization jobs
    # ---------------------------------------------------------------
    eos_output_path = os.path.join(eos_dir, "dyhpo_surrogate")
    os.makedirs(eos_output_path, exist_ok=True)

    if args.fixed_hp_idx is not None:
        t_steps = args.fixed_t_steps
        with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
            hp_idx    = args.fixed_hp_idx
            hp_params = sampler.candidates_raw[hp_idx]
        print(f"[run_trial] FIXED hp_idx={hp_idx}  t_steps={t_steps}")
    else:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
                    hp_idx, hp_params, t_steps = sampler.suggest(max_t_steps=args.t_steps_cap)
                break
            except Exception as e:
                if attempt == RETRY_ATTEMPTS:
                    raise RuntimeError(f"Failed to get DyHPO suggestion after {RETRY_ATTEMPTS} attempts: {e}") from e
                print(f"[run_trial] suggest() failed (attempt {attempt}): {e}", file=sys.stderr)
                time.sleep(RETRY_DELAY_S)

    print(f"[run_trial] hp_idx={hp_idx}  t_steps={t_steps}")
    print(f"[run_trial] HP params: {hp_params}")

    # ---------------------------------------------------------------
    # 2. Check checkpoint index — resume or cold start?
    # ---------------------------------------------------------------
    with CheckpointIndex(ckpt_index_path) as idx:
        prev = idx.lookup(hp_idx)

    run_dir = os.path.join(project_dir, "runs", sweep_name, f"trial_{hp_idx:04d}")

    can_warm_start = prev is not None and prev['t_steps'] < t_steps

    warm_start_dir = None

    if can_warm_start:
        warm_start_idx  = prev['run_idx']
        prev_t_steps    = prev['t_steps']
        increment_steps = t_steps - prev_t_steps
        run_idx         = warm_start_idx + 1
        print(f"[run_trial] Resuming hp_{hp_idx:04d} from run_idx={warm_start_idx} "
              f"(t={prev_t_steps} → {t_steps}, +{increment_steps} steps)")
    else:
        increment_steps = t_steps
        pretrain = cfg.get("pretrain", {})
        if pretrain:
            # First visit to this HP config: warm start from a shared pretrained model
            warm_start_idx = pretrain['run_idx']
            run_idx        = warm_start_idx + 1
            warm_start_dir = pretrain['run_dir']
            print(f"[run_trial] Pretrain warm start hp_{hp_idx:04d} "
                  f"from {warm_start_dir} (run_idx={warm_start_idx})")
        else:
            warm_start_idx = None
            run_idx        = 0
            print(f"[run_trial] Cold start hp_{hp_idx:04d}")

    # Use t_steps for unique result filename (encodes fidelity level without listing all dataset sizes)
    result_path = os.path.join(results_dir, f"hp{hp_idx:04d}_t{t_steps}_{int(time.time())}.json")

    # ---------------------------------------------------------------
    # 3. Run training
    # ---------------------------------------------------------------
    cmd = build_command(
        cfg, hp_params, run_dir, run_idx, result_path,
        t_steps,
        warm_start_idx=warm_start_idx,
        increment_steps=increment_steps,
        warm_start_dir=warm_start_dir,
    )
    print(f"[run_trial] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, cwd=project_dir)
        if proc.returncode != 0:
            raise RuntimeError(f"run.py exited with code {proc.returncode}")

        with open(result_path) as f:
            result = json.load(f)
        val_loss        = float(result["val_loss"])
        val_mse         = result.get("val_mse")
        proc_val_losses = result.get("proc_val_losses")

        # HPO objective: pick the metric named by cfg["observe_metric"]
        # (default val_rel_err — physical-space relative error, which rewards
        # accuracy near |nLL|~0 and is comparable across preprocessings). Fall
        # back to val_mse then val_loss so older runs still rank.
        observe_metric = cfg.get("observe_metric", "val_rel_err")
        observe_val = result.get(observe_metric)
        if observe_val is None:
            observe_val = val_mse if val_mse is not None else val_loss
        observe_loss = float(observe_val)
        print(f"[run_trial] hp_{hp_idx:04d}  t_steps={t_steps}"
              f"  val_loss={val_loss:.6f}"
              + (f"  val_mse={val_mse:.6e}" if val_mse is not None else "")
              + (f"  {observe_metric}={observe_loss:.6e}"))

        # -----------------------------------------------------------
        # 4. Report result to DyHPO surrogate
        # -----------------------------------------------------------
        with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
            sampler.observe(hp_idx, t_steps, observe_loss, proc_val_losses)
            _write_summary(cfg, sampler, eos_dir)

        # -----------------------------------------------------------
        # 5. Register new checkpoint in index
        # -----------------------------------------------------------
        with CheckpointIndex(ckpt_index_path) as idx:
            idx.register(hp_idx, run_dir, run_idx, t_steps, t_steps,
                         trial_idx=args.trial_idx)

    except Exception as e:
        print(f"[run_trial] Trial FAILED: {e}", file=sys.stderr)
        if args.fixed_hp_idx is None:
            try:
                with DyHPOSampler.locked(state_path, eos_output_path) as sampler:
                    sampler.report_failure(hp_idx)
            except Exception as cleanup_err:
                print(f"[run_trial] report_failure: {cleanup_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
