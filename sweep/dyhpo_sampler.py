#!/usr/bin/env python3
"""
dyhpo_sampler.py  —  DyHPO wrapper for the amplitude HPO sweep.

Wraps DyHPOAlgorithmND with:
  - Continuous HP search space → discrete pre-sampled candidate grid
  - 1D fidelity axis: t_steps only (full dataset is always used)
  - Persistent state (pickle) on AFS between HTCondor jobs

Typical usage in run_trial.py:

    with DyHPOSampler.locked(state_path, output_path) as sampler:
        hp_idx, hp_params, t_steps = sampler.suggest()

    # ... run training ...

    with DyHPOSampler.locked(state_path, output_path) as sampler:
        sampler.observe(hp_idx, t_steps, val_loss)
"""

import fcntl
import itertools
import math
import os
import pickle
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------------

def _sample_candidates(hp_space: List[dict], n: int, seed: int) -> List[dict]:
    """
    Draw n quasi-random (Sobol if scipy available, else uniform random)
    configurations from the HP search space.

    hp_space entry formats (same as sweep_config.yaml search_space):
        {'name': '...', 'type': 'float_log',   'low': ..., 'high': ...}
        {'name': '...', 'type': 'float_uniform','low': ..., 'high': ...}
        {'name': '...', 'type': 'int_uniform',  'low': ..., 'high': ...}
        {'name': '...', 'type': 'int_log',      'low': ..., 'high': ...}
        {'name': '...', 'type': 'categorical',  'choices': [...]}
    """
    rng = np.random.default_rng(seed)

    # Build a [0,1] sobol sample if scipy is available, else uniform
    n_cont = sum(1 for e in hp_space if e['type'] != 'categorical')
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=n_cont, scramble=True, seed=seed)
        unit_cube = sampler.random(n)       # shape (n, n_cont)
    except Exception:
        unit_cube = rng.random((n, n_cont))

    candidates = []
    for row in range(n):
        cfg = {}
        cont_idx = 0
        for entry in hp_space:
            name = entry['name']
            t    = entry['type']
            if t == 'float_log':
                lo, hi = math.log(entry['low']), math.log(entry['high'])
                cfg[name] = math.exp(lo + unit_cube[row, cont_idx] * (hi - lo))
                cont_idx += 1
            elif t == 'float_uniform':
                cfg[name] = entry['low'] + unit_cube[row, cont_idx] * (entry['high'] - entry['low'])
                cont_idx += 1
            elif t == 'int_uniform':
                cfg[name] = int(round(
                    entry['low'] + unit_cube[row, cont_idx] * (entry['high'] - entry['low'])
                ))
                cont_idx += 1
            elif t == 'int_log':
                lo, hi = math.log(entry['low']), math.log(entry['high'])
                cfg[name] = int(round(math.exp(lo + unit_cube[row, cont_idx] * (hi - lo))))
                cont_idx += 1
            elif t == 'categorical':
                choices = entry['choices']
                cfg[name] = choices[rng.integers(len(choices))]
            else:
                raise ValueError(f"Unknown search space type: {t!r}")
        candidates.append(cfg)
    return candidates


def _encode_candidates(hp_space: List[dict], candidates_raw: List[dict]):
    """
    Convert list-of-dicts to numpy array for DyHPO.

    Categorical HPs are label-encoded as integers (0, 1, 2, ...) using the
    order of ``choices`` in hp_space so the mapping is deterministic.

    Returns
    -------
    array : np.ndarray, shape (n, n_params)
    log_indicator : list[bool]  — True for params sampled in log scale
    """
    param_order   = [e['name'] for e in hp_space]
    log_indicator = [e['type'] in ('float_log', 'int_log') for e in hp_space]

    cat_encoders = {
        e['name']: {v: i for i, v in enumerate(e['choices'])}
        for e in hp_space if e['type'] == 'categorical'
    }

    rows = []
    for cfg in candidates_raw:
        row = []
        for name in param_order:
            val = cfg[name]
            if name in cat_encoders:
                val = cat_encoders[name][val]
            row.append(val)
        rows.append(row)
    return np.array(rows, dtype=float), log_indicator


# ---------------------------------------------------------------------------
# DyHPOSampler
# ---------------------------------------------------------------------------

class DyHPOSampler:
    """
    Manages a pool of pre-sampled HP candidates and drives DyHPOAlgorithmND.

    Parameters
    ----------
    hp_space : list[dict]
        search_space from sweep_config.yaml
    fidelity_grid : dict
        ``{'t_steps': [lvl1, ...]}``
    n_candidates : int
        Number of HP configurations to pre-sample (fixed for the sweep lifetime)
    seed : int
    output_path : str
        Directory for DyHPO surrogate checkpoints (on EOS)
    n_startup : int
        Number of random evaluations before the surrogate kicks in
    total_budget : int
        Max total evaluations before DyHPO raises
    """

    def __init__(
        self,
        hp_space: List[dict],
        fidelity_grid: dict,
        n_candidates: int = 300,
        seed: int = 42,
        output_path: str = '.',
        n_startup: int = 10,
        total_budget: int = 10_000,
    ):
        self.hp_space      = hp_space
        self.fidelity_grid = fidelity_grid
        self.n_candidates  = n_candidates
        self.seed          = seed

        # Pre-sample and encode candidates
        self.candidates_raw = _sample_candidates(hp_space, n_candidates, seed)
        self.candidates_array, self.log_indicator = _encode_candidates(hp_space, self.candidates_raw)

        budget_dim = 1

        surrogate_config = {
            'nr_layers':           2,
            'nr_initial_features': self.candidates_array.shape[1],
            'layer1_units':        64,
            'layer2_units':        128,
            'obs_embed_dim':       16,
            'batch_size':          64,
            'nr_epochs':           1000,
            'nr_patience_epochs':  10,
            'learning_rate':       0.001,
            'budget_dim':          budget_dim,
        }

        from sweep.dyhpo.hpo_method import DyHPOAlgorithmND
        self.algorithm = DyHPOAlgorithmND(
            fidelity_grid=fidelity_grid,
            hp_candidates=_preprocess_candidates(self.candidates_array, self.log_indicator),
            log_indicator=self.log_indicator,
            seed=seed,
            total_budget=total_budget,
            output_path=output_path,
            dataset_name='amplitude_sweep',
            surrogate_config=surrogate_config,
            n_startup=n_startup,
        )

        # Raw (not negated) val_loss history: {hp_idx: {combo: val_loss}}
        self._val_loss_history: Dict[int, Dict[tuple, float]] = {}
        # Per-dataset val_losses at the best checkpoint: {hp_idx: {combo: {ds_name: loss}}}
        self._proc_val_loss_history: Dict[int, Dict[tuple, dict]] = {}
        # Insertion-order list of (hp_idx, combo) for chronological history plot
        self._eval_order: List[tuple] = []
        # hp indices suggested but not yet observed (in-flight jobs)
        self._in_flight: set = set()

    # ------------------------------------------------------------------
    # Combo encode / decode helpers
    # ------------------------------------------------------------------

    def _to_combo(self, t_steps: int) -> tuple:
        """Convert t_steps to internal combo tuple."""
        return (self.fidelity_grid['t_steps'].index(t_steps),)

    def _from_combo(self, combo: tuple) -> int:
        """Convert internal combo tuple to t_steps."""
        return self.fidelity_grid['t_steps'][combo[0]]

    def best_predicted_per_combo(self) -> dict:
        """
        Use the surrogate to find the best predicted HP config for each fidelity combo.

        Returns {internal_combo: hp_idx} — the hp_idx with the highest predicted mean
        (lowest predicted val_loss) among all unevaluated (hp_idx, combo) pairs.
        Combos where every hp_idx has already been observed are absent from the result.
        Returns an empty dict if no surrogate model has been trained yet.
        """
        if self.algorithm.model is None:
            return {}

        means, _stds, hp_indices, combos = self.algorithm._predict()

        best = {}  # combo -> (hp_idx, mean)
        for mean, hp_idx, combo in zip(means, hp_indices, combos):
            if combo not in best or mean > best[combo][1]:
                best[combo] = (hp_idx, float(mean))

        return {combo: hp_idx for combo, (hp_idx, _) in best.items()}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def suggest(self, max_t_steps: int = None) -> Tuple[int, dict, int]:
        """
        Returns
        -------
        hp_idx    : int  — index into candidates_raw/candidates_array
        hp_params : dict — {param_name: value}
        t_steps   : int  — training steps for this evaluation

        max_t_steps : if set, clamp the suggested t_steps to the highest
            fidelity level that does not exceed this value.
        """
        hp_idx, combo = self.algorithm.suggest(exclude=self._in_flight)
        self._in_flight.add(hp_idx)
        t_steps = self._from_combo(combo)
        if max_t_steps is not None and t_steps > max_t_steps:
            allowed = [t for t in self.fidelity_grid['t_steps'] if t <= max_t_steps]
            t_steps = max(allowed) if allowed else self.fidelity_grid['t_steps'][0]
        return hp_idx, self.candidates_raw[hp_idx], t_steps

    def observe(self, hp_idx: int, t_steps: int, val_loss: float,
                proc_val_losses: Optional[dict] = None):
        """Record a completed evaluation."""
        self._in_flight.discard(hp_idx)
        combo = self._to_combo(t_steps)
        self._val_loss_history.setdefault(hp_idx, {})[combo] = val_loss
        if proc_val_losses:
            self._proc_val_loss_history.setdefault(hp_idx, {})[combo] = proc_val_losses
        self._eval_order.append((hp_idx, combo))
        self.algorithm.observe(hp_idx, combo, -val_loss)

    def report_failure(self, hp_idx: int):
        """
        Called when a trial fails before observe().  Removes hp_idx from the
        in-flight set and marks it diverged so it won't be re-suggested.
        """
        self._in_flight.discard(hp_idx)
        self.algorithm.diverged_configs.add(hp_idx)

    def best_result(self) -> Optional[Tuple[dict, float]]:
        """Return (hp_params, val_loss) for the best completed evaluation, or None."""
        if not self._val_loss_history:
            return None
        best_hp   = min(self._val_loss_history,
                        key=lambda i: min(self._val_loss_history[i].values()))
        best_loss = min(self._val_loss_history[best_hp].values())
        return self.candidates_raw[best_hp], best_loss

    def all_results(self) -> List[dict]:
        """
        Return a list of result dicts, one per completed (hp_idx, combo) pair,
        sorted by val_loss ascending.

        Each dict has keys: hp_idx, val_loss, t_steps, params,
        and optionally proc_val_losses {ds_name: val_loss} if available.
        """
        records = []
        for hp_idx, obs_dict in self._val_loss_history.items():
            for combo, val_loss in obs_dict.items():
                t_steps = self._from_combo(combo)
                rec = {
                    'hp_idx':   hp_idx,
                    'val_loss': val_loss,
                    't_steps':  t_steps,
                    'params':   self.candidates_raw[hp_idx],
                }
                proc = self._proc_val_loss_history.get(hp_idx, {}).get(combo)
                if proc:
                    rec['proc_val_losses'] = proc
                records.append(rec)
        return sorted(records, key=lambda r: r['val_loss'])

    def all_results_chronological(self) -> List[dict]:
        """
        Same as all_results() but in observation order (for history plots).
        Entries from _eval_order that haven't been observed yet are skipped.
        """
        records = []
        for hp_idx, combo in self._eval_order:
            val_loss = self._val_loss_history.get(hp_idx, {}).get(combo)
            if val_loss is None:
                continue
            t_steps = self._from_combo(combo)
            rec = {
                'hp_idx':   hp_idx,
                'val_loss': val_loss,
                't_steps':  t_steps,
                'params':   self.candidates_raw[hp_idx],
            }
            proc = self._proc_val_loss_history.get(hp_idx, {}).get(combo)
            if proc:
                rec['proc_val_losses'] = proc
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Persistence  (pickle on AFS with fcntl locking)
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Pickle the sampler state (NOT the surrogate model — that uses its own checkpoint)."""
        alg = self.algorithm
        state = {
            'hp_space':          self.hp_space,
            'fidelity_grid':     self.fidelity_grid,
            'n_candidates':      self.n_candidates,
            'seed':              self.seed,
            'candidates_raw':    self.candidates_raw,
            'candidates_array':  self.candidates_array,
            'log_indicator':     self.log_indicator,
            # DyHPOAlgorithmND internal state
            'observations':      alg.observations,
            'initial_random_index': alg.initial_random_index,
            'best_value_observed':  alg.best_value_observed,
            'diverged_configs':  alg.diverged_configs,
            'budget_spent':      alg.budget_spent,
            'no_improvement_patience': alg.no_improvement_patience,
            'init_conf_indices': alg.init_conf_indices,
            # Raw val_loss history
            'val_loss_history':      self._val_loss_history,
            'proc_val_loss_history': self._proc_val_loss_history,
            'eval_order':            self._eval_order,
            'in_flight':             self._in_flight,
        }
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, output_path: str = '.', force_cpu: bool = False) -> 'DyHPOSampler':
        """Restore a previously saved DyHPOSampler from a pickle file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        obj = cls.__new__(cls)
        obj.hp_space         = state['hp_space']
        obj.fidelity_grid    = state['fidelity_grid']
        obj.n_candidates     = state['n_candidates']
        obj.seed             = state['seed']
        obj.candidates_raw   = state['candidates_raw']
        obj.candidates_array = state['candidates_array']
        obj.log_indicator    = state['log_indicator']
        obj._val_loss_history      = state['val_loss_history']
        obj._proc_val_loss_history = state.get('proc_val_loss_history', {})
        obj._eval_order            = state.get('eval_order', [])
        obj._in_flight             = state.get('in_flight', set())

        budget_dim = 1

        surrogate_config = {
            'nr_layers':           2,
            'nr_initial_features': obj.candidates_array.shape[1],
            'layer1_units':        64,
            'layer2_units':        128,
            'obs_embed_dim':       16,
            'batch_size':          64,
            'nr_epochs':           1000,
            'nr_patience_epochs':  10,
            'learning_rate':       0.001,
            'budget_dim':          budget_dim,
        }

        from sweep.dyhpo.hpo_method import DyHPOAlgorithmND
        alg = DyHPOAlgorithmND.__new__(DyHPOAlgorithmND)

        import torch, logging
        alg.dev                     = torch.device('cpu') if force_cpu else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        alg.seed                    = obj.seed
        alg.total_budget            = 10_000
        alg.output_path             = output_path
        alg.dataset_name            = 'amplitude_sweep'
        alg.logger                  = logging.getLogger(__name__)
        alg.log_indicator           = obj.log_indicator
        alg.nr_features             = obj.candidates_array.shape[1]
        alg.hp_candidates           = _preprocess_candidates(obj.candidates_array, obj.log_indicator)
        alg.init_conf_indices       = state['init_conf_indices']
        alg.observations            = state['observations']
        alg.initial_random_index    = state['initial_random_index']
        alg.best_value_observed     = state['best_value_observed']
        alg.diverged_configs        = state['diverged_configs']
        alg.budget_spent            = state['budget_spent']
        alg.no_improvement_patience = state['no_improvement_patience']
        alg.no_improvement_threshold = 20
        alg.budget_spent            = state['budget_spent']
        alg.suggest_time_duration   = 0
        alg.info_dict               = {}
        alg.surrogate_config        = surrogate_config

        # Rebuild axis_schedules, axis_fulls, all_combos, budget_dim
        t_steps_sched      = obj.fidelity_grid['t_steps']
        alg.axis_schedules = [t_steps_sched]
        alg.axis_fulls     = [float(t_steps_sched[-1])]
        alg.budget_dim     = 1
        alg.all_combos     = [(i,) for i in range(len(t_steps_sched))]

        # Re-create surrogate model so it can load its own checkpoint.
        # Guard: only create model if there are actual observations — a corrupted state
        # (suggest called past startup but all training runs failed so observe never ran)
        # would cause encode_contexts([]) → empty TensorList crash.
        alg.model = None
        if alg.initial_random_index >= len(alg.init_conf_indices) and alg.observations:
            from sweep.dyhpo.surrogate_models.dyhpo import DyHPO
            alg.model = DyHPO(
                surrogate_config, alg.dev,
                'amplitude_sweep', output_path, obj.seed,
            )
            try:
                alg.model.load_checkpoint()
            except (FileNotFoundError, KeyError):
                pass

        obj.algorithm = alg
        return obj

    # ------------------------------------------------------------------
    # Locked context manager for concurrent HTCondor jobs
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def locked(state_path: str, output_path: str = '.'):
        """
        Acquire an exclusive file lock, load the sampler, yield it,
        then save and release.  Use for all suggest/observe calls.

        Example
        -------
        with DyHPOSampler.locked(state_path, output_path) as sampler:
            hp_idx, hp_params, t_steps = sampler.suggest()
        """
        lock_path = state_path + '.lock'
        with open(lock_path, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                sampler = DyHPOSampler.load(state_path, output_path)
                yield sampler
                sampler.save(state_path)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Helper: preprocess raw candidate array for DyHPOAlgorithmND
# ---------------------------------------------------------------------------

def _preprocess_candidates(array: np.ndarray, log_indicator: List[bool]) -> np.ndarray:
    """Apply log transform + MinMax scaling (same logic as DyHPOAlgorithm._preprocess_hp_candidates)."""
    from sklearn.preprocessing import MinMaxScaler
    import math
    result = array.astype(float).copy()
    for i, is_log in enumerate(log_indicator):
        if is_log:
            result[:, i] = np.log(result[:, i])
    scaler = MinMaxScaler()
    return scaler.fit_transform(result)
