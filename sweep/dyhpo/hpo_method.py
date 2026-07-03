# Vendored from https://github.com/releaunifreiburg/DyHPO (MIT licence)
# with DyHPOAlgorithmND subclass that accepts a fidelity_grid and converts
# budget level indices to a normalised budget vector before feeding to the surrogate.

import copy
import itertools
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import torch

from sweep.dyhpo.surrogate_models.dyhpo import DyHPO


class DyHPOAlgorithm:
    """Original DyHPO algorithm (scalar budget). Kept for reference."""

    def __init__(
        self,
        hp_candidates: np.ndarray,
        log_indicator: List,
        seed: int = 11,
        max_benchmark_epochs: int = 52,
        fantasize_step: int = 1,
        minimization: bool = True,
        total_budget: int = 500,
        device: str = None,
        dataset_name: str = 'unknown',
        output_path: str = '.',
        surrogate_config: dict = None,
        verbose: bool = False,
        n_startup: int = 1,
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.hp_candidates = hp_candidates
        self.log_indicator = log_indicator

        self.scaler = MinMaxScaler()
        self.hp_candidates = self._preprocess_hp_candidates()

        self.minimization = minimization
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        self.examples: Dict[int, List] = {}
        self.performances: Dict[int, List] = {}

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.max_benchmark_epochs = max_benchmark_epochs
        self.total_budget = total_budget
        self.fantasize_step = fantasize_step
        self.nr_features = self.hp_candidates.shape[1]

        self.init_conf_indices = np.random.choice(
            self.hp_candidates.shape[0], n_startup, replace=False
        )
        self.init_budgets = [1] * n_startup
        self.fraction_random_configs = 0.1

        self.model = None
        self.initial_random_index = 0

        if surrogate_config is None:
            self.surrogate_config = {
                'nr_layers': 2,
                'nr_initial_features': self.nr_features,
                'layer1_units': 64,
                'layer2_units': 128,
                'cnn_nr_channels': 4,
                'cnn_kernel_size': 3,
                'batch_size': 64,
                'nr_epochs': 1000,
                'nr_patience_epochs': 10,
                'learning_rate': 0.001,
            }
        else:
            self.surrogate_config = surrogate_config

        self.best_value_observed = np.NINF
        self.diverged_configs: set = set()
        self.info_dict: Dict = {}
        self.suggest_time_duration = 0
        self.budget_spent = 0
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.no_improvement_threshold = int(max_benchmark_epochs + 0.2 * max_benchmark_epochs)
        self.no_improvement_patience = 0
        self.restart = True

    # ------------------------------------------------------------------
    # Budget preparation  (scalar version — overridden in 2D subclass)
    # ------------------------------------------------------------------

    def _budget_to_tensor(self, budgets_raw: np.ndarray) -> torch.Tensor:
        """Normalise integer budget levels to [0,1] scalar."""
        return torch.tensor(budgets_raw / self.max_benchmark_epochs, dtype=torch.float32)

    def _candidate_budget(self, b: int) -> float:
        """Budget value used when generating candidate configurations."""
        return float(b) / self.max_benchmark_epochs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dataset_and_budgets(self) -> Dict[str, torch.Tensor]:
        train_examples, train_labels, train_budgets, train_curves = self._history_configurations()

        train_examples = np.array(train_examples, dtype=np.single)
        train_labels   = np.array(train_labels,   dtype=np.single)
        train_budgets  = np.array(train_budgets,  dtype=np.single)
        train_curves   = self._patch_curves_to_same_length(train_curves)
        train_curves   = np.array(train_curves,   dtype=np.single)

        train_budgets_t = self._budget_to_tensor(train_budgets).to(self.dev)

        return {
            'X_train':      torch.tensor(train_examples).to(self.dev),
            'train_budgets': train_budgets_t,
            'train_curves':  torch.tensor(train_curves).to(self.dev),
            'y_train':       torch.tensor(train_labels).to(self.dev),
        }

    def _train_surrogate(self):
        data = self._prepare_dataset_and_budgets()
        self.model.train_pipeline(data, load_checkpoint=False)

    def _predict(self) -> Tuple[np.ndarray, np.ndarray, List, List]:
        configurations, hp_indices, budgets, learning_curves = self._generate_candidate_configurations()
        budgets = np.array(budgets, dtype=np.single)
        non_scaled_budgets = copy.deepcopy(budgets)

        budgets_t = self._budget_to_tensor(budgets).to(self.dev)

        configurations  = torch.tensor(np.array(configurations, dtype=np.single)).to(self.dev)
        learning_curves = self._patch_curves_to_same_length(learning_curves)
        learning_curves = torch.tensor(np.array(learning_curves, dtype=np.single)).to(self.dev)

        train_data = self._prepare_dataset_and_budgets()
        test_data  = {
            'X_test':       configurations,
            'test_budgets':  budgets_t,
            'test_curves':   learning_curves,
        }

        means, stds = self.model.predict_pipeline(train_data, test_data)
        return means, stds, hp_indices, non_scaled_budgets

    def suggest(self) -> Tuple[int, int]:
        suggest_time_start = time.time()

        if self.initial_random_index < len(self.init_conf_indices):
            idx    = self.init_conf_indices[self.initial_random_index]
            budget = self.init_budgets[self.initial_random_index]
            self.initial_random_index += 1
        else:
            means, stds, hp_indices, non_scaled_budgets = self._predict()
            best_pred_idx = self._find_suggested_config(means, stds, non_scaled_budgets)
            best_config_index = hp_indices[best_pred_idx]

            if best_config_index in self.examples:
                max_budget = max(self.examples[best_config_index])
                budget = min(max_budget + self.fantasize_step, self.max_benchmark_epochs)
            else:
                budget = self.fantasize_step

            idx = best_config_index

        self.suggest_time_duration = time.time() - suggest_time_start
        self.budget_spent += self.fantasize_step

        if self.budget_spent > self.total_budget:
            raise RuntimeError(
                f"DyHPO total_budget ({self.total_budget}) exhausted after {self.budget_spent} evaluations."
            )

        return int(idx), int(budget)

    def observe(self, hp_index: int, b: int, learning_curve: np.ndarray, alg_time: Optional[float] = None):
        score = learning_curve[-1]

        if np.isnan(learning_curve).any():
            self.diverged_configs.add(hp_index)
            return

        observe_time_start = time.time()

        self.examples[hp_index]     = np.arange(1, b + 1).tolist()
        self.performances[hp_index] = list(learning_curve)

        if self.best_value_observed < score:
            self.best_value_observed = score
            self.no_improvement_patience = 0
        else:
            self.no_improvement_patience += 1

        observe_time_end = time.time()
        train_time_duration = 0

        if self.initial_random_index >= len(self.init_conf_indices):
            if self.model is None:
                self.model = DyHPO(
                    self.surrogate_config, self.dev,
                    self.dataset_name, self.output_path, self.seed,
                )
            if self.no_improvement_patience == self.no_improvement_threshold:
                self.model.restart = True

            t0 = time.time()
            self._train_surrogate()
            train_time_duration = time.time() - t0

    def _prepare_examples(self, hp_indices: List) -> List[np.ndarray]:
        return [self.hp_candidates[i] for i in hp_indices]

    def _generate_candidate_configurations(self) -> Tuple[List, List, List, List]:
        hp_indices, hp_budgets, learning_curves = [], [], []

        for hp_index in range(self.hp_candidates.shape[0]):
            if hp_index in self.diverged_configs:
                continue
            if hp_index in self.examples:
                max_budget = max(self.examples[hp_index])
                next_budget = max_budget + self.fantasize_step
                curve = list(self.performances[hp_index][:max_budget])
                diff = self.surrogate_config['cnn_kernel_size'] - len(curve)
                if diff > 0:
                    curve.extend([0.0] * diff)
            else:
                next_budget = self.fantasize_step
                curve = [0.0] * self.surrogate_config['cnn_kernel_size']

            if next_budget <= self.max_benchmark_epochs:
                hp_indices.append(hp_index)
                hp_budgets.append(next_budget)
                learning_curves.append(curve)

        configurations = self._prepare_examples(hp_indices)
        return configurations, hp_indices, hp_budgets, learning_curves

    def _history_configurations(self) -> Tuple[List, List, List, List]:
        train_examples, train_labels, train_budgets, train_curves = [], [], [], []
        ks = self.surrogate_config['cnn_kernel_size']

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.hp_candidates[hp_index]

            for budget, performance in zip(budgets, performances):
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performance)
                train_curve = list(performances[:budget - 1]) if budget > 1 else [0.0]
                diff = ks - len(train_curve)
                if diff > 0:
                    train_curve.extend([0.0] * diff)
                train_curves.append(train_curve)

        return train_examples, train_labels, train_budgets, train_curves

    def _acq(self, best_value, mean, std, acq_fc='ei'):
        if acq_fc == 'ei':
            if std == 0:
                return 0.0
            z = (mean - best_value) / std
            return (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
        raise NotImplementedError(acq_fc)

    def _find_suggested_config(self, mean_predictions, mean_stds, budgets) -> int:
        highest = np.NINF
        best_index = -1
        for i, (mean, std) in enumerate(zip(mean_predictions, mean_stds)):
            budget = int(budgets[i])
            best_val = self._calculate_fidelity_ymax(budget)
            acq_val  = self._acq(best_val, mean, std)
            if acq_val > highest:
                highest = acq_val
                best_index = i
        return best_index

    def _calculate_fidelity_ymax(self, fidelity: int) -> float:
        exact, lower = [], []
        for idx in self.examples:
            try:
                exact.append(self.performances[idx][fidelity - 1])
            except IndexError:
                lower.append(max(self.performances[idx]))
        if exact:
            return max(exact)
        return max(lower)

    def _preprocess_hp_candidates(self) -> np.ndarray:
        log_candidates = []
        for hp_candidate in self.hp_candidates:
            row = [math.log(v) if self.log_indicator[i] else v for i, v in enumerate(hp_candidate)]
            log_candidates.append(row)
        arr = np.array(log_candidates)
        return self.scaler.fit_transform(arr)

    @staticmethod
    def _patch_curves_to_same_length(curves):
        if not curves:
            return curves
        max_len = max(len(c) for c in curves)
        for c in curves:
            diff = max_len - len(c)
            if diff > 0:
                c.extend([0.0] * diff)
        return curves


# ---------------------------------------------------------------------------
# N+1 dimensional independent-axes DyHPO
# ---------------------------------------------------------------------------

class DyHPOAlgorithmND:
    """
    DyHPO with fully independent fidelity axes — one per dataset plus t_steps.

    Instead of a single integer budget level (lockstep), suggest() returns a
    ``combo`` tuple of 0-indexed level indices — one per axis.  The candidate
    space is the Cartesian product of all level ranges, so different axes may
    have different numbers of levels.

    Observations are stored as {hp_idx: {combo: neg_val_loss}} and contexts
    for the DeepSets encoder are built on-the-fly with leave-one-out for
    training points and full context for candidate predictions.

    Parameters
    ----------
    fidelity_grid : dict
        ``{'t_steps': [lvl1, ...]}``
    hp_candidates : np.ndarray   — already preprocessed (log + MinMax scaled)
    log_indicator : list[bool]
    seed, total_budget, output_path, dataset_name, n_startup : see DyHPOAlgorithm
    surrogate_config : dict  — passed to DyHPO; should contain budget_dim and obs_embed_dim
    """

    def __init__(
        self,
        fidelity_grid: dict,
        hp_candidates: np.ndarray,
        log_indicator: List,
        seed: int = 11,
        total_budget: int = 10_000,
        device: str = None,
        dataset_name: str = 'unknown',
        output_path: str = '.',
        surrogate_config: dict = None,
        n_startup: int = 10,
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.seed          = seed
        self.total_budget  = total_budget
        self.output_path   = output_path
        self.dataset_name  = dataset_name
        self.logger        = logging.getLogger(__name__)

        # HP candidates (pre-processed by DyHPOSampler before passing in)
        self.hp_candidates = hp_candidates
        self.log_indicator = log_indicator
        self.nr_features   = hp_candidates.shape[1]

        # Fidelity: t_steps only
        t_steps_sched       = fidelity_grid['t_steps']
        self.axis_schedules = [t_steps_sched]
        self.axis_fulls     = [float(t_steps_sched[-1])]
        self.budget_dim     = 1
        self.all_combos: List[tuple] = [(i,) for i in range(len(t_steps_sched))]
        # all_combos[0] = (0,) is always the cheapest combo

        # Startup: randomly pick n_startup HP indices, evaluate at cheapest combo
        self.init_conf_indices = np.random.choice(
            self.hp_candidates.shape[0], n_startup, replace=False
        )
        self.initial_random_index = 0

        # Core observation store
        self.observations:    Dict[int, Dict[tuple, float]] = {}  # {hp_idx: {combo: neg_vl}}
        self.diverged_configs: set = set()

        # Surrogate
        if surrogate_config is None:
            self.surrogate_config = {
                'nr_layers':          2,
                'nr_initial_features': self.nr_features,
                'layer1_units':       64,
                'layer2_units':       128,
                'obs_embed_dim':      16,
                'batch_size':         64,
                'nr_epochs':          1000,
                'nr_patience_epochs': 10,
                'learning_rate':      0.001,
                'budget_dim':         self.budget_dim,
            }
        else:
            self.surrogate_config = surrogate_config

        self.model = None  # created on first observe() after startup

        self.best_value_observed    = np.NINF
        self.budget_spent           = 0
        self.no_improvement_patience = 0
        self.no_improvement_threshold = 20
        self.suggest_time_duration  = 0
        self.info_dict: Dict       = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combo_to_normalized(self, combo: tuple) -> tuple:
        """Map combo of 0-indexed level indices to normalised (N+1,) float tuple."""
        return tuple(
            self.axis_schedules[i][lvl] / self.axis_fulls[i]
            for i, lvl in enumerate(combo)
        )

    def _build_context(self, hp_idx: int, exclude_combo=None) -> list:
        """
        Build DeepSets context for hp_idx: list of (norm_budget_tuple, val_loss) pairs.
        If exclude_combo is given, omit that observation (leave-one-out for training).
        """
        obs = self.observations.get(hp_idx, {})
        return [
            (self._combo_to_normalized(c), -neg_vl)
            for c, neg_vl in obs.items()
            if c != exclude_combo
        ]

    def _history_configurations(self) -> Tuple[List, List, List, List]:
        """Training data for surrogate — one row per observed (hp_idx, combo)."""
        examples, labels, budgets, contexts = [], [], [], []
        for hp_idx, obs_dict in self.observations.items():
            for combo, neg_vl in obs_dict.items():
                examples.append(self.hp_candidates[hp_idx])
                labels.append(neg_vl)
                budgets.append(list(self._combo_to_normalized(combo)))
                contexts.append(self._build_context(hp_idx, exclude_combo=combo))
        return examples, labels, budgets, contexts

    def _generate_candidate_configurations(self, exclude=None) -> Tuple[List, List, List, List, List]:
        """All unevaluated (hp_idx, combo) pairs — candidates for EI maximisation.

        Returns configs, hp_indices, budgets, contexts, combos  (5 parallel lists).
        """
        exclude = set(exclude) if exclude else set()
        configs, hp_indices, budgets, contexts, combos = [], [], [], [], []
        for hp_idx in range(len(self.hp_candidates)):
            if hp_idx in self.diverged_configs:
                continue
            if hp_idx in exclude:
                continue
            observed = self.observations.get(hp_idx, {})
            ctx = self._build_context(hp_idx)  # full context (no leave-one-out)
            for combo in self.all_combos:
                if combo not in observed:
                    configs.append(self.hp_candidates[hp_idx])
                    hp_indices.append(hp_idx)
                    budgets.append(list(self._combo_to_normalized(combo)))
                    contexts.append(ctx)
                    combos.append(combo)
        return configs, hp_indices, budgets, contexts, combos

    def _prepare_dataset_and_budgets(self) -> Dict:
        examples, labels, budgets, contexts = self._history_configurations()
        return {
            'X_train':        torch.tensor(np.array(examples, dtype=np.float32)).to(self.dev),
            'train_budgets':  torch.tensor(np.array(budgets,  dtype=np.float32)).to(self.dev),
            'train_contexts': contexts,
            'y_train':        torch.tensor(np.array(labels,   dtype=np.float32)).to(self.dev),
        }

    def _train_surrogate(self):
        data = self._prepare_dataset_and_budgets()
        self.model.train_pipeline(data, load_checkpoint=False)

    def _predict(self, exclude=None) -> Tuple[np.ndarray, np.ndarray, List, List]:
        configs, hp_indices, budgets, contexts, combos = self._generate_candidate_configurations(exclude=exclude)
        if not configs:
            return np.array([]), np.array([]), [], []

        budgets_t      = torch.tensor(np.array(budgets, dtype=np.float32)).to(self.dev)
        configurations = torch.tensor(np.array(configs, dtype=np.float32)).to(self.dev)

        train_data = self._prepare_dataset_and_budgets()
        test_data  = {
            'X_test':        configurations,
            'test_budgets':  budgets_t,
            'test_contexts': contexts,
        }

        means, stds = self.model.predict_pipeline(train_data, test_data)
        return means, stds, hp_indices, combos

    @staticmethod
    def _acq(best_value, mean, std, acq_fc='ei'):
        if acq_fc == 'ei':
            if std == 0:
                return 0.0
            z = (mean - best_value) / std
            return (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
        raise NotImplementedError(acq_fc)

    def _find_best_ei(self, means, stds) -> int:
        """Return index of candidate with highest EI vs global best."""
        best_i, best_ei = -1, np.NINF
        ymax = self.best_value_observed
        for i, (m, s) in enumerate(zip(means, stds)):
            ei = self._acq(ymax, m, s)
            if ei > best_ei:
                best_ei, best_i = ei, i
        return best_i

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, exclude=None) -> Tuple[int, tuple]:
        """
        Returns
        -------
        hp_idx : int    — index into hp_candidates
        combo  : tuple  — 0-indexed level indices, one per fidelity axis

        exclude : set, optional
            hp indices to skip (e.g. already in-flight jobs).  When all
            candidates are excluded the restriction is silently lifted so
            the sweep never deadlocks.
        """
        exclude = set(exclude) if exclude else set()
        suggest_time_start = time.time()

        idx, combo = None, None

        # Startup: consume init_conf_indices, skipping excluded/diverged ones.
        while self.initial_random_index < len(self.init_conf_indices):
            candidate = int(self.init_conf_indices[self.initial_random_index])
            self.initial_random_index += 1
            if candidate not in exclude and candidate not in self.diverged_configs:
                idx, combo = candidate, self.all_combos[0]
                break

        if idx is None:
            if self.model is not None:
                means, stds, hp_indices, combos = self._predict(exclude=exclude)
                if hp_indices:
                    best_i = self._find_best_ei(means, stds)
                    idx    = hp_indices[best_i]
                    combo  = combos[best_i]

            if idx is None:
                # Random fallback: prefer non-excluded, non-diverged candidates.
                available = [i for i in range(len(self.hp_candidates))
                             if i not in self.diverged_configs and i not in exclude]
                if not available:
                    # All candidates in-flight or diverged — lift the exclusion.
                    available = [i for i in range(len(self.hp_candidates))
                                 if i not in self.diverged_configs]
                if not available:
                    available = list(range(len(self.hp_candidates)))
                idx   = int(np.random.choice(available))
                combo = self.all_combos[0]

        self.suggest_time_duration = time.time() - suggest_time_start
        self.budget_spent += 1

        if self.budget_spent > self.total_budget:
            raise RuntimeError(
                f"DyHPO total_budget ({self.total_budget}) exhausted after {self.budget_spent} evals."
            )

        return int(idx), combo

    def observe(self, hp_idx: int, combo: tuple, neg_val_loss: float):
        """
        Record one completed evaluation.

        Parameters
        ----------
        hp_idx       : int    — same index returned by suggest()
        combo        : tuple  — same combo returned by suggest()
        neg_val_loss : float  — negated val_loss (DyHPO maximises internally)
        """
        if np.isnan(neg_val_loss):
            self.diverged_configs.add(hp_idx)
            return

        self.observations.setdefault(hp_idx, {})[combo] = neg_val_loss

        if neg_val_loss > self.best_value_observed:
            self.best_value_observed = neg_val_loss
            self.no_improvement_patience = 0
        else:
            self.no_improvement_patience += 1

        if self.initial_random_index >= len(self.init_conf_indices):
            if self.model is None:
                self.model = DyHPO(
                    self.surrogate_config, self.dev,
                    self.dataset_name, self.output_path, self.seed,
                )
            if self.no_improvement_patience == self.no_improvement_threshold:
                self.model.restart = True
            self._train_surrogate()


# Backward-compat alias
DyHPOAlgorithm2D = DyHPOAlgorithmND
