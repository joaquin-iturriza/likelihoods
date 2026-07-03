#!/usr/bin/env python3
"""
checkpoint_index.py  —  Track training checkpoints for DyHPO resume.

Maps hp_idx (int) → the most recent training checkpoint for that configuration,
including the run_dir, run_idx, steps completed, and t_steps fidelity level.

The index is a JSON file on AFS (shared between HTCondor jobs). Use
CheckpointIndex as a context manager for atomic read-modify-write with
fcntl.LOCK_EX advisory locking:

    with CheckpointIndex(path) as idx:
        entry = idx.lookup(hp_idx)
        idx.register(hp_idx, run_dir, run_idx, steps_done, t_steps)
"""

import fcntl
import json
import os


class CheckpointIndex:
    """
    Persistent, file-locked registry of training checkpoints per hp_idx.

    Parameters
    ----------
    index_path : str
        Path to the JSON index file (on AFS for locking support).
    """

    def __init__(self, index_path: str):
        self.index_path = index_path
        self._data: dict | None = None
        self._fh = None

    # ------------------------------------------------------------------
    # Context manager — holds an exclusive lock for the duration
    # ------------------------------------------------------------------

    def __enter__(self):
        self._fh = open(self.index_path + '.lock', 'w')
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        self._load()
        return self

    def __exit__(self, *_):
        self._save()
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        self._fh = None
        self._data = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, hp_idx: int) -> dict | None:
        """
        Return checkpoint info for hp_idx, or None if not registered.

        Returns a dict with keys: run_dir, run_idx, steps_done, t_steps
        """
        return self._data.get(str(hp_idx))

    def register(
        self,
        hp_idx: int,
        run_dir: str,
        run_idx: int,
        steps_done: int,
        t_steps: int,
        trial_idx: int | None = None,
    ):
        """Register or update the checkpoint entry for hp_idx."""
        prev = self._data.get(str(hp_idx), {})
        trial_indices = list(prev.get('trial_indices', []))
        if trial_idx is not None and trial_idx not in trial_indices:
            trial_indices.append(trial_idx)
        self._data[str(hp_idx)] = {
            'run_dir':       run_dir,
            'run_idx':       run_idx,
            'steps_done':    steps_done,
            't_steps':       t_steps,
            'trial_indices': trial_indices,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self):
        tmp = self.index_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.index_path)
