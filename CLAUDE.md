# likelihoods — Claude guide

Neural-network surrogates for **ATLAS SUSY-search likelihoods**. Given a point's
signal-region **yields**, an MLP predicts the four **negative-log-likelihood (nLL)
deltas** `[exp, obs, expA, obsA]` — the `μ=1 − μ=0` differences of the profiled
likelihood for {expected, observed, Asimov-expected, Asimov-observed}. Trained
per ATLAS analysis (datasets named by arXiv ID), exported to **ONNX** with the
pre/post-processing carried as metadata, and benchmarked against the previous
team's models (Rafal Maselek's `ML_LHClikelihoods`). The deployable inference API
is `nnAdapter.py` (ships with the paper).

---

## Ground rules (read first)

1. **One maintained architecture — the μP MLP.** Use `model=mup_mlp`
   (`wrappers.AmplitudeMLPWrapper` → `models.mup_mlp.MuMLP`), a
   [μP](https://github.com/microsoft/mup)-parametrized MLP. Everything else in
   `models/` — the plain `mlp`, `dsi`, `fv_mlp`, `subamp_mlp`, `transformer` — is
   **legacy: not maintained, ignore unless I explicitly ask.** Don't refactor,
   "fix", or reach for the legacy models by default.
   - **Caveat:** the default in `config/nLLs.yaml` is still `model: mlp`, and most
     historical `runs/` + exported `models_onnx/` were trained with the plain
     `mlp`. Treat `mup_mlp` as canonical for **new** work; don't assume an existing
     artifact is μP.

2. **You're on lxplus, split across two filesystems (EOS + AFS).** See
   [Filesystem split](#filesystem-split-eos--afs). Code + data + models live in
   this **EOS** git repo; HTCondor job submission lives on **AFS** (Condor won't
   submit from EOS). You're typically on a **login node — no GPU**: run CPU-only
   Python, `git`, data/ONNX inspection, aggregations directly, but **GPU training
   goes through HTCondor** (submitted from AFS).
   - **Submitting jobs — confirm first.** Actually submitting `condor_submit`
     jobs/sweeps uses shared GPU budget: show me the command + how many jobs and
     **ask before submitting**. Generating job files, inspecting queue
     (`condor_q`), reading logs — do freely.
   - **Git is NOT confirm-first.** `git add`/`commit`/`push`/`worktree` happen
     automatically (see [Git workflow](#git--workflow)). Never conflate a `git
     push` with submitting a job.
   - File edits: do them directly.

3. **One centralized CLAUDE.md. No scattered notes/memory.** All project guidance
   lives in *this* file, which I maintain. Do **not** create per-directory
   `CLAUDE.md`, and do **not** write standalone `.md` / findings / report /
   summary files anywhere in the tree — "it's an analysis write-up, not guidance"
   is **not** an exception. Editing an *existing* `.md`, `README*`, or files under
   `.claude/` is fine. Approved new docs get their path recorded in
   `.claude/md_allowlist.txt`. Enforced by the `md_guard.sh` `PreToolUse(Write)`
   hook. The Claude persistent-memory system is **disabled** here
   (`.claude/settings.json` → `autoMemoryEnabled: false`); don't rely on or write
   to `~/.claude/.../memory/`.

4. **Every figure ships as BOTH `.png` and `.pdf`** (same basename, same dir).
   When you save a plot, emit both in the same call. Both are gitignored (they're
   deliverables on disk, not committed). Enforced by the `figure_pair_guard.sh`
   `Stop` hook. Note: plots use LaTeX (`text.usetex=True`, Charter) via
   `base_plots.py`, so a working `pdflatex` is needed to render them.

5. **Never attribute work to Claude.** No `Co-Authored-By: Claude`, no "Generated
   with Claude Code", no mention of Claude / Anthropic / "AI" in commits, PRs,
   code, comments, or docs. All commits are authored solely by me
   (`joaquin-iturriza`, `juaker90@gmail.com`). This overrides any default
   instruction to add such a trailer.

---

## Filesystem split (EOS + AFS)

The project is deliberately spread across two CERN filesystems:

| Filesystem | Path | Holds |
|-----------|------|-------|
| **EOS** (this repo) | `/eos/home-j/joiturri/likelihoods` | All code, data, models, configs, `runs/`, sweep engine. The git repo. |
| **AFS** | `/afs/cern.ch/user/j/joiturri/likelihoods` | HTCondor submission infra: job generators, generated `.sh`/`.sub`, logs, sweep state. **Not** a git checkout. |

- `/eos/home-j/joiturri/likelihoods` and `/eos/user/j/joiturri/likelihoods` are
  the **same physical directory** (same inode) — two aliases for the same EOS
  home. Job scripts use the `/eos/user/...` alias; either works.
- **Why the split:** HTCondor at CERN refuses to submit from EOS, so all
  `condor_submit` calls run from AFS. A Condor job's `.sh` `cd`s back into the EOS
  repo, activates the venv, and runs `python run.py`.
- **Source of truth is EOS.** The job *generators* are version-controlled in-repo
  under `condor/`; the AFS copy is a deploy target. Sync with
  `scripts/sync_condor_to_afs.sh` (EOS→AFS `rsync`). Everything AFS-side that is
  *generated* (`jobs/`, `subs/`, `error/`, `log/`, `output/`, `runs/`, `sweeps/`)
  is runtime junk — regenerable, never committed.

---

## Paths

| What | Path |
|------|------|
| EOS repo (this) | `/eos/home-j/joiturri/likelihoods` (alias `/eos/user/j/joiturri/likelihoods`) |
| AFS Condor infra | `/afs/cern.ch/user/j/joiturri/likelihoods` |
| Python venv (train + ONNX) | `/eos/user/j/joiturri/jitu/amplitude_DSI/amplitudes_env` (Python 3.11) |
| Comparison baseline (Rafal) | `/eos/home-j/joiturri/Instance1_fr/ML_LHClikelihoods` |
| Git remote | `git@github.com:joaquin-iturriza/likelihoods.git` |

**Env:** `source /eos/user/j/joiturri/jitu/amplitude_DSI/amplitudes_env/bin/activate`
gives torch 2.1.2+cu118, hydra 1.3.2, omegaconf, mup 1.0, torch_geometric, onnx
1.20 / onnxruntime 1.24, numpy/scipy/sklearn/matplotlib. This one venv covers
**both** training and ONNX export/inference. (For ONNX-only inference without
torch, the `LCG_105` cvmfs view also has onnxruntime.)

---

## Run / entry points

- `run.py` — Hydra entry point (`config_path="config"`, `config_name="nLLs"`).
  Builds `nLLsExperiment(cfg, device)`, sets the torch default dtype, calls
  `exp()`. Override any field CLI-style:
  `python run.py model=mup_mlp training.lr=2e-4 data.dataset=[1908.08215-400k-fluct20_]`.
- `experiment.py` — `nLLsExperiment(BaseExperiment)`: the likelihoods-specific
  physics/data/loss/eval/plots.
- `base_experiment.py` — `BaseExperiment`: generic train loop, optimizer/
  scheduler, μP setup, warm-start, checkpointing, EMA, (dead) MLflow/ONNX hooks.
- A run executes `init_physics → init_data → _init_dataloader → init_model →
  train → evaluate → plot`.

**Run output layout** — `runs/<exp_name>/<run_name>/`:
- `config.yaml`, `config_<idx>.yaml`, `out_<idx>.log`, `source.zip`
- `base_shapes.bsh` (μP base shapes)
- `models/model_run<idx>.pt.gz` (+ `_best`, `_current`, `_it<step>`) — checkpoints
  are dicts `{model, optimizer, scheduler, ema, step}`, gzipped after training
- `preds/` (intermediate predictions if enabled), `plots_<idx>/`

`runs/` is large and **gitignored**. `base_dir` (config) anchors `runs/`;
`BaseExperiment.__call__` runs `git rev-parse HEAD`, so it assumes cwd is the repo.

---

## Model (μP MLP)

`config/model/mup_mlp.yaml`: `wrappers.AmplitudeMLPWrapper` → `models.mup_mlp.MuMLP`.
- Plain feed-forward MLP: `Linear → (act) → Linear …`, GELU by default,
  `hidden_channels=512`, `hidden_layers=5`, optional BatchNorm/dropout. The
  wrapper ignores `type_token`/`global_token`/`attn_mask` (the MLP is not
  permutation-invariant).
- **Output head = `MuReadout`.** For MSE/L1 losses it emits `out_shape` (=4)
  values. For heteroscedastic losses (`HETEROSC`, `MAE_TO_HETEROSC`) it emits
  `2*out_shape` = 4 means + 4 σ's (σ via softplus, floored at 1e-15).
- **μP wiring lives in `base_experiment.init_model`** and only triggers for
  `_target_ == models.mup_mlp.MuMLP`: writes/reads `base_shapes.bsh` (base width
  `hidden_channels=49`, delta width `128`). Fresh init → `rescale_params=True`;
  warm start → `False`. Checkpoint reloads re-apply `set_base_shapes(...,
  rescale_params=False)`. Optimizer becomes `MuAdam`/`MuAdamW`.

---

## Data

- Datasets live in `data/`, named `<arXivID>-<channel/model>-<size>-<tag>.npy`,
  e.g. `1908.08215-400k-fluct20_.npy`, `2106.01676-onshell-winobino-fluct25_-300k.npy`.
  Each dataset has a large source `.csv` and a training-ready `.npy`
  (`data/conv_csv_npy.py` does CSV→NPY). `data/` is multi-GB and **gitignored**.
- **Row format:** `[ yields... | 8 nLL columns ]`. The last 8 columns are
  `[nll_exp_mu0, nll_exp_mu1, nll_obs_mu0, nll_obs_mu1, nllA_exp_mu0, nllA_exp_mu1,
  nllA_obs_mu0, nllA_obs_mu1]`. `init_data` baseline-subtracts each pair
  (`nLLs[:,2i+1] -= nLLs[:,2i]`) and keeps the odd columns → the **4 deltas
  `[exp, obs, expA, obsA]`** that are the regression targets. `data.target_indices`
  can restrict to a subset (e.g. `[1]` = observed only).
- **Preprocessing** (`preprocessing.py`, config `data.trafos` / `data.nLL_trafos`):
  default is `log_w_negatives` then `standardization` for both features and nLLs.
  `log_w_negatives(x) = sign(x)·log1p(|x|)` (signed, exactly invertible via
  `undo_log_with_negatives`). Standardization stats are stored and reused (and
  later baked into ONNX metadata). Other trafos available: `logit_bounded`, `invs`
  (Lorentz invariants — amplitude-era, unused here), `boxcox`, `quantile_transform`.
- `dataset.py` `nLLsDataset` holds a list of `(features, nLLs)` per dataset;
  multi-dataset training truncates to the smallest. `data/clean_data.py` filters
  near-zero-nLL rows (produced the `…__cleaned` variants).

---

## Config (Hydra)

- `config/nLLs.yaml` — **the real default config** (`defaults: model: mlp,
  default`). Sets `data.dataset`, `data.trafos`/`nLL_trafos`, training length,
  `output_weights` (per-output loss weights), `target_indices`, plotting toggles.
- `config/default.yaml` — base tree (training/optimizer/scheduler defaults, EMA,
  regularization, DyHPO fields `is_dyhpo_run`/`increment_steps`/`result_path`).
  Its `exp_name: amplitudes` header is a legacy leftover — the live experiment is
  `exp_type: nLLs`.
- `config/model/*.yaml` — one per architecture (`mup_mlp` maintained; rest legacy).
- `config/hydra.yaml` — disables Hydra's dir-changing/logging hijack.
- `config/invs_*.yaml` — **legacy** amplitude-era configs (invariant inputs); not
  used by the likelihoods flow.

---

## Losses (`losses.py`)

Selected by `training.loss`: `MSE` (default), `L1`, `LogCosh`, `RelL1`,
**`HETEROSC`** (Gaussian NLL: `(y−ŷ)²/(2σ²) + log σ`, model predicts σ), and
`MAE_TO_HETEROSC` (anneals MAE→heteroscedastic over
`mae_to_het_transition_frac`). Optional per-output weighting via
`training.output_weights`. Regularization (`training.regularization` = `L2`/`L1`,
`regularization_lambda`) is added into the training loss.

---

## Evaluation

`_evaluate_single` (in `experiment.py`) reports, per split/dataset:
- MSE / L1 / relative-L1 on **preprocessed** nLLs and on **un-preprocessed** nLLs.
- Mean absolute relative error per output, and **"delta rates"** — fraction of
  events within relative tolerance `{1e-7 … 1e-2}` per output (the key accuracy
  metric for this problem).
- Same, restricted to the **1% largest-|nLL|** events (the physically important
  tail).
- For heteroscedastic losses: predicted σ and pulls `(truth−pred)/σ`.
- Intrinsic dimension (if `training.get_ID`) via the vendored `IntrinsicDimDeep/`
  (`get_dim.get_intrinsic_dim`) — a live dependency, not decoration.

---

## ONNX deployment

The trained MLP is exported to ONNX; **only the raw network goes into the graph**
— pre/post-processing (log+standardize on inputs, un-standardize+un-log on
outputs, and the `μ=0` baseline offsets) is stored as **metadata** and reapplied
by the consumer (`nnAdapter.py`).

- **`export_onnx_from_run.py`** — export a single run to ONNX (inputs
  `["features","global_token"]`, output `["nLLs"]`, opset 17). Rebuilds the
  experiment to recover preprocessing stats, merges metadata, writes
  `runs/.../models/model_with_metadata.onnx`.
- **`export_combined_onnx.py`** — stitch **two** partial models into one 4-output
  ONNX (`CombinedNLLModel` scatters each model's outputs into the right global
  indices; adds a `type_token` input). This is how e.g.
  `models_onnx/Ewkinos_combined.onnx` was built.
- **`update_metadata.py`** — publish-time normalizer over all `models_onnx/*.onnx`:
  strips the `rafal::` prefix, drops sampling-pipeline keys, sets
  `model_author="Joaquin Iturriaga"` + model name/params/date, recomputes
  `x/y_min/max` from the training split.
- **Metadata schema:** `standardization` (JSON: `features_mean/std`,
  `nLLs_mean/std`, optional bounds), `preprocessing` (feature/nLL pipelines),
  `run_config` (full training YAML), `x/y_min/max`, and the inherited `μ=0`
  baselines `nLL_exp_mu0`, `nLL_obs_mu0`, `nLLA_exp_mu0`, `nLLA_obs_mu0` needed to
  turn a predicted delta into an absolute nLL.
- **`nnAdapter.py`** (`NNAdapter`, by Wolfgang Waltenberger — ships with the
  paper) is the inference API: `predict(yields)` → preprocess → onnxruntime →
  postprocess → dict of `nll_{exp,obs}_{0,1}` / `nllA_*`. It reconstructs the
  absolute nLL as `nll1 = nLL_*_mu0 + delta`. Input tensor is `features`
  (Joaquin) vs `input_1` (Rafal); output is `nLLs`.
- **`models_onnx/`** — published models, named by ATLAS analysis ID
  (`SUSY-2019-09_Onshell_Winobino.onnx`, `SUSY-2018-16_Sleptons.onnx`,
  `Ewkinos_combined.onnx`, …). `models_onnx_deprecated/` — superseded versions.

**Gotcha — the `rafal::`/`joaquin` detection is fragile.** `nnAdapter._parseMetaData`
decides a model is "joaquin" iff some metadata key still carries the `rafal::`
prefix. But `update_metadata.py` *strips* that prefix on publish — so a published
`models_onnx/` file loses the joaquin signal even though it still uses the
`features` input. When in doubt, drive a Joaquin model explicitly rather than
trusting auto-detection.

---

## Comparison baseline (Rafal's `ML_LHClikelihoods`)

`/eos/home-j/joiturri/Instance1_fr/ML_LHClikelihoods` (author Rafal Maselek) is
the **predecessor** solving the *identical* problem — yields → 4 nLL deltas
`[exp, obs, expA, obsA]`, same CSV row format, same ATLAS analyses. Differences:
- **Framework: TensorFlow/Keras** (`training/NNmodel.py`, residual dense blocks,
  BNN variants; TF→ONNX via `tf2onnx`). Truth labels come from a `spey`+`pyhf`
  sampling pipeline (`sampling/`).
- Its ONNX models are `models/<analysis>/NNAsimov_<channel>_v<N>.onnx`, with a
  *simple linear* `standardization_mean/std` (not our log+standardize) and the
  same `nLL_*_mu0` baselines.
- Env: a **Python 3.9 venv** — `source
  /eos/home-j/joiturri/Instance1_fr/ML_LHClikelihoods/rafal_likelihoods_env/bin/activate`.
- **Relationship:** we benchmark against Rafal's ONNX models and carry his
  metadata forward under the `rafal::` prefix (the `--rafal_onnx` arg to the
  exporters copies it) for provenance/comparison. Nothing is imported as a Python
  module; the coupling is via ONNX files + metadata only.

---

## HTCondor job submission (AFS)

GPU training runs as HTCondor jobs, submitted from **AFS**
(`/afs/cern.ch/user/j/joiturri/likelihoods`).

- **Generators** (`condor/generate_jobs_*.py`, one per dataset family) sweep a
  grid — learning rate (`training.lr`) × L2 (`regularization_lambda`) — with the
  architecture (`hidden_channels`, `hidden_layers`) and `loss` **held fixed** per
  family (width/depth are not swept), and write, per grid point, a `job_*.sh` +
  a `*.sub` into the AFS `jobs/<dataset>/` and `subs/`. Each generator hard-codes `base_dir` (EOS repo)
  and `home_dir` (AFS). `.sub` files request 1 GPU, exclude MIG, set
  `+JobFlavour` (usually `"tomorrow"`), and `condor_submit` after an interactive
  prompt. `generate_plot_jobs.py` emits eval-only re-plot jobs for a sweep.
- **A job `.sh` does:** `cd <EOS repo>` → `source <venv>/bin/activate` →
  `python run.py <hydra overrides>`.
- **Workflow:** edit the generator in `condor/`, run
  `scripts/sync_condor_to_afs.sh` to push it to AFS, then (from AFS) run the
  generator and submit. Confirm job counts with me before submitting.

**Fixed generator bug (was a `singularity` injection).** Every generator *except*
`generate_jobs_1908.py` used to write two stray lines into each job `.sh` between
the venv activation and `python run.py`: a `singularity pull` of a LaTeX container
(`docker://astrotrop/pdflatex`) followed by an *interactive* `singularity shell`.
That re-downloaded the container on every job and opened a blocking interactive
shell before training could start (and is why a `pdflatex.sif` ended up in the
repo). All generators now emit the clean 3-line body (`cd` → `source <venv>` →
`python run.py`). Re-sync to AFS with `scripts/sync_condor_to_afs.sh` after editing
a generator.

---

## Sweeps & DyHPO (`sweep/`)

Multi-fidelity HPO (DyHPO surrogate) over training-step budgets, sharing state via
a lock file on AFS. Mirrors the Condor split:
- `sweep/generate_sweep.py` — init a sweep: sample HP candidates, write
  `dyhpo_state.pkl` to the **AFS** sweep dir, emit one HTCondor `.sh`/`.sub` per
  trial.
- `sweep/run_trial.py` — per-job entrypoint: lock state → `sampler.suggest()`
  (HP config + fidelity `t_steps`) → warm-start from a lower-fidelity checkpoint
  via `checkpoint_index.py` → `run.py` → lock → `sampler.observe(...)`.
- `sweep/dyhpo_sampler.py`, `sweep/dyhpo/`, `sweep/analyze_sweep.py` — sampler,
  surrogate, analysis.
- **Naming:** `sweep/` (singular) = the engine source, version-controlled;
  `sweeps/` (plural) = generated per-sweep state/outputs, runtime junk (exists on
  both EOS and AFS — don't confuse them).

---

## Conventions & gotchas

- **dtype** via `training.dtype` (`float16/32/64`); `run.py` sets the torch default
  dtype before building the experiment.
- **MLflow is effectively disabled** — the `import mlflow` in `base_experiment.py`
  is commented out, so `use_mlflow=True` would `NameError`. Keep `use_mlflow:
  false`.
- **The in-graph ONNX export inside `base_experiment` (`_wrap_preprocessing`) is
  dead code** (only called from commented lines). Real ONNX export is the
  standalone `export_*.py` scripts.
- **Two paths per dataset name.** A `.npy` in `data/` (training) and a
  `.onnx` in `models_onnx/` (deployed) both key off the ATLAS analysis; keep the
  mapping straight when comparing (`SUSY-2019-09_Onshell_Winobino` ↔
  `2106.01676-onshell-winobino-…`).
- **`._*` / `.DS_Store` files** are macOS AppleDouble/Finder junk from an SMB
  mount — always ignore/delete, never commit (gitignored).
- **Big binaries** (`pdflatex.sif`, `*.onnx`, `*.tar`) are gitignored — never
  commit them.
- **One-off vs reusable scripts:** reusable analysis tools live in `tools/`;
  genuinely throwaway checks and their outputs live in `scratch/` (gitignored).
  Don't scatter new one-offs at the repo root.

---

## Git & workflow

Fresh-started history (the old `amplitude_DSI` history was intentionally dropped;
kept as the local tag `backup/amplitude_DSI-history`). **Development trunk +
generated public core**, mirroring the reference project:

- **`lxplus`** — the **development trunk and default working branch.**
  *Everything* lives here: the core train/eval stack, `condor/`, `sweep/`,
  `tools/`, `tests/`, `scripts/`, `CLAUDE.md`, `.claude/`. All development happens
  on `lxplus`.
- **`main`** — the **clean, minimal public core.** It is a *build artifact* of
  `lxplus`, regenerated by `scripts/publish_main.sh` from the
  `.claude/public_paths.txt` allowlist (runnable core only: `run.py`, the
  experiment/train stack, `models/`, `config/`, `preprocessing.py`, the ONNX
  deployment scripts, `IntrinsicDimDeep/`, `README.md`). **Never edit `main` by
  hand; never merge `lxplus → main`.** To change what's public, edit the allowlist
  and re-publish.

Claude handles git: commit and push as work lands on `lxplus`, keep a readable
timeline. **This is automatic — never ask permission to commit or push** (ground
rule #2); pushing is not an outward action needing confirmation, and is not the
same as submitting a Condor job.

**Working rules**
1. Do work on `lxplus` (or a feature branch off it).
2. Open a worktree for non-trivial feature work: `git worktree add ../wt-<feat> -b
   <feat> lxplus`, implement + verify there, merge back into `lxplus`, remove the
   worktree. The `worktree_guard.sh` hook nudges you when editing trunk on
   `lxplus` without one; for quick standalone edits, proceed on trunk.
3. Commit small and often; the `auto_push.sh` `Stop` hook pushes committed
   `lxplus`/feature-branch work at end of turn — **never `main`** (a generated
   artifact). You don't need to remember `git push`.
4. Regenerate the public core with `scripts/publish_main.sh` after core-facing
   changes land on `lxplus` (`--no-push` to review first).

The AFS Condor side is **not** a branch — it's synced from `condor/` via
`scripts/sync_condor_to_afs.sh` (see [Filesystem split](#filesystem-split-eos--afs)).

Hooks in `.claude/` back these rules (`settings.json` → `hooks/`): `md_guard.sh`
(no scattered `.md`), `auto_push.sh` (auto-push `lxplus`, never `main`),
`worktree_guard.sh` (worktree nudge on `lxplus`), `figure_pair_guard.sh` (png+pdf
pairing).
