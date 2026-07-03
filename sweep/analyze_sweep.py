#!/usr/bin/env python3
"""
analyze_sweep.py  —  Inspect a completed (or in-progress) DyHPO sweep.

Usage (from lxplus login node or locally with AFS/EOS mounted):
    python sweep/analyze_sweep.py /afs/cern.ch/user/j/joiturri/likelihoods/sweeps/1911_ewkinos_004/
    python sweep/analyze_sweep.py sweeps/1911_ewkinos_004/ --top 20
    python sweep/analyze_sweep.py sweeps/1911_ewkinos_004/ --no-plots

Produces:
  - Text summary of best results printed to stdout
  - PDF with 4 plots saved under <project_dir>/runs/<sweep_name>/<sweep_name>_analysis.pdf
"""

import argparse
import os
import re
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def open_sweep(sweep_dir, force_cpu=False):
    sweep_dir   = os.path.abspath(sweep_dir)
    state_path  = os.path.join(sweep_dir, "dyhpo_state.pkl")
    config_path = os.path.join(sweep_dir, "sweep_config.yaml")

    if not os.path.exists(state_path):
        sys.exit(f"DyHPO state not found: {state_path}\nHas generate_sweep.py been run?")
    if not os.path.exists(config_path):
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eos_output_path = os.path.join(
        cfg["paths"]["eos_sweep_dir"], cfg["sweep_name"], "dyhpo_surrogate"
    )

    from sweep.dyhpo_sampler import DyHPOSampler
    sampler = DyHPOSampler.load(state_path, eos_output_path, force_cpu=force_cpu)
    return sampler, cfg


def _afs_trial_info(afs_sweep_dir: str, sweep_name: str, hp_idx: int) -> list:
    """Return a list of dicts for AFS trial(s) that ran for hp_idx."""
    out_dir = os.path.join(afs_sweep_dir, sweep_name, "output")
    log_dir = os.path.join(afs_sweep_dir, sweep_name, "log")
    if not os.path.isdir(out_dir):
        return []

    matches = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".out"):
            continue
        fm = re.match(r'trial_(\d+)\.(\d+)\.0\.out$', fname)
        if not fm:
            continue
        trial_idx = int(fm.group(1))
        job_id    = fm.group(2)

        with open(os.path.join(out_dir, fname)) as f:
            content = f.read()

        hp_m = re.search(r'hp_idx=(\d+)', content)
        if not hp_m or int(hp_m.group(1)) != hp_idx:
            continue

        log_candidates = (
            [f for f in os.listdir(log_dir) if f.startswith(f"trial_{trial_idx:04d}.{job_id}.")]
            if os.path.isdir(log_dir) else []
        )
        log_file = os.path.join(log_dir, log_candidates[0]) if log_candidates else None

        gpu, start, end = "n/a", "n/a", "still running"
        if log_file and os.path.exists(log_file):
            with open(log_file) as f:
                lc = f.read()
            gm  = re.search(r'DeviceName\s*=\s*"([^"]+)"', lc)
            sm  = re.search(r'001 \([^)]+\) (\d{2}/\d{2} \d{2}:\d{2}:\d{2})', lc)
            em  = re.search(r'005 \([^)]+\) (\d{2}/\d{2} \d{2}:\d{2}:\d{2})', lc)
            gpu   = gm.group(1) if gm else "not found"
            start = sm.group(1) if sm else "n/a"
            end   = em.group(1) if em else "still running"

        matches.append(dict(trial_idx=trial_idx, job_id=job_id,
                            out_file=os.path.join(out_dir, fname), log_file=log_file,
                            gpu=gpu, start=start, end=end))
    return matches


def print_summary(sampler, cfg, top_n):
    results  = sampler.all_results()
    best     = sampler.best_result()
    t_full   = cfg["fidelity_schedule"]["t_steps"][-1]
    full_results = [r for r in results if r["t_steps"] == t_full]

    print(f"\n{'='*60}")
    print(f"  Sweep: {cfg['sweep_name']}")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Full-fidelity evaluations ({t_full:,} steps): {len(full_results)}")
    print(f"  HP candidates in pool: {sampler.n_candidates}")
    print(f"{'='*60}")

    if best:
        params, loss = best
        print(f"\nBest val_loss (any fidelity): {loss:.6f}")
        print("Best params:")
        for k, v in params.items():
            print(f"  {k} = {v}")

    if full_results:
        best_full = full_results[0]
        print(f"\nBest full-fidelity val_loss: {best_full['val_loss']:.6f}")
        print("Full-fidelity best params:")
        for k, v in best_full['params'].items():
            print(f"  {k} = {v}")

    best_r = full_results[0] if full_results else (results[0] if results else None)
    if best_r:
        trials = _afs_trial_info(
            cfg["paths"]["afs_sweep_dir"], cfg["sweep_name"], best_r["hp_idx"]
        )
        print(f"\nBest run logs  (hp_{best_r['hp_idx']:04d}):")
        if trials:
            for t in trials:
                print(f"  AFS trial_{t['trial_idx']:04d}  (job {t['job_id']})")
                print(f"    GPU:             {t['gpu']}")
                print(f"    Start / End:     {t['start']}  →  {t['end']}")
                print(f"    HTCondor log:    {t['log_file'] or 'not found'}")
                print(f"    Training stdout: {t['out_file']}")
        else:
            print("  (AFS output directory not accessible from this node)")

    print(f"\nTop-{min(top_n, len(results))} results:\n")
    for r in results[:top_n]:
        ps = "  ".join(
            f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
            for k, v in r['params'].items()
        )
        print(f"  hp_{r['hp_idx']:04d}  val_loss={r['val_loss']:.6f}"
              f"  t_steps={r['t_steps']}  {ps}")
    print()


def save_plots(sampler, cfg, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np
    except ImportError as e:
        print(f"Skipping plots — missing dependency: {e}")
        return

    results = sampler.all_results()
    if len(results) < 2:
        print("Not enough completed evaluations for plots (need >= 2).")
        return

    hp_space     = cfg.get("search_space", [])
    param_names  = [e["name"] for e in hp_space]
    t_steps_vals = cfg["fidelity_schedule"]["t_steps"]
    t_full       = t_steps_vals[-1]
    full_r       = [r for r in results if r["t_steps"] == t_full]
    scatter_r    = full_r if len(full_r) >= 2 else results
    scatter_label = "full fidelity" if len(full_r) >= 2 else "all fidelity levels"

    # Use log scale only when all val_loss values are strictly positive
    all_losses = [r["val_loss"] for r in results]
    use_log_scale = all(v > 0 for v in all_losses)

    FONTSIZE = 14
    LABEL_FS = 11
    TICK_FS  = 10
    COLORS   = ["black", "#0343DE", "#A52A2A", "darkorange", "#2ca02c"]

    with PdfPages(out_path) as pdf:

        # --- 1. Optimization history ---
        chron = sampler.all_results_chronological()
        values = [r["val_loss"] for r in chron]
        best_so_far = [min(values[:i+1]) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(range(len(values)), values, color=COLORS[0], s=15, alpha=0.4, zorder=2,
                   label="val_loss")
        ax.plot(range(len(values)), best_so_far, color=COLORS[1], linewidth=1.5, zorder=3,
                label="best so far")
        best_idx = int(np.argmin(values))
        ax.scatter(best_idx, values[best_idx], s=120, marker="*", color=COLORS[3], zorder=4,
                   label=f"best ({values[best_idx]:.4f})")
        if use_log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Evaluation index", fontsize=LABEL_FS)
        ax.set_ylabel("val_loss", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=TICK_FS, frameon=False)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
        ax.set_title(f"Optimization history  —  {cfg['sweep_name']}", fontsize=FONTSIZE)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- 2. HP importance (Random Forest) ---
        if len(param_names) > 0 and len(results) >= 2:
            try:
                from sklearn.ensemble import RandomForestRegressor

                log_params = {e["name"] for e in hp_space if e["type"] in ("float_log", "int_log")}
                cat_params = {e["name"] for e in hp_space if e["type"] == "categorical"}
                cat_encodings = {}
                for e in hp_space:
                    if e["type"] == "categorical":
                        cat_encodings[e["name"]] = {c: i for i, c in enumerate(e["choices"])}

                X_imp, y_imp = [], []
                for r in results:
                    row = []
                    for e in hp_space:
                        v = r["params"].get(e["name"])
                        if e["name"] in cat_params:
                            row.append(float(cat_encodings[e["name"]].get(v, np.nan)))
                        elif e["name"] in log_params and v is not None and v > 0:
                            row.append(np.log(v))
                        else:
                            row.append(float(v) if v is not None else np.nan)
                    X_imp.append(row)
                    y_imp.append(np.log(r["val_loss"]) if (use_log_scale and r["val_loss"] > 0) else r["val_loss"])

                X_imp = np.array(X_imp)
                y_imp = np.array(y_imp)
                mask  = np.isfinite(X_imp).all(axis=1) & np.isfinite(y_imp)
                if mask.sum() >= 2:
                    rf = RandomForestRegressor(n_estimators=200, random_state=42)
                    rf.fit(X_imp[mask], y_imp[mask])
                    importances = rf.feature_importances_
                    order  = np.argsort(importances)[::-1]
                    labels = [param_names[i].split(".")[-1] for i in order]

                    n_p = len(param_names)
                    fig, ax = plt.subplots(figsize=(max(6, n_p * 1.4), 4))
                    bars = ax.bar(range(n_p), importances[order],
                                  color=COLORS[1], alpha=0.8, edgecolor="black", linewidth=0.5)
                    ax.set_xticks(range(n_p))
                    ax.set_xticklabels(labels, fontsize=LABEL_FS, rotation=20, ha="right")
                    ax.set_ylabel("Feature importance (RF Gini)", fontsize=LABEL_FS)
                    ax.set_ylim(0, min(1.0, importances.max() * 1.35))
                    ax.tick_params(labelsize=TICK_FS)
                    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
                    for bar, imp in zip(bars, importances[order]):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f"{imp:.2f}", ha="center", va="bottom", fontsize=9)
                    n_used = int(mask.sum())
                    note = "" if n_used >= 20 else f"  (n={n_used}, interpret with caution)"
                    ax.set_title(f"HP importance (Random Forest, all evals){note}", fontsize=FONTSIZE)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            except ImportError:
                print("Skipping HP importance — sklearn not available.")

        # --- 3. HP vs val_loss: scatter for continuous, box plot for categorical ---
        if len(scatter_r) >= 2 and len(param_names) > 0:
            full_vals     = [r["val_loss"] for r in scatter_r]
            best_full_val = min(full_vals)
            best_full_idx = full_vals.index(best_full_val)

            ncols = min(len(param_names), 3)
            nrows = (len(param_names) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes_flat = np.array(axes).flatten() if len(param_names) > 1 else [axes]

            for ax, entry in zip(axes_flat, hp_space):
                pname = entry["name"]
                ptype = entry.get("type", "")

                if ptype == "categorical":
                    choices = entry.get("choices", [])
                    grouped = {c: [] for c in choices}
                    for r in scatter_r:
                        v = r["params"].get(pname)
                        if v in grouped:
                            grouped[v].append(r["val_loss"])
                    data_bp = [grouped.get(c, []) for c in choices]
                    short   = [
                        str(c).replace('"', '').replace('[', '').replace(']', '').replace(',', '+')[:22]
                        for c in choices
                    ]
                    ax.boxplot(
                        [d if d else [float('nan')] for d in data_bp],
                        positions=range(len(choices)),
                        patch_artist=True,
                        boxprops=dict(facecolor=COLORS[2], alpha=0.45),
                        medianprops=dict(color="black", linewidth=2),
                    )
                    # scatter individual points on top
                    for i, (c, d) in enumerate(zip(choices, data_bp)):
                        ax.scatter([i] * len(d), d, color=COLORS[0], s=18, alpha=0.5, zorder=3)
                        ymin = min(full_vals)
                        yann = ymin * 0.98 if ymin > 0 else ymin * 1.02
                        ax.annotate(f"n={len(d)}", (i, yann), ha="center", fontsize=8)
                    ax.set_xticks(range(len(choices)))
                    ax.set_xticklabels(short, fontsize=8, rotation=20, ha="right")
                    if use_log_scale:
                        ax.set_yscale("log")
                else:
                    pvals     = [r["params"].get(pname) for r in scatter_r]
                    best_pval = pvals[best_full_idx]
                    is_log    = ptype in ("float_log", "int_log")
                    ax.scatter(pvals, full_vals, color="#aaaaaa", s=25, alpha=0.75, zorder=2)
                    ax.scatter(best_pval, best_full_val, s=150, marker="*", color=COLORS[3],
                               zorder=5, label="best")
                    ax.axvline(best_pval, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                    ax.axhline(best_full_val, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                    if use_log_scale:
                        ax.set_yscale("log")
                    if is_log:
                        ax.set_xscale("log")
                    ax.legend(fontsize=TICK_FS, frameon=False)

                ax.set_xlabel(pname.split(".")[-1], fontsize=LABEL_FS)
                ax.set_ylabel("val_loss", fontsize=LABEL_FS)
                ax.tick_params(labelsize=TICK_FS)
                ax.grid(True, which="both", linewidth=0.4, alpha=0.4)

            for ax in axes_flat[len(param_names):]:
                ax.set_visible(False)

            fig.suptitle(
                f"HP vs val_loss — {scatter_label} (n={len(scatter_r)})  ★ = best",
                fontsize=FONTSIZE, y=1.01,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- 4. Data transform breakdown ---
        # For each categorical HP, show val_loss distributions side-by-side
        # and the best continuous-HP config within each transform choice.
        cat_entries = [e for e in hp_space if e.get("type") == "categorical"]
        if cat_entries and len(scatter_r) >= 2:
            full_vals = [r["val_loss"] for r in scatter_r]

            for entry in cat_entries:
                pname   = entry["name"]
                choices = entry.get("choices", [])
                grouped = {c: [] for c in choices}
                for r in scatter_r:
                    v = r["params"].get(pname)
                    if v in grouped:
                        grouped[v].append(r)

                n_choices = len(choices)
                fig, axes = plt.subplots(1, n_choices, figsize=(5 * n_choices, 5),
                                         squeeze=False, sharey=True)
                axes = axes[0]

                for ax, choice in zip(axes, choices):
                    group = grouped.get(choice, [])
                    losses = [r["val_loss"] for r in group]
                    short = (str(choice).replace('"', '').replace('[', '')
                                        .replace(']', '').replace(',', ' +'))

                    if losses:
                        # scatter all points
                        ax.scatter(range(len(losses)), sorted(losses),
                                   color=COLORS[1], s=30, alpha=0.7, zorder=3)
                        best_loss = min(losses)
                        ax.axhline(best_loss, color=COLORS[3], linestyle="--",
                                   linewidth=1.2, label=f"best={best_loss:.4f}")
                        ax.legend(fontsize=8, frameon=False)
                        # annotate best config params (continuous HPs only)
                        best_r = min(group, key=lambda r: r["val_loss"])
                        cont_params = {
                            k.split(".")[-1]: v for k, v in best_r["params"].items()
                            if not any(e["name"] == k and e.get("type") == "categorical"
                                       for e in hp_space)
                        }
                        ann = "\n".join(
                            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in cont_params.items()
                        )
                        ax.text(0.05, 0.97, ann, transform=ax.transAxes,
                                fontsize=7, va="top", family="monospace",
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
                    else:
                        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                                ha="center", va="center", fontsize=LABEL_FS, color="gray")

                    if use_log_scale:
                        ax.set_yscale("log")
                    ax.set_title(short, fontsize=10)
                    ax.set_xlabel("rank", fontsize=LABEL_FS)
                    ax.tick_params(labelsize=TICK_FS)
                    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)

                axes[0].set_ylabel("val_loss", fontsize=LABEL_FS)
                short_pname = pname.split(".")[-1]
                fig.suptitle(
                    f"val_loss by {short_pname}  —  {scatter_label} (n={len(scatter_r)})",
                    fontsize=FONTSIZE,
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"Plots saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a DyHPO HPO sweep for the likelihoods project",
        epilog=(
            "Examples:\n"
            "  python sweep/analyze_sweep.py /afs/.../sweeps/1911_ewkinos_004/\n"
            "  python sweep/analyze_sweep.py sweeps/1911_ewkinos_004/ --top 20\n"
            "  python sweep/analyze_sweep.py sweeps/1911_ewkinos_004/ --no-plots\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep_dir", help="Path to the AFS sweep directory")
    parser.add_argument("--top",      type=int, default=10, help="Number of top results to print")
    parser.add_argument("--no-plots", action="store_true",  help="Skip PDF generation")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force surrogate on CPU (use on login nodes without a free GPU)")
    parser.add_argument("--out",      default=None,
                        help="Override output PDF path")
    args = parser.parse_args()

    sampler, cfg = open_sweep(args.sweep_dir, force_cpu=args.cpu)

    print_summary(sampler, cfg, top_n=args.top)

    if not args.no_plots:
        if args.out:
            plot_path = args.out
        else:
            sweep_name = cfg["sweep_name"]
            plot_dir   = os.path.join(cfg["paths"]["project_dir"], "runs", sweep_name)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path  = os.path.join(plot_dir, f"{sweep_name}_analysis.pdf")
        save_plots(sampler, cfg, plot_path)


if __name__ == "__main__":
    main()
