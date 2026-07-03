"""
Learnability comparison: all four nLLs (Exp, Obs, ExpA, ObsA).

Three metrics:
  1. Local conditional std  — normalized mean std of target among k-NN neighbours
                              lower = more learnable
  2. k-NN R²                — cross-validated R² from a k-NN regressor
                              higher = more learnable
  3. Mutual information     — per-feature MI with the target (marginal sum)
                              higher = more informative features

Usage examples:
  # EWKinos, threshold at 60 (original)
  python learnability_comparison.py

  # EWKinos, full dataset + auto threshold matching same event fraction
  python learnability_comparison.py --both

  # Any other dataset, both modes
  python learnability_comparison.py --npy data/some_other.npy --name "Sleptons" --both

  # Fixed threshold on a different dataset
  python learnability_comparison.py --npy data/other.npy --threshold 50 --name "Sleptons"
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in tools/)

NLL_NAMES    = ["Exp", "Obs", "ExpA", "ObsA"]
COLORS       = ["#E05C5C", "#5588CC", "#E8A838", "#7B44CC"]
SUBSAMPLE_N  = 30000
K_LCS        = [5, 10, 20, 50]
K_KNN        = [5, 10, 20]
CV           = 5
MI_NEIGHBORS = 5

EWKINOS_FEATURE_NAMES = [
    "CRVV_hghmet", "CRVV_lowmet",
    "CRtau_hghmet", "CRtau_lowmet",
    "CRtop_hghmet", "CRtop_lowmet",
    "SRee_eMLLc_hgh", "SRee_eMLLc_low_hi", "SRee_eMLLc_low_lo",
    "SRee_eMLLd_hgh", "SRee_eMLLd_low_hi", "SRee_eMLLd_low_lo",
    "SRee_eMLLe_hgh", "SRee_eMLLe_low_hi", "SRee_eMLLe_low_lo",
    "SRee_eMLLf_hgh", "SRee_eMLLf_low_hi", "SRee_eMLLf_low_lo",
    "SRee_eMLLg_hgh", "SRee_eMLLg_low_hi",
    "SRee_eMLLh_hgh", "SRee_eMLLh_low_hi",
    "SRmm_eMLLa_hgh", "SRmm_eMLLa_low_hi", "SRmm_eMLLa_low_lo",
    "SRmm_eMLLb_hgh", "SRmm_eMLLb_low_hi", "SRmm_eMLLb_low_lo",
    "SRmm_eMLLc_hgh", "SRmm_eMLLc_low_hi", "SRmm_eMLLc_low_lo",
    "SRmm_eMLLd_hgh", "SRmm_eMLLd_low_hi", "SRmm_eMLLd_low_lo",
    "SRmm_eMLLe_hgh", "SRmm_eMLLe_low_hi", "SRmm_eMLLe_low_lo",
    "SRmm_eMLLf_hgh", "SRmm_eMLLf_low_hi", "SRmm_eMLLf_low_lo",
    "SRmm_eMLLg_hgh", "SRmm_eMLLg_low_hi",
    "SRmm_eMLLh_hgh", "SRmm_eMLLh_low_hi",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(npy_path, threshold, subsample_n=SUBSAMPLE_N):
    """
    Load dataset, filter each nLL by threshold (or use all events if None),
    subsample to a common size, standardize features.

    Returns (subsets_raw, subsets_scaled, n_features, actual_thresholds)
    where actual_thresholds is a dict {nll_name: threshold_used}.
    """
    print(f"Loading {npy_path} (mmap)...")
    data = np.load(npy_path, mmap_mode="r")
    n_cols = data.shape[1]
    n_features = n_cols - 8   # last 8 cols are always the 4 nLL pairs
    print(f"  {data.shape[0]:,} events, {n_features} features + 8 nLL cols")

    X_all  = data[:, :n_features].astype(np.float64)
    nlls   = data[:, -8:].astype(np.float64)
    deltas = nlls[:, 1::2] - nlls[:, ::2]   # (N, 4): Exp, Obs, ExpA, ObsA

    rng = np.random.default_rng(42)
    subsets = []
    min_n = None
    actual_thresholds = {}

    for i, name in enumerate(NLL_NAMES):
        d = deltas[:, i]
        if threshold is None:
            X_sub, y_sub = X_all, d
            tval = None
        elif isinstance(threshold, str) and (threshold.endswith("%") or threshold.endswith("pct")):
            suffix = "%" if threshold.endswith("%") else "pct"
            pct = 100.0 - float(threshold[:-len(suffix)])
            tval = float(np.percentile(d, pct))
            mask = d > tval
            X_sub, y_sub = X_all[mask], d[mask]
        else:
            tval = float(threshold)
            mask = d > tval
            X_sub, y_sub = X_all[mask], d[mask]

        actual_thresholds[name] = tval
        n = len(X_sub)
        tag = f">{tval:.1f}" if tval is not None else " (all)"
        print(f"  {name}{tag}: {n:,} events  "
              f"delta=[{y_sub.min():.2f}, {y_sub.max():.2f}]  mean={y_sub.mean():.2f}")
        subsets.append((X_sub, y_sub))
        if min_n is None or n < min_n:
            min_n = n

    target_n = min(min_n, subsample_n)
    print(f"\n  Subsampling each nLL to {target_n:,} events...")
    subsets_raw = []
    for X_sub, y_sub in subsets:
        idx = rng.choice(len(X_sub), target_n, replace=False)
        subsets_raw.append((X_sub[idx], y_sub[idx]))

    scaler = StandardScaler()
    scaler.fit(np.vstack([X for X, _ in subsets_raw]))
    subsets_scaled = [(scaler.transform(X), y) for X, y in subsets_raw]

    return subsets_raw, subsets_scaled, n_features, actual_thresholds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def local_conditional_std(X, y, k=20):
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(X)
    local_var = np.array([y[idx[i, 1:]].var() for i in range(len(X))])
    return np.sqrt(local_var.mean()) / y.std()


def run_local_cond_std(subsets_scaled):
    print("\n[1] Local conditional std")
    results = {k: [] for k in K_LCS}
    for k in K_LCS:
        print(f"  k={k}:", end="")
        for (Xs, y), name in zip(subsets_scaled, NLL_NAMES):
            val = local_conditional_std(Xs, y, k=k)
            results[k].append(val)
            print(f"  {name}={val:.4f}", end="")
        print()
    return results


def run_knn_r2(subsets_scaled):
    print("\n[2] k-NN R²")
    results = {k: [] for k in K_KNN}
    for k in K_KNN:
        print(f"  k={k}:", end="")
        for (Xs, y), name in zip(subsets_scaled, NLL_NAMES):
            scores = cross_val_score(
                KNeighborsRegressor(n_neighbors=k, n_jobs=-1),
                Xs, y, cv=CV, scoring="r2"
            )
            results[k].append((scores.mean(), scores.std()))
            print(f"  {name}={scores.mean():.3f}±{scores.std():.3f}", end="")
        print()
    return results


def run_mutual_info(subsets_raw, n_features):
    print("\n[3] Mutual information")
    mi_all = []
    for (X, y), name in zip(subsets_raw, NLL_NAMES):
        mi = mutual_info_regression(X, y, n_neighbors=MI_NEIGHBORS, random_state=42)
        mi_all.append(mi)
        print(f"  {name}: total MI={mi.sum():.4f}")
    return mi_all


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(pdf, lcs_results, knn_results, mi_all, n_features,
                 actual_thresholds, dataset_name, feature_names=None):
    FS = 10
    n_nll  = len(NLL_NAMES)
    x      = np.arange(n_nll)
    w      = 0.18

    tvals  = [actual_thresholds[n] for n in NLL_NAMES]
    tstr   = (f">{tvals[0]:.1f}" if tvals[0] is not None else "full dataset")
    title_prefix = f"{dataset_name} — {tstr}"

    # ---- Page 1: scalar metrics -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Learnability: {title_prefix}", fontsize=FS + 2)

    ax = axes[0]
    k_list  = sorted(lcs_results)
    n_k     = len(k_list)
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * w
    for ki, (k, offset) in enumerate(zip(k_list, offsets)):
        alpha = 0.45 + 0.55 * ki / max(n_k - 1, 1)
        for ni, (val, col) in enumerate(zip(lcs_results[k], COLORS)):
            ax.bar(x[ni] + offset, val, w * 0.9, color=col, alpha=alpha)
    for name, col in zip(NLL_NAMES, COLORS):
        ax.bar(0, 0, color=col, label=name)
    for ki, k in enumerate(k_list):
        ax.bar(0, 0, color="gray", alpha=0.45 + 0.55 * ki / max(n_k - 1, 1), label=f"k={k}")
    ax.set_xticks(x); ax.set_xticklabels(NLL_NAMES, fontsize=FS)
    ax.set_ylabel("Norm. local cond. std", fontsize=FS)
    ax.set_title("Metric 1: Local Conditional Std", fontsize=FS + 1)
    ax.legend(fontsize=FS - 2, ncol=2)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)

    ax = axes[1]
    for ki, (k, offset) in enumerate(zip(K_KNN,
            np.linspace(-(len(K_KNN) - 1) / 2, (len(K_KNN) - 1) / 2, len(K_KNN)) * w)):
        alpha = 0.45 + 0.55 * ki / max(len(K_KNN) - 1, 1)
        for ni, ((m, s), col) in enumerate(zip(knn_results[k], COLORS)):
            ax.bar(x[ni] + offset, m, w * 0.9, yerr=s, color=col, alpha=alpha, capsize=3)
    for name, col in zip(NLL_NAMES, COLORS):
        ax.bar(0, 0, color=col, label=name)
    for ki, k in enumerate(K_KNN):
        ax.bar(0, 0, color="gray", alpha=0.45 + 0.55 * ki / max(len(K_KNN) - 1, 1), label=f"k={k}")
    ax.set_xticks(x); ax.set_xticklabels(NLL_NAMES, fontsize=FS)
    ax.set_ylabel("R²", fontsize=FS)
    ax.set_title("Metric 2: k-NN R²", fontsize=FS + 1)
    ax.legend(fontsize=FS - 2, ncol=2)
    ax.axhline(0, color="k", lw=1.0, ls="--")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 2: per-feature MI ------------------------------------------
    fnames = feature_names if feature_names else [f"f{i}" for i in range(n_features)]
    order  = np.argsort(mi_all[0])[::-1]
    x_feat = np.arange(n_features)
    bar_w  = 0.22

    fig, ax = plt.subplots(figsize=(max(12, n_features * 0.35), 6))
    for i, (name, col, mi) in enumerate(zip(NLL_NAMES, COLORS, mi_all)):
        ax.bar(x_feat + (i - 1.5) * bar_w, mi[order], bar_w, color=col, alpha=0.85, label=name)
    ax.set_xticks(x_feat)
    ax.set_xticklabels([fnames[i] for i in order], rotation=60, ha="right",
                       fontsize=max(4, min(6, 120 // n_features)))
    ax.set_ylabel("Mutual Information", fontsize=FS)
    ax.set_title(f"Metric 3: Per-feature MI — {title_prefix}", fontsize=FS + 1)
    ax.legend(fontsize=FS)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 3: summary --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Summary — {title_prefix}", fontsize=FS + 2)

    ax = axes[0]
    totals = [mi.sum() for mi in mi_all]
    bars = ax.bar(NLL_NAMES, totals, color=COLORS, alpha=0.85, width=0.5)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=FS)
    ax.set_ylabel("Total MI sum", fontsize=FS)
    ax.set_title("Metric 3: Total MI", fontsize=FS + 1)

    ax = axes[1]
    lcs_k10 = lcs_results[10]
    r2_k10  = [knn_results[10][i][0] for i in range(n_nll)]
    ax2 = ax.twinx()
    for ni, (lv, rv, col) in enumerate(zip(lcs_k10, r2_k10, COLORS)):
        ax.bar(ni - 0.2, lv, 0.35, color=col, alpha=0.55)
        ax2.bar(ni + 0.2, rv, 0.35, color=col, alpha=0.9)
        ax.text(ni - 0.2, max(lv + 0.01, 0.01), f"{lv:.3f}", ha="center", fontsize=FS - 1)
        ax2.text(ni + 0.2, rv + (0.02 if rv >= 0 else -0.07), f"{rv:.3f}",
                 ha="center", fontsize=FS - 1)
    ax.set_xticks(range(n_nll)); ax.set_xticklabels(NLL_NAMES, fontsize=FS)
    ax.set_ylabel("Local cond. std", fontsize=FS)
    ax2.set_ylabel("k-NN R²", fontsize=FS)
    ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Metrics 1 & 2 at k=10\n(transparent=LCS, solid=R²)", fontsize=FS + 1)
    for name, col in zip(NLL_NAMES, COLORS):
        ax.bar(0, 0, color=col, label=name)
    ax.legend(fontsize=FS - 1)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run_one_mode(pdf, npy_path, threshold, dataset_name, feature_names):
    subsets_raw, subsets_scaled, n_features, actual_thresholds = load_data(
        npy_path, threshold
    )
    lcs_results = run_local_cond_std(subsets_scaled)
    knn_results = run_knn_r2(subsets_scaled)
    mi_all      = run_mutual_info(subsets_raw, n_features)
    plot_results(pdf, lcs_results, knn_results, mi_all, n_features,
                 actual_thresholds, dataset_name, feature_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ewk_default = os.path.join(BASE_DIR, "data",
                               "1911.12606-EWKinos-1M-z4-nll400-delta200.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", default=ewk_default,
                        help="Path to .npy dataset (default: EWKinos)")
    parser.add_argument("--name", default=None,
                        help="Dataset name for plot titles (default: filename stem)")
    parser.add_argument("--threshold", default="60",
                        help="Threshold value, or 'Xpct' for top-X%% (e.g. '3pct'). "
                             "Use 'none' for no threshold.")
    parser.add_argument("--percentile", type=float, default=None,
                        help="Use top-X%% as threshold (overrides --threshold). "
                             "E.g. --percentile 3 selects the top 3%% of each delta.")
    parser.add_argument("--both", action="store_true",
                        help="Run both full-dataset and threshold modes in one PDF")
    parser.add_argument("--out", default=None,
                        help="Output PDF path (default: plots/<dataset_name>_learnability.pdf)")
    args = parser.parse_args()

    # Resolve threshold (--percentile takes priority over --threshold)
    if args.percentile is not None:
        thr_str   = f"{args.percentile}pct"
        threshold = thr_str
    else:
        thr_str = args.threshold.strip().lower()
        if thr_str == "none":
            threshold = None
        elif thr_str.endswith("pct"):
            threshold = thr_str
        else:
            threshold = float(thr_str)

    # Dataset name
    dataset_name = args.name or os.path.splitext(os.path.basename(args.npy))[0]

    # Feature names (only known for EWKinos)
    feature_names = EWKINOS_FEATURE_NAMES if "EWKinos" in args.npy else None

    # Output path
    safe_name = dataset_name.replace("/", "_").replace(" ", "_")
    thr_tag   = thr_str.replace(".", "p")
    out_pdf   = args.out or os.path.join(BASE_DIR, "plots",
                                          f"{safe_name}_learnability_{thr_tag}.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        if args.both:
            # First: full dataset
            print(f"\n{'='*60}")
            print("MODE: full dataset (no threshold)")
            print('='*60)
            run_one_mode(pdf, args.npy, None, dataset_name, feature_names)

            # Then: with threshold
            print(f"\n{'='*60}")
            print(f"MODE: threshold={args.threshold}")
            print('='*60)
            run_one_mode(pdf, args.npy, threshold, dataset_name, feature_names)
        else:
            run_one_mode(pdf, args.npy, threshold, dataset_name, feature_names)

    print(f"\nDone. Output: {out_pdf}")


if __name__ == "__main__":
    main()
