import sys
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in tools/)
sys.path.insert(0, BASE)
from plots import plot_histograms_4outputs

d = np.load(f"{BASE}/validations/SUSY-2018-16_Sleptons/results.npz")
preds  = d["preds"]   # (N, 8): [exp0, exp1, obs0, obs1, expA0, expA1, obsA0, obsA1]
truths = d["truths"]

# Extract nLL_1 columns: indices 1, 3, 5, 7 → [exp, obs, expA, obsA]
# plot_histograms_4outputs expects data as list of (n_outputs, N) arrays
truth_nll1 = truths[:, [1, 3, 5, 7]].T  # (4, N)
pred_nll1  = preds[:,  [1, 3, 5, 7]].T  # (4, N)

out = f"{BASE}/validation_plots/SUSY-2018-16_Sleptons_hist_logy.pdf"

plot_histograms_4outputs(
    file=out,
    data=[truth_nll1, pred_nll1],
    labels=["Truth", "Predicted"],
    n_bins=100,
    xlabel=r"nLL$_1$",
    title="SUSY-2018-16 Sleptons — nLL$_1$ truth vs predicted",
    logy=True,
    plot_ratios=True,
    output_names=["Expected", "Observed", "Expected Asimov", "Observed Asimov"],
)
print(f"Saved to {out}")
