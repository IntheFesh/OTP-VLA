"""
Figure 2: Backbone-vs-demonstration correlation comparison
The paper's "money chart" — visualization of the 8× squared-correlation gap.

Status: Script only, NOT executed yet. Render Day 5+ when GPU mode available.

Data sources:
  - results/phase0/gate1_sanity_and_m3.json (Mantel test outputs)
  - results/phase0/h_OFT_distance_matrix.npy (inter-task distances on backbone)
  - results/phase0/demo_traj_distance_matrix.npy (inter-task distances on trajectories)
  - results/phase0/lang_distance_matrix.npy (inter-task distances on language)

Output: paper/figures/figure2_correlation.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Configuration ----
DATA_DIR = Path("results/phase0")
OUTPUT_PATH = Path("paper/figures/figure2_correlation.pdf")

# Style (consistent across all paper figures)
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (7.0, 3.2),  # IEEE two-column width
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_distance_matrices():
    """Load inter-task distance matrices from phase0 results."""
    D_h = np.load(DATA_DIR / "h_OFT_distance_matrix.npy")
    D_traj = np.load(DATA_DIR / "demo_traj_distance_matrix.npy")
    D_lang = np.load(DATA_DIR / "lang_distance_matrix.npy")
    return D_h, D_traj, D_lang


def _upper_triangle(D):
    """Extract upper-triangle entries (45 pairs for 10-task matrix)."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def _load_mantel_results():
    """Load Mantel test r and p-values."""
    with open(DATA_DIR / "gate1_sanity_and_m3.json") as f:
        results = json.load(f)
    r_h = results["gate1"]["mantel_r"]
    r_traj = results["m3"]["mantel_r"]
    p_h = results["gate1"]["mantel_p"]
    p_traj = results["m3"]["mantel_p"]
    return r_h, r_traj, p_h, p_traj


def make_figure():
    """Render Figure 2 with two side-by-side scatter plots."""
    D_h, D_traj, D_lang = _load_distance_matrices()
    r_h, r_traj, p_h, p_traj = _load_mantel_results()

    # Extract upper-triangle pairs
    x_lang = _upper_triangle(D_lang)
    y_h = _upper_triangle(D_h)
    y_traj = _upper_triangle(D_traj)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    # ---- Left panel: Gate 1 ----
    ax1.scatter(x_lang, y_h, s=20, c="black", alpha=0.7, edgecolors="none")
    # Linear regression overlay
    slope, intercept = np.polyfit(x_lang, y_h, 1)
    xline = np.linspace(x_lang.min(), x_lang.max(), 100)
    ax1.plot(xline, slope * xline + intercept, "k-", linewidth=1)
    ax1.set_xlabel("Inter-task language distance $D^{\\ell}$")
    ax1.set_ylabel("Inter-task backbone distance $D^h$")
    ax1.set_title("Gate 1: Backbone")
    ax1.text(0.05, 0.92,
             f"$r = {r_h:+.2f}$\n$p < 0.001$",
             transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", linewidth=0.5))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- Right panel: M3 ----
    ax2.scatter(x_lang, y_traj, s=20, c="black", alpha=0.7, edgecolors="none")
    slope2, intercept2 = np.polyfit(x_lang, y_traj, 1)
    ax2.plot(xline, slope2 * xline + intercept2, "k-", linewidth=1)
    ax2.set_xlabel("Inter-task language distance $D^{\\ell}$")
    ax2.set_ylabel("Inter-task trajectory distance $D^{\\mathrm{traj}}$")
    ax2.set_title("M3: Demonstration")
    ax2.text(0.05, 0.92,
             f"$r = {r_traj:+.2f}$\n$p = {p_traj:.2f}$",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", linewidth=0.5))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ---- Annotation: squared correlation ratio ----
    rho_h_sq = r_h ** 2
    rho_traj_sq = r_traj ** 2
    ratio = rho_h_sq / rho_traj_sq
    fig.suptitle(
        f"$\\rho^2_h = {rho_h_sq:.3f}$  vs  $\\rho^2_{{demo}} = {rho_traj_sq:.3f}$  "
        f"$(\\approx {ratio:.0f}\\times$ gap$)$",
        fontsize=11, y=1.02
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    make_figure()


