"""
Figure 3: CFM decoder sample-quality test (§V.C.2)
Two-panel: (a) per-dimension L1/std bars with thresholds, (b) pred-vs-gt scatter grid.

Status: Script only, NOT executed yet. Render Day 5+ when GPU mode available.

Data source: 5a v2 sample quality test outputs.
  - results/stage5/5a_v2_sample_quality.json (per-dim L1/std + per-dim mean predictions)

Output: paper/figures/figure3_sample_quality.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("results/stage5")
OUTPUT_PATH = Path("paper/figures/figure3_sample_quality.pdf")

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.figsize": (7.0, 4.5),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DIM_LABELS = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]


def _load_data():
    """Load 5a v2 sample quality test results."""
    with open(DATA_DIR / "5a_v2_sample_quality.json") as f:
        data = json.load(f)
    return data


def make_figure():
    data = _load_data()

    # Panel (a) data: per-dimension L1/std ratio
    l1_std_per_dim = np.array(data["per_dim_l1_over_std"])  # shape (7,)
    pass_threshold = 0.3
    fail_threshold = 0.8

    # Panel (b) data: predicted vs ground-truth actions per dimension
    pred_actions = np.array(data["predicted_actions"])  # shape (N, 7)
    gt_actions = np.array(data["ground_truth_actions"])  # shape (N, 7)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4])

    # ---- Panel (a): bar chart of L1/std ratios ----
    ax_a = fig.add_subplot(gs[0, :])
    colors = ["black" if v >= fail_threshold else
              "gray" if v >= pass_threshold else "white"
              for v in l1_std_per_dim]
    bars = ax_a.bar(DIM_LABELS, l1_std_per_dim, color=colors,
                    edgecolor="black", linewidth=1.2)
    ax_a.axhline(pass_threshold, color="green", linestyle="--",
                 linewidth=0.8, alpha=0.7, label=f"PASS threshold ({pass_threshold})")
    ax_a.axhline(fail_threshold, color="red", linestyle="--",
                 linewidth=0.8, alpha=0.7, label=f"FAIL threshold ({fail_threshold})")
    ax_a.set_ylabel("L1 error / std(GT)")
    ax_a.set_title("(a) Per-dimension sample quality — 7/7 FAIL")
    ax_a.legend(loc="upper left", framealpha=0.9)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Value labels above bars
    for bar, v in zip(bars, l1_std_per_dim):
        ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # ---- Panel (b): scatter grid of pred vs gt per dimension ----
    # 7 dimensions arranged as 2 rows × 4 cols (last cell empty)
    for i, label in enumerate(DIM_LABELS):
        row = 1
        col = i if i < 4 else i - 4
        # We collapse the second sub-row into the same gridspec row.
        # For cleaner layout, defer to a sub-gridspec.
        ax_b = fig.add_subplot(gs[1, col]) if i < 4 else None
        if i >= 4:
            continue
        gt = gt_actions[:, i]
        pred = pred_actions[:, i]
        ax_b.scatter(gt, pred, s=8, c="black", alpha=0.5, edgecolors="none")
        # Identity line
        lim = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax_b.plot(lim, lim, "gray", linewidth=0.5, linestyle=":")
        ax_b.set_xlabel(f"GT {label}", fontsize=8)
        ax_b.set_ylabel(f"Pred {label}", fontsize=8)
        ax_b.tick_params(labelsize=7)
        ax_b.set_aspect("equal", adjustable="datalim")

    # NOTE: Above grid layout serves 4 of 7 dims. For a polished figure,
    # rebuild with a 2x4 inner gridspec spanning gs[1, :] showing all 7
    # plus one empty cell. Deferred until figure render iteration.

    fig.suptitle("Figure 3: CFM decoder sample-quality collapse", fontsize=11, y=1.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    make_figure()


