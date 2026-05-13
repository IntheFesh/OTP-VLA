"""
Figure 4: Oracle ablation (§V.C.3)
Shows decoder loss does NOT improve with ground-truth trajectory injection,
ruling out the "moving target" hypothesis for CFM decoder failure.

Status: Script only, NOT executed yet. Render Day 5+ when GPU mode available.

Data sources:
  - logs/5a_v2_train.log (normal training: head_out → decoder)
  - logs/5a_v3_train.log (oracle: gt_trajectory → decoder)

Output: paper/figures/figure4_oracle.pdf
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG_NORMAL = Path("logs/5a_v2_train.log")
LOG_ORACLE = Path("logs/5a_v3_train.log")
OUTPUT_PATH = Path("paper/figures/figure4_oracle.pdf")

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.figsize": (5.5, 3.3),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_decoder_loss_curve(log_path):
    """Extract (step, decoder_loss) pairs from training log."""
    pattern = re.compile(r"step (\d+) .* decoder_loss=([0-9.]+)")
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def make_figure():
    steps_normal, loss_normal = parse_decoder_loss_curve(LOG_NORMAL)
    steps_oracle, loss_oracle = parse_decoder_loss_curve(LOG_ORACLE)

    fig, ax = plt.subplots()

    ax.plot(steps_normal, loss_normal,
            color="black", linewidth=1.2, linestyle="-",
            label="Normal: head($o, \\ell$) → decoder")
    ax.plot(steps_oracle, loss_oracle,
            color="black", linewidth=1.2, linestyle="--",
            label="Oracle: gt trajectory → decoder")

    # Annotate convergence values
    final_normal = loss_normal[-50:].mean() if len(loss_normal) >= 50 else loss_normal[-1]
    final_oracle = loss_oracle[-50:].mean() if len(loss_oracle) >= 50 else loss_oracle[-1]
    ax.axhline(final_normal, color="gray", linewidth=0.4, alpha=0.5)
    ax.axhline(final_oracle, color="gray", linewidth=0.4, alpha=0.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Decoder loss (nats)")
    ax.set_title("Oracle trajectory does not reduce decoder loss")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Caption-style annotation in plot
    ax.text(0.5, -0.18,
            f"Final loss: normal $\\approx${final_normal:.2f}, "
            f"oracle $\\approx${final_oracle:.2f}.\n"
            "Moving-target hypothesis is rejected; CFM decoder failure is intrinsic.",
            transform=ax.transAxes, fontsize=9, ha="center",
            verticalalignment="top", style="italic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    make_figure()
