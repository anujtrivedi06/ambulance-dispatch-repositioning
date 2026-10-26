"""Plot the Phase 4 reward ablation.

    python scripts/run_phase4.py --out-csv results/phase4.csv
    python scripts/plot_phase4.py results/phase4.csv

Two panels, because the headline and the damage show up on different metrics. City-wide
coverage is what every method was trying to maximise; worst-zone coverage is what the
binary reward quietly destroys while doing it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# LSPI and CEM carry no reward tag on purpose. LSPI is policy iteration rather than a
# single fit, and CEM optimises the coverage metric directly instead of learning from a
# reward signal at all, so colouring either as "binary reward" would misattribute the
# comparison the figure is making.
REWARD_COLOUR = {
    "base": "#444444",
    "coverage": "tab:red",
    "response": "tab:blue",
    "": "tab:green",
}
REWARD_LABEL = {
    "base": "heuristic (no learning)",
    "coverage": "binary 8-minute reward",
    "response": "continuous response-time reward",
    "": "policy iteration / direct search",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/phase4.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/phase4_reward.png"))
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    labels = [r["policy"] for r in rows]
    colours = [REWARD_COLOUR.get(r["reward"], "tab:gray") for r in rows]
    assert all(r["reward"] in REWARD_COLOUR for r in rows), "unlabelled reward category"
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        ("coverage", "coverage within 8 min", "What every method optimised"),
        ("worst_zone_coverage", "worst-zone coverage", "What the binary reward cost"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, panels):
        values = np.array([float(r[metric]) for r in rows])
        errors = np.array([float(r[f"{metric}_sd"]) for r in rows]) / args.reps**0.5
        ax.bar(x, values, yerr=errors, capsize=3, color=colours)
        baseline = values[0]
        ax.axhline(baseline, color="black", ls="--", lw=1.2, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(1.0, values.max() * 1.15))
        ax.grid(axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REWARD_COLOUR[k]) for k in REWARD_LABEL
    ]
    fig.legend(
        handles,
        list(REWARD_LABEL.values()),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Learned repositioning: the training signal mattered more than the model",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
