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


def build_compact(args) -> None:
    """One narrow panel carrying both messages, for a two-column report.

    The three-panel version is unreadable at column width, and shrinking it would put a
    figure in the report that nobody can read. Horizontal bars instead, so the policy
    names have room, with coverage and worst-zone side by side.
    """
    with args.csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    labels = [r["policy"].replace(" (base)", "") for r in rows]
    y = np.arange(len(rows))
    height = 0.38

    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.barh(
        y + height / 2, [float(r["coverage"]) for r in rows], height,
        color=[REWARD_COLOUR.get(r["reward"], "tab:gray") for r in rows],
        label="coverage",
    )
    ax.barh(
        y - height / 2, [float(r["worst_zone_coverage"]) for r in rows], height,
        color=[REWARD_COLOUR.get(r["reward"], "tab:gray") for r in rows],
        alpha=0.45, label="worst-zone coverage",
    )
    ax.axvline(float(rows[0]["coverage"]), color="black", ls="--", lw=1.1, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("fraction within 8 min", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Solid: coverage.  Faded: worst zone.", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = args.out.with_name("phase4_reward_compact.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/phase4.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/phase4_reward.png"))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument(
        "--compact",
        action="store_true",
        help="single narrow panel sized for one column of the report",
    )
    args = ap.parse_args()

    if args.compact:
        build_compact(args)
        return

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
