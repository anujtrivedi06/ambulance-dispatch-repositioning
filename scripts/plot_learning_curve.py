"""Plot the RL agent's learning curve against the heuristic baseline.

    python scripts/train_agent.py
    python scripts/plot_learning_curve.py results/learning_curve.csv

The default plot shows held-out policy quality only. TD loss is available via
`--show-loss` but is off by default and should be read with care: training uses a fresh
replay buffer each round, so each round's loss is measured on different data and the
values are not comparable across rounds. Putting an erratic, incomparable series next to
the policy curve invites a trend reading that the numbers do not support.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/learning_curve.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/learning_curve.png"))
    ap.add_argument(
        "--baseline", type=float, default=None,
        help="heuristic coverage to draw as the line to beat",
    )
    ap.add_argument(
        "--show-loss",
        action="store_true",
        help="overlay TD loss (not comparable across rounds; see module docstring)",
    )
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    rounds = [int(r["round"]) for r in rows]
    coverage = [float(r["eval_coverage"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        rounds, coverage, marker="o", color="tab:blue", lw=2,
        label="agent coverage (held-out episodes)",
    )
    if args.baseline is not None:
        ax.axhline(
            args.baseline, color="black", ls="--", lw=1.4,
            label="DMEXCLP heuristic baseline",
        )
    ax.set_xlabel("training round")
    ax.set_ylabel("coverage within 8 min", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if args.show_loss:
        twin = ax.twinx()
        twin.plot(rounds, loss, marker="s", color="tab:red", lw=1.6, alpha=0.8,
                  label="TD loss (not comparable across rounds)")
        twin.set_ylabel("TD training loss", color="tab:red")
        twin.tick_params(axis="y", labelcolor="tab:red")
        h2, l2 = twin.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="lower right")

    spread = max(coverage) - min(coverage)
    ax.set_title(
        f"More training rounds did not improve the policy\n"
        f"(validation coverage spans {spread:.3f} across {len(rounds)} rounds)",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
