"""Plot the Phase 3 anytime curve: solution quality against simulation budget.

    python scripts/run_phase3.py --methods rollout --out results/phase3_rollout.csv
    python scripts/run_phase3.py --methods mcts    --out results/phase3_mcts.csv
    python scripts/plot_phase3.py results/phase3_rollout.csv results/phase3_mcts.csv

Several CSVs can be passed because the two sweeps are run as separate processes; they
are slow enough that running them in parallel is worth the extra file.

The x axis is forks per replication -- the actual simulation work done -- rather than the
nominal budget parameter, because rollout spends `candidates x sims` per decision while
MCTS spends `sims`. Comparing them on the parameter would flatter one of them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"rollout": ("tab:blue", "o"), "mcts": ("tab:orange", "s")}


def load(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="one or more results CSVs written by run_phase3.py --out",
    )
    ap.add_argument("--out", type=Path, default=Path("results/figures/phase3_anytime.png"))
    ap.add_argument("--metric", default="coverage")
    args = ap.parse_args()

    rows = load(args.csv)
    n_reps = 5  # replications behind each point; sets the standard-error bars
    base = next((r for r in rows if r["method"].startswith("dmexclp")), None)

    fig, (ax, ax_cost) = plt.subplots(1, 2, figsize=(11, 4.2))

    for method, (colour, marker) in STYLE.items():
        pts = sorted(
            (
                (float(r["forks_per_rep"]), float(r[args.metric]), float(r[f"{args.metric}_sd"]))
                for r in rows
                if r["method"] == method
            )
        )
        if not pts:
            continue
        x, y, sd = zip(*pts)
        # Markers only, no connecting line: the points do not trend, and joining them
        # would draw a shape the data does not support.
        ax.errorbar(
            x, y, yerr=[s / n_reps**0.5 for s in sd], color=colour, marker=marker,
            capsize=3, label=method, ls="none", ms=6,
        )
        costs = sorted(
            (float(r["forks_per_rep"]), float(r["seconds"])) for r in rows
            if r["method"] == method
        )
        cx, cy = zip(*costs)
        ax_cost.plot(cx, cy, color=colour, marker=marker, label=method, lw=1.6)

    if base is not None:
        mid = float(base[args.metric])
        se = float(base[f"{args.metric}_sd"]) / n_reps**0.5
        ax.axhline(mid, color="black", ls="--", lw=1.4, label="DMEXCLP base policy")
        # The base policy has its own sampling error. Drawing it makes the comparison
        # fair: a search point is only "above the line" if it clears this band too.
        ax.axhspan(mid - se, mid + se, color="black", alpha=0.08, lw=0)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("simulated futures per replication (log scale)")
    ax.set_ylabel(args.metric.replace("_", " "))
    ax.set_title("Quality vs simulation budget")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax_cost.set_xscale("log", base=2)
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("simulated futures per replication (log scale)")
    ax_cost.set_ylabel("wall-clock seconds")
    ax_cost.set_title("Cost of that budget")
    ax_cost.legend(frameon=False, fontsize=9)
    ax_cost.grid(alpha=0.3)

    fig.suptitle(
        "Online search on ambulance repositioning: the base policy is the line to beat",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
