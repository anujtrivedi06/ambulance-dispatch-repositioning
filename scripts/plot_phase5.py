"""Plot the Phase 5 trade-offs.

    python scripts/run_phase5.py --out-csv results/phase5.csv
    python scripts/plot_phase5.py results/phase5.csv

Left: the equity/efficiency frontier, city-wide coverage against the worst-served zone.
Right: what reserving units for urgent calls does to each priority class. Both are drawn
as trade-off curves rather than bar charts because the point in each case is the shape of
the exchange rate, not which single configuration wins.
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
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/phase5.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/phase5_tradeoffs.png"))
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    equity = sorted(
        (r for r in rows if r["family"] == "equity"), key=lambda r: float(r["knob"])
    )
    selective = sorted(
        (r for r in rows if r["family"] == "selective"), key=lambda r: float(r["knob"])
    )
    se = lambda r, m: float(r[f"{m}_sd"]) / args.reps**0.5

    fig, (ax, ax_mid, ax2) = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- equity: there is no frontier here --------------------------------------
    # Deliberately not drawn as a Pareto front. Every point sits inside about two
    # standard errors of lambda = 0, so joining them or marking "non-dominated" ones
    # would draw a trade-off curve out of sampling noise.
    x = [float(r["coverage"]) for r in equity]
    y = [float(r["worst_zone_coverage"]) for r in equity]
    lams = [float(r["knob"]) for r in equity]
    base = equity[0]
    bx, by = float(base["coverage"]), float(base["worst_zone_coverage"])
    sx, sy = se(base, "coverage"), se(base, "worst_zone_coverage")

    ax.add_patch(
        plt.Rectangle(
            (bx - 2 * sx, by - 2 * sy), 4 * sx, 4 * sy,
            color="black", alpha=0.08, lw=0, zorder=0,
        )
    )
    ax.errorbar(
        x, y,
        xerr=[se(r, "coverage") for r in equity],
        yerr=[se(r, "worst_zone_coverage") for r in equity],
        ls="none", color="tab:blue", alpha=0.45, capsize=2, zorder=1,
    )
    scatter = ax.scatter(x, y, c=lams, cmap="viridis", s=70, zorder=2)
    plt.colorbar(scatter, ax=ax, label="equity weight λ")
    ax.annotate(
        "±2 s.e. of λ=0",
        (bx - 2 * sx, by + 2 * sy),
        textcoords="offset points", xytext=(4, 4), fontsize=8, color="#555555",
    )
    ax.set_xlabel("city-wide coverage within 8 min")
    ax.set_ylabel("worst-zone coverage")
    ax.set_title("No equity/efficiency frontier:\nevery λ lands in the same noise band",
                 fontsize=11)
    ax.grid(alpha=0.3)

    # --- equity: the knob works, the outcome does not move ----------------------
    decisions = args.csv.with_name(args.csv.stem + "_decisions.csv")
    if decisions.exists():
        with decisions.open(encoding="utf-8") as fh:
            drows = sorted(csv.DictReader(fh), key=lambda r: float(r["lambda"]))
        ax_mid.plot(
            [float(r["lambda"]) for r in drows],
            [100 * (1 - float(r["same_choice_fraction"])) for r in drows],
            marker="o", color="tab:purple", lw=1.8,
        )
        ax_mid.set_xlabel("equity weight λ")
        ax_mid.set_ylabel("% of decisions changed vs λ=0")
        ax_mid.set_title(
            "The knob is not inert: it moves\ndecisions, outcomes stay flat", fontsize=11
        )
        ax_mid.grid(alpha=0.3)
    else:
        ax_mid.axis("off")

    # --- priority trade-off -----------------------------------------------------
    reserves = [float(r["knob"]) for r in selective]
    for metric, colour, label in (
        ("coverage_p1", "tab:red", "urgent (priority 1)"),
        ("coverage_p2", "tab:blue", "routine (priority 2)"),
        ("coverage", "#444444", "all calls"),
    ):
        ax2.errorbar(
            reserves,
            [float(r[metric]) for r in selective],
            yerr=[se(r, metric) for r in selective],
            marker="o", color=colour, label=label, capsize=3, lw=1.6,
        )
    ax2.set_xlabel("units reserved from routine calls")
    ax2.set_ylabel("coverage within 8 min")
    ax2.set_xticks(reserves)
    ax2.set_title("Selective dispatch: holding units back\nfor urgent calls", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
