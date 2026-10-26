"""Plot the regime sweep: does the configuration explain the negative results?

    python scripts/run_regimes.py --out-csv results/regimes.csv
    python scripts/plot_regimes.py results/regimes.csv

Three panels. The first two show, for each axis, how much the repositioning choice is
worth (headroom) against how distinguishable the best station is from the runner-up. The
third shows what search and the equity weight actually collected in each regime, with
confidence intervals, so "no effect" is visibly different from "not measured precisely".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/regimes.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/regimes.png"))
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    dmexclp = [r for r in rows if r["policy"] == "dmexclp"]

    # The two sweeps share a centre point, and it belongs on *both* axes. Selecting by
    # the `axis` label alone would drop it from whichever sweep did not claim it, leaving
    # that panel a trend line short of its anchor.
    centre_stations = {int(r["stations"]) for r in dmexclp if r["axis"] == "load"}
    centre_rate = {float(r["call_rate"]) for r in dmexclp if r["axis"] == "stations"}
    if len(centre_stations) != 1 or len(centre_rate) != 1:
        raise SystemExit("expected one shared centre point per axis")
    centre_stations, centre_rate = centre_stations.pop(), centre_rate.pop()

    load = sorted(
        (r for r in dmexclp if int(r["stations"]) == centre_stations),
        key=lambda r: float(r["call_rate"]),
    )
    stations = sorted(
        (r for r in dmexclp if float(r["call_rate"]) == centre_rate),
        key=lambda r: int(r["stations"]),
    )

    fig, (ax_load, ax_st, ax_delta) = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, subset, key, xlabel, title in (
        (ax_load, load, "call_rate", "calls per hour",
         "Higher load: repositioning\nmatters less, not more"),
        (ax_st, stations, "stations", "number of stations",
         "More stations: the top two\ncandidates get closer"),
    ):
        x = [float(r[key]) for r in subset]
        ax.bar(
            range(len(x)), [float(r["headroom"]) for r in subset],
            color="tab:blue", alpha=0.75, width=0.55,
            label="headroom (DMEXCLP − random parking)",
        )
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("coverage gained by choosing well")
        ax.set_ylim(0, 0.12)
        ax.set_title(title, fontsize=11)

        twin = ax.twinx()
        twin.plot(
            range(len(x)), [100 * float(r["top2_gap_relative"]) for r in subset],
            color="tab:red", marker="o", lw=2, label="best vs 2nd best (% of best gain)",
        )
        twin.set_ylabel("best vs 2nd-best gain (%)", color="tab:red")
        twin.tick_params(axis="y", labelcolor="tab:red")
        twin.set_ylim(0, 14)

        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, frameon=False, fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=0.3)

    # --- what search and equity actually collected ------------------------------
    arms = [r for r in rows if r["policy"] in ("rollout", "equity")]
    labels, deltas, lows, highs, colours = [], [], [], [], []
    for r in arms:
        labels.append(
            f"{r['policy']}\n{float(r['call_rate']):g}/h, {r['stations']}st"
        )
        d = float(r["delta_vs_dmexclp"])
        deltas.append(d)
        lows.append(d - float(r["delta_ci_low"]))
        highs.append(float(r["delta_ci_high"]) - d)
        colours.append(
            "tab:red" if r["delta_significant"] == "1" else "tab:gray"
        )

    order = np.arange(len(labels))
    ax_delta.axhline(0, color="black", lw=1.2)
    ax_delta.errorbar(
        order, deltas, yerr=[lows, highs], fmt="none", ecolor="#888888", capsize=3
    )
    ax_delta.scatter(order, deltas, c=colours, s=60, zorder=3)
    ax_delta.set_xticks(order)
    ax_delta.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax_delta.set_ylabel("coverage change vs DMEXCLP")
    ax_delta.set_title(
        "Nothing collected the headroom.\nRed = significant, and it is harm", fontsize=11
    )
    ax_delta.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "The negative results are not an artefact of one configuration",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
