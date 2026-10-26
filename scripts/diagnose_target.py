"""How much does the response-time target decide what any policy can achieve?

    python scripts/diagnose_target.py

Every negative result in this project -- search not paying, learning only matching the
heuristic, equity having no lever -- traces back to one measured quantity: the best and
second-best station differ by about 10% of a single marginal gain, so there is very little
for a method to win. This asks where that number comes from.

It comes from the geometry. With 1.5-minute blocks on an 8x8 grid, the median trip between
any two nodes is 7.5 minutes, against an 8-minute target. One station already reaches over
half the demand, so a second one adds little that the first did not already cover, and
candidate stations end up nearly interchangeable.

Tightening the target changes that, and the sweep below shows by how much. This is a
statement about the testbed, not about ambulance dispatch, and it belongs in the
limitations of any claim made from it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city, mexclp
from ambulance.location import coverage_matrix
from ambulance.policies import DMEXCLP


def measure(city, target: float, args) -> dict[str, float]:
    sites, allocation = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob,
        max_sites=args.stations, target=target,
    )
    run_city = city.with_bases(sites)
    cov = coverage_matrix(run_city, target)
    demand = run_city.demand_weights

    probe = DMEXCLP(busy_prob=args.busy_prob, target=target)
    gaps: list[float] = []
    bests: list[float] = []

    class Probe(DMEXCLP):
        def reposition(self, sim, amb_id):
            ranked = sorted(probe.gains(sim, amb_id).values(), reverse=True)
            # Skip decisions where even the best option adds nothing: at a tight target
            # the marginal gain can be zero everywhere, and a ratio against zero is noise
            # rather than a measurement.
            if len(ranked) >= 2 and ranked[0] > 1e-12:
                gaps.append(ranked[0] - ranked[1])
                bests.append(ranked[0])
            return super().reposition(sim, amb_id)

    calls = generate_calls(run_city, args.days * 24 * 60, np.random.default_rng(1000))
    Simulator(
        run_city, calls, n_ambulances=args.ambulances, seed=2000, home_bases=allocation
    ).run(Probe(busy_prob=args.busy_prob, target=target))

    n = len(run_city.nodes)
    return {
        "target_minutes": target,
        "pairs_within_target": float(cov.mean()),
        "demand_one_station_reaches": float(
            np.mean([demand[cov[i]].sum() for i in range(n)])
        ),
        "top2_gap_relative": float(np.mean(gaps) / np.mean(bests)) if bests else 0.0,
        "decisions_scored": len(gaps),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="4,5,6.5,8")
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0)
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out-csv", type=Path, default=Path("results/target_sweep.csv"))
    args = p.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=7,
    )
    median_trip = float(np.median(city.travel_matrix()))
    print(
        f"{args.grid}x{args.grid} grid: median trip between any two nodes "
        f"{median_trip:.2f} min\n"
    )
    head = (
        f"{'target':>8}{'pairs in reach':>16}{'1 station reaches':>19}"
        f"{'best vs 2nd':>14}"
    )
    print(head)
    print("-" * len(head))

    rows = []
    for text in args.targets.split(","):
        row = measure(city, float(text), args)
        rows.append(row)
        print(
            f"{row['target_minutes']:>8.1f}"
            f"{row['pairs_within_target']:>15.1%}"
            f"{row['demand_one_station_reaches']:>19.1%}"
            f"{row['top2_gap_relative']:>13.1%}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out_csv}")
    print(
        "\nThe 8-minute target used throughout this project is the generous end of this\n"
        "range. The negative results should be read as holding there, and the tighter\n"
        "targets are where a search or learning method has the most room to prove itself."
    )


if __name__ == "__main__":
    main()
