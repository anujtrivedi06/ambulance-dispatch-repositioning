"""Phase 2: static base optimisation and dynamic coverage-aware repositioning.

Compares, on identical paired call streams:

  B0  random                          -- sanity floor
  B1  nearest-idle, hand-placed bases -- what a service does by default
  B2a nearest-idle, p-median bases    -- optimised stations, mean-distance objective
  B2b nearest-idle, MEXCLP bases      -- optimised stations, coverage objective
  B3  DMEXCLP on the B2b stations     -- dynamic repositioning on top of B2b
  B3h DMEXCLP on the B1 stations      -- ablation: repositioning without good stations

B3 differs from B2b only in where freed units are sent, which isolates the value of
repositioning from the value of good station placement. B3h repeats that contrast on the
hand-placed stations, to test whether dynamic repositioning can substitute for station
optimisation or only complements it.

    python scripts/run_phase2.py --reps 30 --days 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import (
    expected_covered_demand,
    format_table,
    grid_city,
    mexclp,
    p_median,
    p_median_cost,
)
from ambulance.experiment import aggregate, column, evaluate
from ambulance.policies import DMEXCLP, NearestIdleHomeBase, RandomPolicy
from ambulance.stats import comparison_table, paired_test

METRICS = ["coverage", "mean_response", "p90_response", "worst_zone_coverage"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0)
    p.add_argument(
        "--busy-prob",
        type=float,
        default=0.35,
        help="q in the MEXCLP coverage model; roughly the fleet utilisation",
    )
    args = p.parse_args()

    city = grid_city(
        width=args.grid,
        height=args.grid,
        n_bases=args.stations,
        base_call_rate=args.call_rate,
        seed=7,
    )

    # --- solve the two static location models -------------------------------------
    pm_sites = p_median(city, args.stations)
    mx_sites, mx_alloc = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob, max_sites=args.stations
    )
    pm_city, mx_city = city.with_bases(pm_sites), city.with_bases(mx_sites)
    default_alloc = [city.bases[i % len(city.bases)] for i in range(args.ambulances)]

    print(
        f"city: {args.grid}x{args.grid} grid, {args.stations} stations, "
        f"{args.ambulances} ambulances, {args.call_rate} calls/h, q={args.busy_prob}\n"
    )
    print("station placement")
    print(f"  hand-placed : {sorted(city.bases)}  "
          f"p-median cost {p_median_cost(city, city.bases):.2f}  "
          f"E[cover] {expected_covered_demand(city, default_alloc, args.busy_prob):.3f}")
    print(f"  p-median    : {sorted(pm_sites)}  "
          f"p-median cost {p_median_cost(city, pm_sites):.2f}")
    print(f"  mexclp      : {sorted(mx_sites)}  "
          f"E[cover] {expected_covered_demand(city, mx_alloc, args.busy_prob):.3f}  "
          f"allocation {sorted(mx_alloc)}")
    print()

    runs = {
        "B0 random": (RandomPolicy(seed=0), city, None),
        "B1 nearest/hand": (NearestIdleHomeBase(), city, None),
        "B2a nearest/p-median": (NearestIdleHomeBase(), pm_city, None),
        "B2b nearest/mexclp": (NearestIdleHomeBase(), mx_city, mx_alloc),
        "B3 dmexclp/mexclp": (
            DMEXCLP(busy_prob=args.busy_prob), mx_city, mx_alloc,
        ),
        "B3h dmexclp/hand": (DMEXCLP(busy_prob=args.busy_prob), city, None),
    }

    per_rep = {}
    for name, (policy, run_city, alloc) in runs.items():
        per_rep[name] = evaluate(
            policy,
            run_city,
            days=args.days,
            reps=args.reps,
            n_ambulances=args.ambulances,
            home_bases=alloc,
        )
        print(f"  ran {name}")
    print()
    print(format_table({name: aggregate(r) for name, r in per_rep.items()}))

    for ref, alt in [
        ("B1 nearest/hand", "B2b nearest/mexclp"),
        ("B2b nearest/mexclp", "B3 dmexclp/mexclp"),
        ("B1 nearest/hand", "B3h dmexclp/hand"),
        ("B1 nearest/hand", "B3 dmexclp/mexclp"),
    ]:
        print(f"\n{alt}  vs  {ref}")
        results = [
            paired_test(column(per_rep[ref], m), column(per_rep[alt], m), metric=m)
            for m in METRICS
        ]
        print(comparison_table(results, ref.split()[0], alt.split()[0]))


if __name__ == "__main__":
    main()
