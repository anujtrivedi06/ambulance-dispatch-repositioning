"""Evaluate Phase-1 policies on paired, seeded scenarios.

Every policy sees the identical call stream in each replication, so differences between
them are not swamped by the variance of the arrival process.

    python scripts/run_baseline.py --days 7 --reps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, format_table, generate_calls, grid_city, summarise
from ambulance.policies import NearestIdleHomeBase, RandomPolicy


def evaluate(policy, city, days: int, reps: int, n_ambulances: int) -> dict[str, float]:
    """Mean metrics over `reps` replications of a `days`-long horizon."""
    per_rep = []
    for rep in range(reps):
        calls = generate_calls(city, days * 24 * 60, np.random.default_rng(1000 + rep))
        policy.reset()
        sim = Simulator(city, calls, n_ambulances=n_ambulances, seed=2000 + rep)
        records = sim.run(policy)
        per_rep.append(summarise(records, city))
    return {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--reps", type=int, default=10)
    # 6 units at 4 calls/h puts the system near 65% utilisation: enough slack that
    # repositioning decisions matter, enough load that bad ones cost response time.
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0, help="mean calls/hour")
    args = p.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, base_call_rate=args.call_rate, seed=7
    )
    print(
        f"city: {args.grid}x{args.grid} grid, {len(city.bases)} bases, "
        f"{len(city.hospitals)} hospitals, {args.ambulances} ambulances, "
        f"{args.call_rate} calls/h\n"
    )

    policies = [RandomPolicy(seed=0), NearestIdleHomeBase()]
    rows = {
        pol.name: evaluate(pol, city, args.days, args.reps, args.ambulances)
        for pol in policies
    }
    print(format_table(rows))


if __name__ == "__main__":
    main()
