"""Phase 3: how much does online search buy, per unit of compute?

Rollout and MCTS both approximate the value of a repositioning action by simulating
futures. Both are therefore limited not by the idea but by the variance of the estimate
they build from a small number of samples. This script measures the quality/compute
trade-off directly: the same policy at a range of simulation budgets, against the DMEXCLP
base policy it is rolling out, with wall-clock cost reported alongside.

    python scripts/run_phase3.py --methods rollout --budgets 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import format_table, grid_city, mexclp
from ambulance.experiment import aggregate, column, evaluate
from ambulance.policies import DMEXCLP, RolloutPolicy
from ambulance.stats import comparison_table, paired_test

METRICS = ["coverage", "mean_response", "p90_response", "worst_zone_coverage"]


def build(method: str, budget: int, args) -> object:
    if method == "rollout":
        return RolloutPolicy(
            n_sims=budget,
            horizon=args.horizon,
            n_candidates=args.candidates,
            value_mode=args.value_mode,
            busy_prob=args.busy_prob,
        )
    if method == "mcts":
        from ambulance.policies import MCTSPolicy

        return MCTSPolicy(
            n_sims=budget,
            horizon=args.horizon,
            depth=args.depth,
            n_candidates=args.candidates,
            value_mode=args.value_mode,
            busy_prob=args.busy_prob,
        )
    raise ValueError(f"unknown method {method!r}")


def _write_csv(path: Path, per_rep, cost, args) -> None:
    """One row per policy, so the anytime curve can be plotted without re-running."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["policy", "method", "budget", "forks_per_rep", "seconds"]
            + METRICS
            + [f"{m}_sd" for m in METRICS]
        )
        for name, reps in per_rep.items():
            agg = aggregate(reps)
            elapsed, sims = cost[name]
            method = name.split("(")[0].strip()
            budget = ""
            for part in name.replace(")", "").split(","):
                if part.startswith(("m=", "n=")):
                    budget = part[2:]
            writer.writerow(
                [name, method, budget, f"{sims / args.reps:.0f}", f"{elapsed:.2f}"]
                + [f"{agg[m]:.5f}" for m in METRICS]
                + [f"{column(reps, m).std(ddof=1):.5f}" for m in METRICS]
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0)
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument("--methods", default="rollout,mcts")
    p.add_argument("--budgets", default="1,2,4,8,16,32")
    p.add_argument("--horizon", type=float, default=240.0, help="lookahead minutes")
    p.add_argument("--candidates", type=int, default=2, help="stations shortlisted")
    p.add_argument("--depth", type=int, default=2, help="MCTS tree depth")
    p.add_argument("--value-mode", default="coverage", choices=["coverage", "response"])
    p.add_argument("--out", type=Path, default=None, help="write per-policy results CSV")
    args = p.parse_args()

    city = grid_city(
        width=args.grid,
        height=args.grid,
        n_bases=args.stations,
        base_call_rate=args.call_rate,
        seed=7,
    )
    sites, alloc = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob, max_sites=args.stations
    )
    run_city = city.with_bases(sites)
    kw = dict(
        days=args.days,
        reps=args.reps,
        n_ambulances=args.ambulances,
        home_bases=alloc,
    )

    print(
        f"city: {args.grid}x{args.grid}, {args.stations} stations (MEXCLP), "
        f"{args.ambulances} ambulances, {args.call_rate} calls/h\n"
        f"search: horizon {args.horizon:g} min, {args.candidates} candidates, "
        f"value={args.value_mode}, {args.reps} reps x {args.days} days\n"
    )

    budgets = [int(b) for b in args.budgets.split(",") if b]
    methods = [m for m in args.methods.split(",") if m]

    per_rep: dict[str, list] = {}
    cost: dict[str, tuple[float, float]] = {}

    t0 = time.perf_counter()
    base = DMEXCLP(busy_prob=args.busy_prob)
    per_rep["dmexclp (base)"] = evaluate(base, run_city, **kw)
    cost["dmexclp (base)"] = (time.perf_counter() - t0, 0.0)
    print(f"  dmexclp (base)            {cost['dmexclp (base)'][0]:6.1f}s")

    for method in methods:
        for budget in budgets:
            policy = build(method, budget, args)
            t0 = time.perf_counter()
            per_rep[policy.name] = evaluate(policy, run_city, **kw)
            elapsed = time.perf_counter() - t0
            cost[policy.name] = (elapsed, float(policy.sims_run))
            print(
                f"  {policy.name:<26}{elapsed:6.1f}s"
                f"  {policy.sims_run / args.reps:8.0f} sims/rep"
            )

    print()
    print(format_table({name: aggregate(r) for name, r in per_rep.items()}))

    print("\ncost")
    head = f"{'policy':<28}{'wall-clock':>12}{'sims/rep':>12}{'x base cost':>14}"
    print(head)
    print("-" * len(head))
    base_cost = cost["dmexclp (base)"][0]
    for name, (elapsed, sims) in cost.items():
        print(f"{name:<28}{elapsed:>11.1f}s{sims / args.reps:>12.0f}"
              f"{elapsed / base_cost:>13.1f}x")

    if args.out is not None:
        _write_csv(args.out, per_rep, cost, args)
        print(f"\nwrote {args.out}")

    print("\nsignificance vs the base policy being rolled out")
    for name in per_rep:
        if name == "dmexclp (base)":
            continue
        results = [
            paired_test(
                column(per_rep["dmexclp (base)"], m), column(per_rep[name], m), metric=m
            )
            for m in METRICS
        ]
        print(f"\n{name}  vs  dmexclp (base)")
        print(comparison_table(results, "dmexclp", name.split("(")[0]))


if __name__ == "__main__":
    main()
