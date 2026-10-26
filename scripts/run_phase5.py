"""Phase 5: the two trade-offs the earlier phases kept running into.

1. **Equity vs efficiency.** Sweeping the weight in `EquityDMEXCLP` traces what city-wide
   coverage costs to protect the worst-served zone. At lam = 0 the policy is DMEXCLP
   exactly, so the frontier starts at the Phase 2/3/4 result.

2. **Priority and selective dispatch.** Sweeping the reserve threshold measures whether
   holding a unit back from a routine call is ever worth it, read off the two urgency
   classes separately rather than from a pooled average that hides the whole question.

    python scripts/run_phase5.py --reps 30 --days 7 --out-csv results/phase5.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, format_table, generate_calls, grid_city, mexclp
from ambulance.experiment import aggregate, column, evaluate
from ambulance.policies import DMEXCLP, EquityDMEXCLP, SelectiveDispatch
from ambulance.stats import comparison_table, paired_test

EQUITY_METRICS = ["coverage", "worst_zone_coverage", "zone_coverage_gap", "mean_response"]
PRIORITY_METRICS = ["coverage_p1", "coverage_p2", "mean_response_p1", "mean_response_p2"]
CSV_METRICS = sorted(set(EQUITY_METRICS + PRIORITY_METRICS + ["p90_response"]))


def pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Names of the points not dominated on both objectives (higher is better)."""
    front = set()
    for name, x, y in points:
        if not any(
            (ox >= x and oy >= y and (ox > x or oy > y))
            for other, ox, oy in points
            if other != name
        ):
            front.add(name)
    return front


def measure_decision_agreement(
    city, lambdas: list[float], home_bases, n_ambulances: int, days: int, busy_prob: float
) -> list[tuple[float, float, int]]:
    """How often each lambda picks a different station than plain DMEXCLP would.

    Without this, a flat set of outcomes has two very different explanations: the equity
    weight never changes the decision, or it changes the decision and the decision does
    not matter. Those call for opposite conclusions, so the run measures it rather than
    leaving it to be argued.
    """
    probes = {lam: EquityDMEXCLP(lam=lam, busy_prob=busy_prob) for lam in lambdas}
    agree = {lam: 0 for lam in lambdas}
    total = 0

    class Probe(DMEXCLP):
        def reposition(self, sim, amb_id):
            nonlocal total
            choices = {lam: p.reposition(sim, amb_id) for lam, p in probes.items()}
            baseline_choice = choices[lambdas[0]]
            total += 1
            for lam in lambdas:
                if choices[lam] == baseline_choice:
                    agree[lam] += 1
            return baseline_choice

    calls = generate_calls(city, days * 24 * 60, np.random.default_rng(1000))
    sim = Simulator(
        city, calls, n_ambulances=n_ambulances, seed=2000, home_bases=home_bases
    )
    sim.run(Probe())
    return [(lam, agree[lam] / max(total, 1), total) for lam in lambdas]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0)
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument(
        "--lambdas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.85,1.0"
    )
    p.add_argument("--reserves", default="0,1,2,3,4")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=7,
    )
    sites, alloc = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob, max_sites=args.stations
    )
    run_city = city.with_bases(sites)
    kw = dict(
        days=args.days, reps=args.reps, n_ambulances=args.ambulances, home_bases=alloc
    )

    print(
        f"city: {args.grid}x{args.grid}, {args.stations} MEXCLP stations, "
        f"{args.ambulances} ambulances, {args.call_rate} calls/h\n"
        f"eval: {args.reps} paired reps x {args.days} days\n"
    )

    per_rep: dict[str, list] = {}
    family: dict[str, str] = {}
    knob: dict[str, float] = {}

    for lam_text in args.lambdas.split(","):
        lam = float(lam_text)
        policy = EquityDMEXCLP(lam=lam, busy_prob=args.busy_prob)
        per_rep[policy.name] = evaluate(policy, run_city, **kw)
        family[policy.name], knob[policy.name] = "equity", lam
        print(f"  ran {policy.name}")

    for reserve_text in args.reserves.split(","):
        reserve = int(reserve_text)
        policy = SelectiveDispatch(DMEXCLP(busy_prob=args.busy_prob), reserve=reserve)
        per_rep[policy.name] = evaluate(policy, run_city, **kw)
        family[policy.name], knob[policy.name] = "selective", float(reserve)
        print(f"  ran {policy.name}")

    aggregates = {name: aggregate(reps) for name, reps in per_rep.items()}
    baseline = "equity(lam=0)"

    print("\nequity / efficiency")
    equity_names = [n for n in per_rep if family[n] == "equity"]
    head = f"{'policy':<20}" + "".join(m[:18].rjust(20) for m in EQUITY_METRICS)
    print(head)
    print("-" * len(head))
    points = [
        (n, aggregates[n]["coverage"], aggregates[n]["worst_zone_coverage"])
        for n in equity_names
    ]
    front = pareto_front(points)
    for name in equity_names:
        mark = " *" if name in front else ""
        print(
            f"{name:<20}"
            + "".join(f"{aggregates[name][m]:>20.4f}" for m in EQUITY_METRICS)
            + mark
        )
    print("* = on the Pareto front for (coverage, worst-zone coverage)")

    print("\npriority / selective dispatch")
    sel_names = [n for n in per_rep if family[n] == "selective"]
    head = f"{'policy':<22}" + "".join(m[:18].rjust(20) for m in PRIORITY_METRICS)
    print(head)
    print("-" * len(head))
    for name in sel_names:
        print(
            f"{name:<22}"
            + "".join(f"{aggregates[name][m]:>20.4f}" for m in PRIORITY_METRICS)
        )

    print()
    print(format_table(aggregates))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["policy", "family", "knob", "on_pareto_front"]
                + CSV_METRICS
                + [f"{m}_sd" for m in CSV_METRICS]
            )
            for name, reps in per_rep.items():
                agg = aggregates[name]
                # The front is defined over the equity sweep only; a 0 against a
                # selective-dispatch row would read as "dominated" when it was never a
                # candidate, so those are left blank.
                on_front = int(name in front) if family[name] == "equity" else ""
                writer.writerow(
                    [name, family[name], knob[name], on_front]
                    + [f"{agg[m]:.5f}" for m in CSV_METRICS]
                    + [f"{column(reps, m).std(ddof=1):.5f}" for m in CSV_METRICS]
                )
        print(f"\nwrote {args.out_csv}")

    lambdas = [float(t) for t in args.lambdas.split(",")]
    agreement = measure_decision_agreement(
        run_city, lambdas, alloc, args.ambulances, args.days, args.busy_prob
    )
    print("\nhow often each lambda picks a different station than lam=0")
    head = f"{'lambda':>10}{'same choice':>16}{'decisions':>14}"
    print(head)
    print("-" * len(head))
    for lam, same, total_decisions in agreement:
        print(f"{lam:>10.2f}{same:>15.1%}{total_decisions:>14d}")
    if args.out_csv is not None:
        path = args.out_csv.with_name(args.out_csv.stem + "_decisions.csv")
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["lambda", "same_choice_fraction", "decisions"])
            writer.writerows(
                [[f"{lam:g}", f"{same:.5f}", n] for lam, same, n in agreement]
            )
        print(f"wrote {path}")

    print(f"\nsignificance vs {baseline} (plain DMEXCLP)")
    for name in per_rep:
        if name == baseline:
            continue
        metrics = EQUITY_METRICS if family[name] == "equity" else PRIORITY_METRICS
        results = [
            paired_test(column(per_rep[baseline], m), column(per_rep[name], m), metric=m)
            for m in metrics
        ]
        print(f"\n{name}  vs  {baseline}")
        print(comparison_table(results, "dmexclp", name.split("(")[0]))


if __name__ == "__main__":
    main()
