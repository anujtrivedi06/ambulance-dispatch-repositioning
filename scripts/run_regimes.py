"""Does the configuration explain the Phase 3 and Phase 5 negatives?

Three phases concluded that sophisticated methods do not pay on this problem, and all
three rested on one unverified claim: that the candidate stations are near-equivalent in
outcome terms *at the configuration tested* -- 4 stations, 6 ambulances, 65% utilisation.
That was a caveat, not a result. This script makes it a result.

The key quantity is **headroom**: the coverage gap between the best known repositioning
rule (DMEXCLP) and parking idle units at random, holding dispatch fixed. Headroom is how
much the repositioning choice is worth at all. Where it is near zero, no method can win
however clever, and the Phase 3 and 5 negatives are statements about the configuration.
Where it is large, a method has something to compete for -- and if search and equity still
fail to collect it, those negatives are about the methods.

Two one-dimensional sweeps through a shared centre point, so each axis is interpretable
on its own:

  load     call rate 4, 5, 6 per hour at 4 stations   (utilisation ~65%, ~81%, ~97%)
  stations 4, 6, 8 stations at 4 calls per hour       (a finer action space)

    python scripts/run_regimes.py --reps 8 --days 3 --out-csv results/regimes.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import grid_city, mexclp, p_median
from ambulance.experiment import aggregate, column, evaluate
from ambulance.policies import (
    DMEXCLP,
    EquityDMEXCLP,
    RandomReposition,
    RolloutPolicy,
)
from ambulance.stats import paired_test

METRICS = ["coverage", "mean_response", "worst_zone_coverage", "queued_frac"]


def build_regime(args, call_rate: float, n_stations: int):
    """A city with `n_stations` stations, and a home allocation across them.

    Stations come from p-median so the action space is a well-spread set of the requested
    size. MEXCLP then allocates the fleet among exactly those sites, rather than choosing
    its own: otherwise it would open at most one site per ambulance and the station axis
    could never exceed the fleet size, which is the whole point of the sweep.
    """
    city = grid_city(
        width=args.grid,
        height=args.grid,
        n_bases=max(n_stations, 2),
        base_call_rate=call_rate,
        seed=7,
    )
    sites = p_median(city, n_stations)
    _, allocation = mexclp(
        city,
        args.ambulances,
        busy_prob=args.busy_prob,
        candidates=sites,
        max_sites=n_stations,
    )
    return city.with_bases(sorted(sites)), allocation


def measure_candidate_gap(city, allocation, args, call_rate: float) -> dict[str, float]:
    """How distinguishable the best station is from the runner-up, per decision.

    Headroom (best vs random) and this gap (best vs second-best) answer different
    questions, and conflating them would make the sweep meaningless. Repositioning can
    matter enormously in aggregate while the top two candidates are indistinguishable --
    and it is the second quantity that decides whether *search* can pay, because search
    never chooses between the best option and a random one, only between close rivals.
    """
    import numpy as np

    from ambulance import generate_calls
    from ambulance.sim import Simulator

    probe_policy = DMEXCLP(busy_prob=args.busy_prob)
    gaps: list[float] = []
    bests: list[float] = []

    class Probe(DMEXCLP):
        def reposition(self, sim, amb_id):
            ranked = sorted(probe_policy.gains(sim, amb_id).values(), reverse=True)
            if len(ranked) >= 2:
                gaps.append(ranked[0] - ranked[1])
                bests.append(ranked[0])
            return super().reposition(sim, amb_id)

    calls = generate_calls(
        city, args.days * 24 * 60, np.random.default_rng(1000)
    )
    sim = Simulator(
        city,
        calls,
        n_ambulances=args.ambulances,
        seed=2000,
        home_bases=allocation,
    )
    sim.run(Probe(busy_prob=args.busy_prob))
    if not gaps:
        return {"top2_gap": 0.0, "top2_gap_relative": 0.0, "decisions": 0}
    mean_gap = float(np.mean(gaps))
    mean_best = float(np.mean(bests))
    return {
        "top2_gap": mean_gap,
        "top2_gap_relative": mean_gap / mean_best if mean_best > 0 else 0.0,
        "decisions": len(gaps),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument("--call-rates", default="4,5,6")
    p.add_argument("--station-counts", default="4,6,8")
    p.add_argument("--centre-rate", type=float, default=4.0)
    p.add_argument("--centre-stations", type=int, default=4)
    p.add_argument("--rollout-sims", type=int, default=4)
    p.add_argument("--no-rollout", action="store_true", help="skip the expensive arm")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    regimes: list[tuple[str, float, int]] = []
    for text in args.call_rates.split(","):
        rate = float(text)
        regimes.append(("load", rate, args.centre_stations))
    for text in args.station_counts.split(","):
        count = int(text)
        if (args.centre_rate, count) not in [(r, s) for _, r, s in regimes]:
            regimes.append(("stations", args.centre_rate, count))

    print(
        f"{args.ambulances} ambulances, {args.grid}x{args.grid} grid, "
        f"{args.reps} paired reps x {args.days} days per regime\n"
    )

    rows = []
    for axis, rate, stations in regimes:
        city, allocation = build_regime(args, rate, stations)
        kw = dict(
            days=args.days,
            reps=args.reps,
            n_ambulances=args.ambulances,
            home_bases=allocation,
        )
        label = f"{rate:g}/h, {stations} stations"
        print(f"regime {label:22s} (axis: {axis})")

        arms = {
            "random-reposition": RandomReposition(seed=0),
            "dmexclp": DMEXCLP(busy_prob=args.busy_prob),
            "equity": EquityDMEXCLP(lam=0.5, busy_prob=args.busy_prob),
        }
        if not args.no_rollout:
            arms["rollout"] = RolloutPolicy(
                n_sims=args.rollout_sims,
                horizon=240.0,
                n_candidates=2,
                busy_prob=args.busy_prob,
            )

        per_rep = {}
        for name, policy in arms.items():
            start = time.perf_counter()
            per_rep[name] = evaluate(policy, city, **kw)
            print(f"    {name:20s}{time.perf_counter() - start:7.1f}s")

        agg = {name: aggregate(reps) for name, reps in per_rep.items()}
        headroom = agg["dmexclp"]["coverage"] - agg["random-reposition"]["coverage"]

        # Everything is measured against DMEXCLP, the reference every earlier phase used.
        deltas = {}
        for name in arms:
            if name == "dmexclp":
                continue
            test = paired_test(
                column(per_rep["dmexclp"], "coverage"),
                column(per_rep[name], "coverage"),
                metric="coverage",
            )
            deltas[name] = test

        gap = measure_candidate_gap(city, allocation, args, rate)
        print(
            f"    headroom (dmexclp - random) {headroom:+.4f}"
            f"   utilisation proxy (queued) {agg['dmexclp']['queued_frac']:.3f}"
        )
        print(
            f"    best-vs-second-best gain gap {gap['top2_gap']:.5f}"
            f"  ({gap['top2_gap_relative']:.1%} of the best gain,"
            f" {gap['decisions']} decisions)"
        )
        for name, test in deltas.items():
            star = "*" if test.significant else " "
            print(
                f"    {name:20s}{test.mean_diff:+.4f}"
                f"  [{test.ci_low:+.4f},{test.ci_high:+.4f}]  p={test.t_p:.3g} {star}"
            )
        print()

        for name in arms:
            rows.append(
                {
                    "axis": axis,
                    "call_rate": rate,
                    "stations": stations,
                    "policy": name,
                    "headroom": headroom,
                    "top2_gap": gap["top2_gap"],
                    "top2_gap_relative": gap["top2_gap_relative"],
                    **{m: agg[name][m] for m in METRICS},
                    "coverage_sd": column(per_rep[name], "coverage").std(ddof=1),
                    "delta_vs_dmexclp": 0.0 if name == "dmexclp" else deltas[name].mean_diff,
                    "delta_significant": ""
                    if name == "dmexclp"
                    else int(deltas[name].significant),
                }
            )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
