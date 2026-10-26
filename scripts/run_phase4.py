"""Phase 4: learning a repositioning policy, and what the training signal costs you.

Five learners, one feature set, one simulator:

  ADP-LSTD (coverage)   linear value function fitted by least-squares TD
  ADP-LSTD (response)   the same, with a continuous reward
  LSPI                  policy iteration on top of LSTD
  CEM                   direct search over the *same linear class*, no value function
  Deep NFQ (coverage)   neural value function, richer features, binary reward
  Deep NFQ (response)   the same network with a continuous reward

The comparison is built to separate two things that are easy to confuse: the function
class (linear vs neural) and the training signal (predict value vs optimise the policy,
binary reward vs continuous). CEM shares its hypothesis class with ADP-LSTD exactly, so
any gap between them is attributable to the objective and not the representation.

Training and evaluation use disjoint call-stream seeds, so nothing here is scored on the
episodes it was fitted to.

    python scripts/run_phase4.py --reps 30 --days 7
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import format_table, grid_city, mexclp
from ambulance.adp import CollectingPolicy, collect, fit_lstd, train_lspi
from ambulance.deeprl import DeepValuePolicy, train_nfq
from ambulance.experiment import aggregate, column, evaluate
from ambulance.features import FEATURE_NAMES, FeatureExtractor
from ambulance.policies import ADPPolicy, DMEXCLP
from ambulance.policy_search import ACTION_FEATURES, cross_entropy_search
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
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument("--train-episodes", type=int, default=40)
    p.add_argument("--train-days", type=int, default=3)
    p.add_argument("--cem-iterations", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("results/phase4_weights.json"))
    p.add_argument("--out-csv", type=Path, default=None, help="write per-policy metrics")
    args = p.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=7,
    )
    sites, alloc = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob, max_sites=args.stations
    )
    run_city = city.with_bases(sites)
    extractor = FeatureExtractor(run_city, busy_prob=args.busy_prob)
    train_kw = dict(
        home_bases=alloc, n_ambulances=args.ambulances, days=args.train_days
    )
    eval_kw = dict(
        days=args.days, reps=args.reps, n_ambulances=args.ambulances, home_bases=alloc
    )

    print(
        f"city: {args.grid}x{args.grid}, {args.stations} MEXCLP stations, "
        f"{args.ambulances} ambulances, {args.call_rate} calls/h\n"
        f"train: {args.train_episodes} episodes x {args.train_days} days  |  "
        f"eval: {args.reps} paired reps x {args.days} days (disjoint seeds)\n"
    )

    policies: dict[str, object] = {"dmexclp (base)": DMEXCLP(busy_prob=args.busy_prob)}
    weights: dict[str, list[float]] = {}

    for mode in ("coverage", "response"):
        t0 = time.perf_counter()
        transitions = collect(
            run_city,
            CollectingPolicy(extractor, epsilon=0.3, seed=0),
            episodes=args.train_episodes,
            reward_mode=mode,
            **train_kw,
        )
        theta = fit_lstd(transitions)
        weights[f"adp-lstd ({mode})"] = theta.tolist()
        policies[f"adp-lstd ({mode})"] = ADPPolicy(theta, extractor, label=f"adp-lstd ({mode})")
        print(f"  fitted adp-lstd ({mode}): {len(transitions)} transitions,"
              f" {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    history = train_lspi(
        run_city, extractor, iterations=4,
        episodes=max(10, args.train_episodes // 4), reward_mode="coverage", **train_kw,
    )
    weights["lspi"] = history[-1].tolist()
    policies["lspi"] = ADPPolicy(history[-1], extractor, label="lspi")
    print(f"  fitted lspi: {len(history)} iterations, {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    search = cross_entropy_search(
        run_city, extractor, iterations=args.cem_iterations, population=20,
        episodes=3, seed=0, **train_kw,
    )
    weights["cem"] = search.theta.tolist()
    policies["cem"] = ADPPolicy(search.theta, extractor, label="cem")
    print(f"  searched cem: best train score {search.score:.3f},"
          f" {time.perf_counter() - t0:.0f}s")

    for mode, extra in (("coverage", {}), ("response", {"fresh_buffer": True, "epsilon": 0.5})):
        t0 = time.perf_counter()
        net, hist = train_nfq(
            run_city, extractor, rounds=5, episodes=20, epochs=60,
            reward_mode=mode, seed=0, **train_kw, **extra,
        )
        policies[f"deep-nfq ({mode})"] = DeepValuePolicy(
            net, extractor, label=f"deep-nfq ({mode})"
        )
        print(f"  trained deep-nfq ({mode}): {hist[-1]['transitions']} transitions,"
              f" loss {hist[-1]['loss']:.3f}, {time.perf_counter() - t0:.0f}s")

    print("\nlearned weights on the features an action can actually change")
    head = f"{'policy':<24}" + "".join(f"{n[:14]:>16}" for n in ACTION_FEATURES)
    print(head)
    print("-" * len(head))
    for name, theta in weights.items():
        row = [theta[FEATURE_NAMES.index(n)] for n in ACTION_FEATURES]
        print(f"{name:<24}" + "".join(f"{v:>16.3f}" for v in row))

    print()
    per_rep = {}
    for name, policy in policies.items():
        per_rep[name] = evaluate(policy, run_city, **eval_kw)
        print(f"  evaluated {name}")
    print()
    print(format_table({name: aggregate(r) for name, r in per_rep.items()}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"features": list(FEATURE_NAMES), "weights": weights}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["policy", "family", "reward"] + METRICS + [f"{m}_sd" for m in METRICS]
            )
            for name, reps in per_rep.items():
                agg = aggregate(reps)
                family = name.split(" (")[0]
                reward = name.split("(")[1].rstrip(")") if "(" in name else ""
                writer.writerow(
                    [name, family, reward]
                    + [f"{agg[m]:.5f}" for m in METRICS]
                    + [f"{column(reps, m).std(ddof=1):.5f}" for m in METRICS]
                )
        print(f"wrote {args.out_csv}")

    print("\nsignificance vs the DMEXCLP heuristic")
    for name in per_rep:
        if name == "dmexclp (base)":
            continue
        results = [
            paired_test(
                column(per_rep["dmexclp (base)"], m), column(per_rep[name], m), metric=m
            )
            for m in METRICS
        ]
        print(f"\n{name}  vs  dmexclp")
        print(comparison_table(results, "dmexclp", name.split()[0]))


if __name__ == "__main__":
    main()
