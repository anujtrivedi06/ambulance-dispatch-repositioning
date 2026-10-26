"""Train the RL agent across several seeds and report the spread.

A single training run is not a result. Neural fitted value iteration depends on network
initialisation, on the epsilon-greedy data it happens to collect, and on which round the
validation block happens to favour; two seeds of the same configuration can differ by more
than the effect being claimed. This trains the same agent under several seeds and reports
the mean and standard deviation across them, so the headline number carries its own
uncertainty rather than being the one run that happened to be saved.

    python scripts/train_multiseed.py --seeds 5 --rounds 8

Each seed selects its own checkpoint on a validation block disjoint from training and test
episodes, and is then scored on the untouched test block. The agent written to disk is the
one with the best *validation* score across seeds -- never the best test score, which
would make the reported figure a best-of-N.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import grid_city, mexclp
from ambulance.deeprl import DeepValuePolicy, save_agent, train_nfq
from ambulance.experiment import aggregate, column, evaluate
from ambulance.features import FeatureExtractor
from ambulance.policies import DMEXCLP, RandomReposition
from ambulance.stats import paired_test

METRICS = ["coverage", "mean_response", "p90_response", "worst_zone_coverage"]
VALIDATION_OFFSET = 50_000


def train_one(seed: int, run_city, extractor, allocation, args):
    """Train one agent, selecting its checkpoint on validation episodes."""
    best = {"coverage": -1.0, "round": 0, "state": None}

    def curve_eval(round_index: int, net) -> dict[str, float]:
        scores = aggregate(
            evaluate(
                DeepValuePolicy(net, extractor), run_city, days=3,
                reps=args.curve_reps, n_ambulances=args.ambulances,
                home_bases=allocation, seed_offset=VALIDATION_OFFSET,
            )
        )
        if scores["coverage"] > best["coverage"]:
            best.update(
                coverage=scores["coverage"],
                round=round_index,
                state={k: v.clone() for k, v in net.state_dict().items()},
            )
        return {f"eval_{m}": scores[m] for m in METRICS}

    net, history = train_nfq(
        run_city, extractor,
        rounds=args.rounds, episodes=args.episodes, epochs=args.epochs,
        reward_mode=args.reward, epsilon=args.epsilon, fresh_buffer=True,
        seed=seed, eval_fn=curve_eval,
        home_bases=allocation, n_ambulances=args.ambulances, days=3,
    )
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return net, history, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ambulances", type=int, default=6)
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--call-rate", type=float, default=4.0)
    p.add_argument("--busy-prob", type=float, default=0.35)
    p.add_argument("--reward", default="response", choices=["coverage", "response"])
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--curve-reps", type=int, default=6)
    p.add_argument("--test-reps", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("models/nfq_agent.pt"))
    p.add_argument("--csv", type=Path, default=Path("results/seed_spread.csv"))
    args = p.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=7,
    )
    sites, allocation = mexclp(
        city, args.ambulances, busy_prob=args.busy_prob, max_sites=args.stations
    )
    run_city = city.with_bases(sites)
    extractor = FeatureExtractor(run_city, busy_prob=args.busy_prob)
    shifted = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=21,
    ).with_bases(sites)
    shifted_extractor = FeatureExtractor(shifted, busy_prob=args.busy_prob)
    test_kw = dict(
        days=7, reps=args.test_reps, n_ambulances=args.ambulances,
        home_bases=allocation,
    )

    print(
        f"training {args.seeds} seeds x {args.rounds} rounds, reward={args.reward}\n"
        f"validation episodes offset {VALIDATION_OFFSET}, test episodes offset 0\n"
    )

    baseline_runs = evaluate(DMEXCLP(busy_prob=args.busy_prob), run_city, **test_kw)
    floor_runs = evaluate(RandomReposition(seed=0), run_city, **test_kw)
    baseline = aggregate(baseline_runs)
    print(
        f"baselines   DMEXCLP {baseline['coverage']:.4f}   "
        f"random parking {aggregate(floor_runs)['coverage']:.4f}\n"
    )

    rows, best_overall = [], {"val": -1.0, "seed": None, "net": None, "history": None}
    for seed in range(args.seeds):
        net, history, best = train_one(seed, run_city, extractor, allocation, args)
        test = aggregate(evaluate(DeepValuePolicy(net, extractor), run_city, **test_kw))
        shift = aggregate(
            evaluate(DeepValuePolicy(net, shifted_extractor), shifted, **test_kw)
        )
        rows.append(
            {
                "seed": seed,
                "selected_round": best["round"],
                "validation_coverage": best["coverage"],
                **{f"test_{m}": test[m] for m in METRICS},
                "shifted_coverage": shift["coverage"],
            }
        )
        print(
            f"  seed {seed}  round {best['round']}  "
            f"val {best['coverage']:.4f}  test {test['coverage']:.4f}  "
            f"shifted {shift['coverage']:.4f}"
        )
        # Selection on validation only. Picking by test score would turn the reported
        # number into a maximum over seeds rather than an estimate of the method.
        if best["coverage"] > best_overall["val"]:
            best_overall.update(
                val=best["coverage"], seed=seed, net=net, history=history
            )

    cov = np.array([r["test_coverage"] for r in rows])
    mean_resp = np.array([r["test_mean_response"] for r in rows])
    shifted_cov = np.array([r["shifted_coverage"] for r in rows])

    print(
        f"\nacross {args.seeds} seeds"
        f"\n  test coverage      {cov.mean():.4f} +/- {cov.std(ddof=1):.4f}"
        f"   (range {cov.min():.4f} to {cov.max():.4f})"
        f"\n  test mean response {mean_resp.mean():.3f} +/- {mean_resp.std(ddof=1):.3f}"
        f"\n  shifted coverage   {shifted_cov.mean():.4f} +/- {shifted_cov.std(ddof=1):.4f}"
        f"\n  DMEXCLP baseline   {baseline['coverage']:.4f}"
        f"\n  seeds beating the baseline: "
        f"{int((cov > baseline['coverage']).sum())} of {args.seeds}"
    )

    # The paired test uses the selected agent, since that is the one being submitted.
    selected = DeepValuePolicy(best_overall["net"], extractor)
    selected_runs = evaluate(selected, run_city, **test_kw)
    print(f"\nselected agent: seed {best_overall['seed']} (best validation score)")
    for metric in METRICS:
        result = paired_test(
            column(baseline_runs, metric), column(selected_runs, metric), metric=metric
        )
        star = "*" if result.significant else " "
        print(
            f"  {metric:22s}{result.mean_a:>9.4f}{result.mean_b:>9.4f}"
            f"{result.mean_diff:>+9.4f}  p={result.t_p:<8.3g}{star}"
        )

    config = {
        "algorithm": "neural fitted value iteration (post-decision states)",
        "reward": args.reward,
        "rounds": args.rounds,
        "seeds_trained": args.seeds,
        "selected_seed": best_overall["seed"],
        "selection": "best validation coverage across seeds",
        "stations_nodes": [int(s) for s in sites],
        "home_bases": [int(b) for b in allocation],
        "test_coverage_mean": float(cov.mean()),
        "test_coverage_std": float(cov.std(ddof=1)),
    }
    save_agent(best_overall["net"], args.out, config)
    print(f"\nsaved {args.out}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")

    summary_path = args.csv.with_name("seed_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "config": config,
                "baseline_coverage": baseline["coverage"],
                "test_coverage_mean": float(cov.mean()),
                "test_coverage_std": float(cov.std(ddof=1)),
                "shifted_coverage_mean": float(shifted_cov.mean()),
                "seeds_beating_baseline": int((cov > baseline["coverage"]).sum()),
                "per_seed": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
