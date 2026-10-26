"""Train, evaluate and save the RL agent.

This is the entry point for the deliverable agent: it trains the neural fitted value
iteration policy, records a learning curve on held-out episodes, tests whether the learned
policy survives a demand shift it never saw, and writes the trained weights to disk.

    python scripts/train_agent.py --rounds 8 --out models/nfq_agent.pt

The learning curve is evaluated on episodes drawn from seeds disjoint from training, after
every round. Training loss is reported alongside it deliberately: the two come apart on
this problem, and watching them come apart is the clearest evidence for the project's main
finding -- that fitting the value function better does not make the policy better.
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
from ambulance.experiment import aggregate, evaluate
from ambulance.features import FeatureExtractor
from ambulance.policies import DMEXCLP

EVAL_METRICS = ["coverage", "mean_response", "p90_response", "worst_zone_coverage"]

# Disjoint from the test episodes (offset 0) and the training episodes (10_000+).
VALIDATION_OFFSET = 50_000


def main() -> None:
    p = argparse.ArgumentParser()
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
    p.add_argument("--final-reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("models/nfq_agent.pt"))
    p.add_argument("--curve-csv", type=Path, default=Path("results/learning_curve.csv"))
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
    train_kw = dict(home_bases=allocation, n_ambulances=args.ambulances, days=3)

    print(
        f"environment: {args.grid}x{args.grid} grid, {args.stations} stations, "
        f"{args.ambulances} ambulances, {args.call_rate} calls/h\n"
        f"agent: neural fitted value iteration, reward={args.reward}, "
        f"epsilon={args.epsilon}, {args.rounds} rounds\n"
    )

    # Model selection runs on a validation block of episodes disjoint from both the
    # training seeds and the test seeds. Choosing a checkpoint on the episodes it is
    # then scored on would make the reported number a best-of-N rather than an estimate.
    best = {"coverage": -1.0, "round": 0, "state": None}

    def curve_eval(round_index: int, net) -> dict[str, float]:
        policy = DeepValuePolicy(net, extractor)
        scores = aggregate(
            evaluate(
                policy, run_city, days=3, reps=args.curve_reps,
                n_ambulances=args.ambulances, home_bases=allocation,
                seed_offset=VALIDATION_OFFSET,
            )
        )
        if scores["coverage"] > best["coverage"]:
            best.update(
                coverage=scores["coverage"],
                round=round_index,
                state={k: v.clone() for k, v in net.state_dict().items()},
            )
        print(
            f"  round {round_index:2d}  val coverage {scores['coverage']:.4f}"
            f"  mean {scores['mean_response']:.2f}"
        )
        return {f"eval_{m}": scores[m] for m in EVAL_METRICS}

    net, history = train_nfq(
        run_city, extractor,
        rounds=args.rounds, episodes=args.episodes, epochs=args.epochs,
        reward_mode=args.reward, epsilon=args.epsilon, fresh_buffer=True,
        seed=args.seed, eval_fn=curve_eval, **train_kw,
    )

    if best["state"] is not None:
        net.load_state_dict(best["state"])
        print(
            f"\nselected round {best['round']} by validation coverage "
            f"({best['coverage']:.4f}); later rounds were not better"
        )

    config = {
        "algorithm": "neural fitted value iteration (post-decision states)",
        "reward": args.reward,
        "rounds": args.rounds,
        "selected_round": best["round"],
        "episodes_per_round": args.episodes,
        "epsilon": args.epsilon,
        "grid": args.grid,
        "stations": args.stations,
        "ambulances": args.ambulances,
        "call_rate": args.call_rate,
        "busy_prob": args.busy_prob,
        "stations_nodes": [int(s) for s in sites],
        "home_bases": [int(b) for b in allocation],
    }
    save_agent(net, args.out, config)
    print(f"\nsaved trained agent to {args.out}")

    args.curve_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.curve_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"wrote {args.curve_csv}")

    # --- final comparison on the training demand profile -------------------------
    print("\nfinal evaluation (30 paired reps x 7 days, held-out seeds)")
    agent = DeepValuePolicy(net, extractor, label="rl agent (NFQ)")
    baseline = DMEXCLP(busy_prob=args.busy_prob)
    results = {}
    for policy in (baseline, agent):
        results[policy.name] = aggregate(
            evaluate(
                policy, run_city, days=7, reps=args.final_reps,
                n_ambulances=args.ambulances, home_bases=allocation,
            )
        )
        print(
            f"  {policy.name:18s} coverage {results[policy.name]['coverage']:.4f}"
            f"  mean {results[policy.name]['mean_response']:.2f}"
        )

    # --- generalisation: a demand profile the agent never trained on -------------
    # Shifting the hotspots moves demand somewhere the learned value function has never
    # been asked about. A policy that only memorised this city collapses here; one that
    # learned something about coverage should degrade about as much as the heuristic.
    print("\ngeneralisation to a shifted demand profile (unseen during training)")
    shifted = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=21,
    ).with_bases(sites)
    shifted_extractor = FeatureExtractor(shifted, busy_prob=args.busy_prob)
    shifted_results = {}
    for policy in (
        DMEXCLP(busy_prob=args.busy_prob),
        DeepValuePolicy(net, shifted_extractor, label="rl agent (NFQ)"),
    ):
        shifted_results[policy.name] = aggregate(
            evaluate(
                policy, shifted, days=7, reps=args.final_reps,
                n_ambulances=args.ambulances, home_bases=allocation,
            )
        )
        drop = (
            results[policy.name]["coverage"]
            - shifted_results[policy.name]["coverage"]
        )
        print(
            f"  {policy.name:18s} coverage {shifted_results[policy.name]['coverage']:.4f}"
            f"   (drop of {drop:+.4f} from the trained profile)"
        )

    summary = {
        "config": config,
        "trained_profile": results,
        "shifted_profile": shifted_results,
    }
    out_json = args.curve_csv.with_name("agent_summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
