"""Draw the environment and what the learned policy does in it.

    python scripts/plot_city.py --agent models/nfq_agent.pt

Left: the city the agent acts in -- demand intensity per node, the stations it can park
at, and the hospitals units must reach after a transport. Right: how often each policy
parks a freed unit at each station, over a simulated week.

The right panel is the point. A learning curve says whether the agent got better; this
says what it actually learned to do, and whether that differs from the heuristic it
matches on aggregate metrics. Two policies can score identically and behave quite
differently, and that is worth seeing before claiming they are equivalent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ambulance import Simulator, generate_calls, grid_city, mexclp
from ambulance.policies import DMEXCLP


def record_choices(city, policy, allocation, n_ambulances, days, seed=1000):
    """Count where a policy sends freed units over one simulated stretch."""
    counts: dict[int, int] = {b: 0 for b in city.bases}

    class Recorder(type(policy)):
        pass

    inner = policy
    original = inner.reposition

    def wrapped(sim, amb_id):
        choice = original(sim, amb_id)
        counts[choice] = counts.get(choice, 0) + 1
        return choice

    inner.reposition = wrapped
    try:
        calls = generate_calls(city, days * 24 * 60, np.random.default_rng(seed))
        sim = Simulator(
            city, calls, n_ambulances=n_ambulances, seed=seed, home_bases=allocation
        )
        sim.run(inner)
    finally:
        inner.reposition = original
    total = max(sum(counts.values()), 1)
    return {b: counts.get(b, 0) / total for b in city.bases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=Path, default=Path("models/nfq_agent.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/city_and_policy.png"))
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--ambulances", type=int, default=6)
    ap.add_argument("--call-rate", type=float, default=4.0)
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    city = grid_city(
        width=args.grid, height=args.grid, n_bases=args.stations,
        base_call_rate=args.call_rate, seed=7,
    )
    sites, allocation = mexclp(city, args.ambulances, busy_prob=0.35,
                              max_sites=args.stations)
    run_city = city.with_bases(sites)

    fig, (ax_map, ax_pol) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- the environment ---------------------------------------------------------
    nodes = run_city.nodes
    pos = np.array([run_city.graph.nodes[n]["pos"] for n in nodes], dtype=float)
    demand = run_city.demand_weights
    grid = np.zeros((args.grid, args.grid))
    for (x, y), d in zip(pos, demand):
        grid[int(y), int(x)] = d
    im = ax_map.imshow(grid, origin="lower", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax_map, label="share of demand", fraction=0.046)

    for u, v in run_city.graph.edges:
        (x1, y1) = run_city.graph.nodes[u]["pos"]
        (x2, y2) = run_city.graph.nodes[v]["pos"]
        ax_map.plot([x1, x2], [y1, y2], color="#888888", lw=0.4, alpha=0.5, zorder=1)

    station_pos = np.array([run_city.graph.nodes[s]["pos"] for s in run_city.bases])
    ax_map.scatter(
        station_pos[:, 0], station_pos[:, 1], marker="s", s=150,
        facecolor="tab:blue", edgecolor="white", linewidth=1.5, zorder=3,
        label=f"stations ({len(run_city.bases)})",
    )
    hosp_pos = np.array([run_city.graph.nodes[h]["pos"] for h in run_city.hospitals])
    ax_map.scatter(
        hosp_pos[:, 0], hosp_pos[:, 1], marker="P", s=170,
        facecolor="tab:green", edgecolor="white", linewidth=1.5, zorder=3,
        label="hospitals",
    )
    for station, (x, y) in zip(run_city.bases, station_pos):
        ax_map.annotate(
            str(station), (x, y), textcoords="offset points", xytext=(8, 6),
            fontsize=8, color="tab:blue", weight="bold",
        )
    ax_map.set_title(
        f"The environment: {args.grid}x{args.grid} road graph,\n"
        f"demand hotspots, {len(run_city.bases)} stations", fontsize=11
    )
    # Below the axes: the top-left corner is a demand hotspot in this city, and a
    # legend box there hides the thing the panel exists to show.
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, fontsize=8, frameon=False)
    ax_map.set_xticks([]); ax_map.set_yticks([])

    # --- what each policy does ---------------------------------------------------
    kw = dict(allocation=allocation, n_ambulances=args.ambulances, days=args.days)
    series = {
        "DMEXCLP heuristic": record_choices(run_city, DMEXCLP(busy_prob=0.35), **kw)
    }
    if args.agent.exists():
        from ambulance.deeprl import DeepValuePolicy, load_agent
        from ambulance.features import FeatureExtractor

        net, _ = load_agent(args.agent)
        series["RL agent"] = record_choices(
            run_city, DeepValuePolicy(net, FeatureExtractor(run_city, busy_prob=0.35)),
            **kw,
        )
    else:
        print(f"  (no agent at {args.agent}; plotting the heuristic only)")

    labels = [str(b) for b in run_city.bases]
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for i, (name, counts) in enumerate(series.items()):
        ax_pol.bar(
            x + i * width - 0.4 + width / 2,
            [counts[b] for b in run_city.bases],
            width, label=name,
        )
    ax_pol.set_xticks(x)
    ax_pol.set_xticklabels(labels)
    ax_pol.set_xlabel("station node")
    ax_pol.set_ylabel("share of repositioning decisions")
    ax_pol.set_title(
        "What the policies learned to do:\nwhere a freed unit is sent", fontsize=11
    )
    ax_pol.legend(fontsize=9, frameon=False)
    ax_pol.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    if len(series) == 2:
        a, b = (list(series.values())[0], list(series.values())[1])
        drift = sum(abs(a[s] - b[s]) for s in run_city.bases) / 2
        print(
            f"total variation between the two policies' station usage: {drift:.1%}"
        )


if __name__ == "__main__":
    main()
