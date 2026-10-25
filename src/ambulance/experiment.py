"""Running policies over seeded, paired replications."""

from __future__ import annotations

import numpy as np

from .city import City, generate_calls
from .metrics import summarise
from .sim import Simulator

CALL_SEED_BASE = 1000
SIM_SEED_BASE = 2000


def evaluate(
    policy,
    city: City,
    days: int = 7,
    reps: int = 30,
    n_ambulances: int = 6,
    home_bases: list[int] | None = None,
) -> list[dict[str, float]]:
    """Metrics for each replication.

    Replication `r` always uses call seed `CALL_SEED_BASE + r` and simulator seed
    `SIM_SEED_BASE + r`, so every policy faces an identical call stream and identical
    travel-noise stream in replication `r`. Comparisons are therefore paired: the
    arrival process cancels out instead of drowning the policy effect.
    """
    out = []
    for rep in range(reps):
        calls = generate_calls(
            city, days * 24 * 60, np.random.default_rng(CALL_SEED_BASE + rep)
        )
        policy.reset()
        sim = Simulator(
            city,
            calls,
            n_ambulances=n_ambulances,
            seed=SIM_SEED_BASE + rep,
            home_bases=home_bases,
        )
        out.append(summarise(sim.run(policy), city))
    return out


def aggregate(per_rep: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across replications."""
    return {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}


def column(per_rep: list[dict[str, float]], metric: str) -> np.ndarray:
    return np.array([m[metric] for m in per_rep])
