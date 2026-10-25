"""Direct policy search over the same linear class the ADP value function uses.

ADP fits weights to predict value and then acts greedily on the prediction. This module
skips the prediction step and optimises the weights against simulated coverage itself,
with the cross-entropy method.

The two share a hypothesis class exactly, so comparing them isolates one question: does
fitting a value function recover a good policy, or does the gap between *predicting value
well* and *ranking actions well* cost you? Only the action-varying features are searched,
since the rest cannot change an argmax and would only add dimensions to the search.

CEM is chosen over a policy gradient because the search space is three-dimensional and
every evaluation is a noisy simulation: a derivative-free method that only needs a ranking
of candidates is the better fit, and there is no gradient to estimate cheaply anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .city import City, generate_calls
from .features import FEATURE_NAMES, FeatureExtractor
from .metrics import summarise
from .sim import Simulator

# Only these three features differ between candidate stations; the others are identical
# across every action at a given decision epoch and cancel in the argmax.
ACTION_FEATURES = (
    "expected_covered_demand",
    "worst_zone_coverage",
    "demand_weighted_distance",
)


@dataclass
class SearchResult:
    theta: np.ndarray
    score: float
    history: list[tuple[float, float]] = field(default_factory=list)  # (mean, best)


def expand(weights: np.ndarray) -> np.ndarray:
    """Place searched weights into a full-length theta, zeroing the rest."""
    theta = np.zeros(len(FEATURE_NAMES))
    for w, name in zip(weights, ACTION_FEATURES):
        theta[FEATURE_NAMES.index(name)] = w
    return theta


def evaluate_theta(
    theta: np.ndarray,
    city: City,
    extractor: FeatureExtractor,
    episodes: int = 3,
    days: int = 3,
    n_ambulances: int = 6,
    home_bases: list[int] | None = None,
    seed_offset: int = 0,
    metric: str = "coverage",
) -> float:
    """Mean metric over a few seeded episodes. Higher is better for every metric used."""
    from .policies.adp import ADPPolicy

    policy = ADPPolicy(theta, extractor)
    scores = []
    for ep in range(episodes):
        calls = generate_calls(
            city, days * 24 * 60, np.random.default_rng(5000 + seed_offset + ep)
        )
        sim = Simulator(
            city,
            calls,
            n_ambulances=n_ambulances,
            seed=6000 + seed_offset + ep,
            home_bases=home_bases,
        )
        m = summarise(sim.run(policy), city)
        scores.append(m[metric] if metric == "coverage" else -m[metric])
    return float(np.mean(scores))


def cross_entropy_search(
    city: City,
    extractor: FeatureExtractor,
    iterations: int = 12,
    population: int = 24,
    elite_frac: float = 0.25,
    episodes: int = 3,
    sigma0: float = 1.0,
    seed: int = 0,
    metric: str = "coverage",
    **eval_kw,
) -> SearchResult:
    """Cross-entropy method over the action-varying weights.

    Each generation is scored on the *same* episodes so candidates are compared on one
    set of futures -- the same common-random-numbers argument as in Phase 3, and just as
    necessary: without it CEM chases simulation noise and the elite set is arbitrary.
    The episode seeds change between generations, so the search cannot overfit one week
    of calls.
    """
    rng = np.random.default_rng(seed)
    dim = len(ACTION_FEATURES)
    mean = np.zeros(dim)
    sigma = np.full(dim, sigma0)
    n_elite = max(2, int(population * elite_frac))

    best_theta, best_score = expand(mean), -np.inf
    history: list[tuple[float, float]] = []

    for it in range(iterations):
        candidates = rng.normal(mean, sigma, size=(population, dim))
        scores = np.array(
            [
                evaluate_theta(
                    expand(c),
                    city,
                    extractor,
                    episodes=episodes,
                    seed_offset=100 * it,
                    metric=metric,
                    **eval_kw,
                )
                for c in candidates
            ]
        )
        elite = candidates[np.argsort(scores)[-n_elite:]]
        mean, sigma = elite.mean(axis=0), elite.std(axis=0) + 1e-3

        top = int(np.argmax(scores))
        if scores[top] > best_score:
            best_score, best_theta = float(scores[top]), expand(candidates[top])
        history.append((float(scores.mean()), float(scores.max())))

    return SearchResult(theta=best_theta, score=best_score, history=history)
