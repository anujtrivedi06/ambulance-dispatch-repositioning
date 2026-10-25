"""Approximate dynamic programming: learning a value function over post-decision states.

The agent collects transitions by running the simulator under an exploring behaviour
policy, then fits a linear value function by least-squares temporal differences. The
learned policy then ranks candidate stations by that function alone -- one dot product per
station, no simulation at decision time. That is the whole point: Phase 3's estimate cost
hundreds of simulated futures per decision and still could not separate the candidates;
this one costs twelve multiplications and is fitted on tens of thousands of transitions.

Decision epochs are irregularly spaced, so discounting is continuous-time: a gap of `dt`
minutes discounts by `exp(-beta * dt)`. Using a fixed per-decision discount would make a
decision followed by a quiet night worth the same as one followed by a busy minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .city import City
from .experiment import CALL_SEED_BASE, SIM_SEED_BASE
from .city import generate_calls
from .features import FeatureExtractor
from .policies.base import Policy
from .policies.coverage import DMEXCLP
from .sim import Simulator

# 1/240 per minute: a reward four hours out is worth about a third of one now, which is
# roughly the span over which a repositioning choice still influences anything.
DEFAULT_BETA = 1.0 / 240.0
CENSORED_PENALTY = 40.0


@dataclass
class Transition:
    phi: np.ndarray
    reward: float
    discount: float
    next_phi: np.ndarray


class CollectingPolicy(Policy):
    """Behaviour policy for data collection: a base policy with epsilon-greedy noise.

    Exploration is necessary and slightly awkward. Without it every transition sits on
    the states DMEXCLP visits, and the value function learns nothing about the stations it
    never picks -- so the learned policy cannot discover that one of them is better.
    """

    name = "collecting"

    def __init__(
        self,
        extractor: FeatureExtractor,
        base: Policy | None = None,
        epsilon: float = 0.3,
        seed: int = 0,
    ) -> None:
        self.extractor = extractor
        self.base = base if base is not None else DMEXCLP()
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.decisions: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.decisions = []

    def dispatch(self, sim, call):
        return self.base.dispatch(sim, call)

    def reposition(self, sim, amb_id: int) -> int:
        stations = sim.city.bases
        if self.rng.random() < self.epsilon:
            action = int(self.rng.choice(stations))
        else:
            action = self.base.reposition(sim, amb_id)
        self.decisions.append((sim.time, self.extractor.post_decision(sim, amb_id, action)))
        return action


def rewards_between(
    records, times: np.ndarray, reward_mode: str, target: float
) -> np.ndarray:
    """Reward accrued in each interval between consecutive decision epochs.

    A call contributes at the moment it is *reached*, which is when the outcome is
    determined, not when it arrived.
    """
    # Validated before touching the data: otherwise a misspelled mode is only caught on
    # episodes that happen to have reached a call, and silently returns zeros on the rest.
    if reward_mode not in ("coverage", "response"):
        raise ValueError(f"unknown reward_mode {reward_mode!r}")

    reached, values = [], []
    for rec in records:
        if rec.reached_at is None:
            continue
        reached.append(rec.reached_at)
        if reward_mode == "coverage":
            values.append(1.0 if rec.response_time <= target else 0.0)
        else:
            values.append(-min(rec.response_time, CENSORED_PENALTY) / 10.0)

    out = np.zeros(max(len(times) - 1, 0))
    if not reached:
        return out
    order = np.argsort(reached)
    reached = np.asarray(reached)[order]
    values = np.asarray(values)[order]
    bucket = np.searchsorted(times, reached, side="right") - 1
    for b, v in zip(bucket, values):
        if 0 <= b < len(out):
            out[b] += v
    return out


def collect(
    city: City,
    policy: CollectingPolicy,
    episodes: int,
    days: int = 3,
    n_ambulances: int = 6,
    home_bases: list[int] | None = None,
    reward_mode: str = "coverage",
    target: float = 8.0,
    beta: float = DEFAULT_BETA,
    seed_offset: int = 0,
) -> list[Transition]:
    """Run episodes under `policy` and turn each into TD transitions."""
    transitions: list[Transition] = []
    for ep in range(episodes):
        calls = generate_calls(
            city,
            days * 24 * 60,
            np.random.default_rng(CALL_SEED_BASE + 10_000 + seed_offset + ep),
        )
        policy.reset()
        sim = Simulator(
            city,
            calls,
            n_ambulances=n_ambulances,
            seed=SIM_SEED_BASE + 10_000 + seed_offset + ep,
            home_bases=home_bases,
        )
        records = sim.run(policy)
        if len(policy.decisions) < 2:
            continue

        times = np.array([t for t, _ in policy.decisions])
        phis = [p for _, p in policy.decisions]
        rewards = rewards_between(records, times, reward_mode, target)
        gaps = np.diff(times)
        # The final decision has no successor to bootstrap from, so it is dropped rather
        # than treated as terminal: the episode boundary is an artefact of the horizon,
        # not a real absorbing state.
        for k in range(len(rewards)):
            transitions.append(
                Transition(
                    phi=phis[k],
                    reward=float(rewards[k]),
                    discount=float(np.exp(-beta * gaps[k])),
                    next_phi=phis[k + 1],
                )
            )
    return transitions


def fit_lstd(transitions: list[Transition], ridge: float = 1e-3) -> np.ndarray:
    """Least-squares TD(0) over post-decision states.

    Solves `A theta = b` with `A = sum phi (phi - gamma phi')^T`, `b = sum phi r`. The
    ridge term is not optional: with correlated features and a few thousand samples, A is
    badly conditioned and the unregularised solution swings wildly between runs.
    """
    if not transitions:
        raise ValueError("no transitions to fit")
    d = len(transitions[0].phi)
    A = np.zeros((d, d))
    b = np.zeros(d)
    for t in transitions:
        A += np.outer(t.phi, t.phi - t.discount * t.next_phi)
        b += t.phi * t.reward
    # Scale the ridge per feature. A flat ridge would penalise `demand_weighted_distance`
    # (spread ~0.5) far harder than `bias` (constant 1), which is an arbitrary preference
    # rather than a modelling choice.
    scale = np.array([np.var([t.phi[j] for t in transitions]) for j in range(d)])
    scale = np.where(scale > 1e-12, scale, 1.0)
    A += ridge * np.diag(scale)
    return np.linalg.solve(A, b)


def train_lspi(
    city: City,
    extractor: FeatureExtractor,
    iterations: int = 4,
    episodes: int = 40,
    epsilon: float = 0.3,
    reward_mode: str = "coverage",
    ridge: float = 1e-3,
    seed: int = 0,
    **collect_kw,
) -> list[np.ndarray]:
    """Least-squares policy iteration: collect, fit, act greedily, repeat.

    Each round explores around the *current* learned policy rather than around DMEXCLP,
    which is the standard answer when a value function fitted off one policy ranks
    actions badly for another. Returns the weights after each round so the trajectory
    can be reported rather than just the endpoint.
    """
    from .policies.adp import ADPPolicy

    base: Policy = DMEXCLP()
    history: list[np.ndarray] = []
    for it in range(iterations):
        collector = CollectingPolicy(extractor, base=base, epsilon=epsilon, seed=seed + it)
        transitions = collect(
            city,
            collector,
            episodes=episodes,
            reward_mode=reward_mode,
            seed_offset=1000 * it,
            **collect_kw,
        )
        theta = fit_lstd(transitions, ridge=ridge)
        history.append(theta)
        base = ADPPolicy(theta, extractor)
    return history


def bellman_residual(transitions: list[Transition], theta: np.ndarray) -> float:
    """Root-mean-square TD error, for reporting how well the fit actually holds."""
    errs = [
        t.reward + t.discount * (theta @ t.next_phi) - (theta @ t.phi)
        for t in transitions
    ]
    return float(np.sqrt(np.mean(np.square(errs))))
