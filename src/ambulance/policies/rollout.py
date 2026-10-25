"""One-step lookahead with Monte Carlo rollout.

At each repositioning decision the agent simulates, for each candidate station, a handful
of sampled futures in which it goes there and the fleet is then run by a cheap base policy
(DMEXCLP). It picks the station with the best average simulated outcome.

This is the classic rollout algorithm: approximate the Q-value of each action by
`immediate action + value of the base policy thereafter`. Its appeal is the policy
improvement property -- with exact expectations, rollout is no worse than the base policy
it rolls out -- and its cost is that the expectations are estimated from a few noisy
samples, so with a small budget it can land below its own base policy. Measuring where
that crossover sits is the point of the Phase 3 budget study.

Two things keep the estimates honest:

- The forked futures *resample* call arrivals rather than replaying the real ones, so the
  policy never sees the future it is being graded on.
- Every candidate station is scored against the same set of sampled futures (common random
  numbers), so the comparison is paired exactly as in the outer experiment. Without this,
  the noise between candidates swamps the differences being measured.
"""

from __future__ import annotations

import numpy as np

from ..city import Call
from ..location import TARGET_MINUTES
from .base import Policy
from .coverage import DMEXCLP


class RolloutPolicy(Policy):
    """Nearest-idle dispatch; repositioning chosen by rollout over a base policy.

    Dispatch is left myopic deliberately. For an urgent call the nearest free unit is
    almost always right, and spending the simulation budget there would buy little; the
    combinatorial choice worth searching is where the *idle* units wait.
    """

    def __init__(
        self,
        n_sims: int = 4,
        horizon: float = 120.0,
        n_candidates: int = 3,
        value_mode: str = "coverage",
        base: Policy | None = None,
        busy_prob: float = 0.35,
        target: float = TARGET_MINUTES,
        seed: int = 0,
    ) -> None:
        self.n_sims = n_sims
        self.horizon = horizon
        self.n_candidates = n_candidates
        if value_mode not in ("coverage", "response"):
            raise ValueError(f"unknown value_mode {value_mode!r}")
        self.value_mode = value_mode
        self.base = base if base is not None else DMEXCLP(busy_prob=busy_prob, target=target)
        self.ranker = DMEXCLP(busy_prob=busy_prob, target=target)
        self.target = target
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sims_run = 0  # for the cost side of the budget study

    @property
    def name(self) -> str:
        return (
            f"rollout({self.value_mode[:3]},k={self.n_candidates},"
            f"m={self.n_sims},H={self.horizon:g})"
        )

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.sims_run = 0

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        shortlist = self._shortlist(sim, amb_id)
        if len(shortlist) == 1:
            return shortlist[0]

        # Common random numbers: one seed per sampled future, reused for every candidate.
        seeds = self.rng.integers(0, 2**63 - 1, size=self.n_sims)

        best_base, best_value = shortlist[0], -np.inf
        for station in shortlist:
            total = 0.0
            for seed in seeds:
                total += self._simulate(sim, amb_id, station, int(seed))
            value = total / self.n_sims
            # Strict improvement only, so ties fall to the earlier (better-ranked)
            # candidate and the policy degrades gracefully to its base policy.
            if value > best_value + 1e-12:
                best_base, best_value = station, value
        return best_base

    def _shortlist(self, sim, amb_id: int) -> list[int]:
        """The `n_candidates` stations the base policy rates highest.

        Scoring every station would spend most of the budget on options the heuristic
        already knows are poor. The top-ranked one comes first, so a rollout that learns
        nothing returns exactly the base policy's choice.
        """
        gains = self.ranker.gains(sim, amb_id)
        me = sim.ambulances[amb_id]
        ranked = sorted(
            gains, key=lambda b: (-gains[b], sim.travel_time_to(me, b))
        )
        return ranked[: max(1, self.n_candidates)]

    def _simulate(self, sim, amb_id: int, station: int, seed: int) -> float:
        """Value of sending `amb_id` to `station` in one sampled future."""
        rng = np.random.default_rng(seed)
        fork = sim.fork(rng, self.horizon)
        fork.send_to_base(fork.ambulances[amb_id], station)
        fork.resume(self.base, until=sim.time + self.horizon)
        self.sims_run += 1
        return self._value(fork)

    # A call left unreached at the horizon is scored at this many minutes, so that
    # "never got there" is worse than any response time actually observed.
    CENSORED_PENALTY = 40.0

    def _value(self, fork) -> float:
        """Simulated value of the sampled future, higher is better.

        Only calls sampled inside the fork count. Missions already in flight at the
        decision epoch are committed and identical across candidates, so including them
        would add variance without adding signal.

        `coverage` scores the objective directly but is a mean of indicators over a
        handful of calls, so it is extremely coarse: at a two-hour horizon it moves in
        steps of about 1/7 while the difference between two good stations is ~0.002.
        `response` scores negative mean response time instead -- a biased proxy for the
        objective, but a continuous one, which is the better trade at feasible budgets.
        """
        if not fork.sampled_ids:
            return 0.0  # nothing sampled: every candidate ties, shortlist order decides

        if self.value_mode == "coverage":
            hits = sum(
                1
                for cid in fork.sampled_ids
                if (rec := fork.records[cid]).reached_at is not None
                and rec.response_time <= self.target
            )
            return hits / len(fork.sampled_ids)

        total = 0.0
        for cid in fork.sampled_ids:
            rec = fork.records[cid]
            rt = rec.response_time
            total += self.CENSORED_PENALTY if rt is None else min(rt, self.CENSORED_PENALTY)
        return -total / len(fork.sampled_ids)
