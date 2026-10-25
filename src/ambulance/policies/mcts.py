"""Monte Carlo tree search over sequences of repositioning decisions.

Rollout evaluates each candidate station once and stops thinking. MCTS spends its budget
unevenly instead: it keeps a tree over the next few repositioning decisions, descends it
by UCT so promising branches get sampled more, and falls back on the base policy past the
tree's depth.

This is *open-loop* MCTS. A node is identified by the sequence of actions that reached it,
not by a simulator state, because the states differ between iterations -- each iteration
samples its own future, so the fleet state at depth 2 is never twice the same. Open-loop
search is the standard answer for stochastic domains where chance nodes would otherwise
branch without bound, and it is what makes a tree affordable here at all. The cost is
that a node averages over the states its action sequence can lead to, which blurs the
very distinctions the search is trying to make.

Values are normalised to [0, 1] from the range observed at the root before UCT sees them,
so the exploration constant means the same thing whichever value mode is in use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..city import Call
from ..location import TARGET_MINUTES
from .base import Policy
from .coverage import DMEXCLP
from .rollout import RolloutPolicy


class _Pause(Exception):
    """Raised inside a forked run to hand a repositioning decision back to the search."""

    def __init__(self, amb_id: int) -> None:
        super().__init__(amb_id)
        self.amb_id = amb_id


class _PausingPolicy(Policy):
    """Dispatches like the base policy, but stops the run at every repositioning choice.

    Unwinding by exception is safe here: the simulator sets the unit idle and clears its
    route *before* asking for a station, so the state it leaves behind is exactly the
    decision epoch the search wants to act on.
    """

    name = "pausing"

    def __init__(self, base: Policy) -> None:
        self.base = base

    def dispatch(self, sim, call: Call) -> int | None:
        return self.base.dispatch(sim, call)

    def reposition(self, sim, amb_id: int) -> int:
        raise _Pause(amb_id)


@dataclass
class _Node:
    """One action sequence. Children are keyed by the station chosen next."""

    amb_id: int
    actions: list[int]
    visits: int = 0
    children: dict[int, "_Node"] = field(default_factory=dict)
    action_visits: dict[int, int] = field(default_factory=dict)
    action_value: dict[int, float] = field(default_factory=dict)

    def mean(self, action: int) -> float:
        n = self.action_visits.get(action, 0)
        return self.action_value.get(action, 0.0) / n if n else 0.0


class MCTSPolicy(RolloutPolicy):
    """Nearest-idle dispatch; repositioning chosen by open-loop MCTS.

    Inherits the dispatch rule, the candidate shortlist and the value function from
    `RolloutPolicy`, so the two differ only in how the simulation budget is spent --
    which is the comparison Phase 3 is trying to make.
    """

    def __init__(
        self,
        n_sims: int = 16,
        horizon: float = 240.0,
        depth: int = 2,
        n_candidates: int = 2,
        value_mode: str = "coverage",
        exploration: float = 0.7,
        base: Policy | None = None,
        busy_prob: float = 0.35,
        target: float = TARGET_MINUTES,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n_sims=n_sims,
            horizon=horizon,
            n_candidates=n_candidates,
            value_mode=value_mode,
            base=base,
            busy_prob=busy_prob,
            target=target,
            seed=seed,
        )
        self.depth = depth
        self.exploration = exploration

    @property
    def name(self) -> str:
        return (
            f"mcts({self.value_mode[:3]},k={self.n_candidates},"
            f"n={self.n_sims},d={self.depth},H={self.horizon:g})"
        )

    def reposition(self, sim, amb_id: int) -> int:
        shortlist = self._shortlist(sim, amb_id)
        if len(shortlist) == 1:
            return shortlist[0]

        root = _Node(amb_id=amb_id, actions=[])
        lo, hi = math.inf, -math.inf
        deadline = sim.time + self.horizon

        for _ in range(self.n_sims):
            rng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))
            fork = sim.fork(rng, self.horizon)
            value = self._descend(fork, root, shortlist, amb_id, deadline, 0, lo, hi)
            lo, hi = min(lo, value), max(hi, value)

        # Pick the best observed mean, not the most-visited child: at these budgets UCT
        # has not had the visits for visit counts to be the more reliable signal.
        visited = [a for a in shortlist if root.action_visits.get(a, 0) > 0]
        if not visited:
            return shortlist[0]
        best = max(visited, key=lambda a: (root.mean(a), -shortlist.index(a)))
        return best

    def _descend(
        self,
        fork,
        node: _Node,
        shortlist: list[int],
        amb_id: int,
        deadline: float,
        depth: int,
        lo: float,
        hi: float,
    ) -> float:
        action = self._select(node, shortlist, lo, hi)
        fork.send_to_base(fork.ambulances[amb_id], action)

        next_amb = None
        if depth + 1 < self.depth:
            next_amb = self._advance_to_decision(fork, deadline)

        if next_amb is None:
            # Past the tree: let the base policy finish the sampled future.
            fork.resume(self.base, until=deadline)
            self.sims_run += 1
            value = self._value(fork)
        else:
            child = node.children.get(action)
            if child is None or child.amb_id != next_amb:
                child = _Node(amb_id=next_amb, actions=node.actions + [action])
                node.children[action] = child
            child_shortlist = self._shortlist(fork, next_amb)
            value = self._descend(
                fork, child, child_shortlist, next_amb, deadline, depth + 1, lo, hi
            )

        node.visits += 1
        node.action_visits[action] = node.action_visits.get(action, 0) + 1
        node.action_value[action] = node.action_value.get(action, 0.0) + value
        return value

    def _advance_to_decision(self, fork, deadline: float) -> int | None:
        """Run the fork to the next repositioning decision, or None if none is left."""
        try:
            fork.resume(_PausingPolicy(self.base), until=deadline)
        except _Pause as pause:
            return pause.amb_id
        return None

    def _select(self, node: _Node, shortlist: list[int], lo: float, hi: float) -> int:
        """UCT over the shortlist, untried actions first in base-policy rank order."""
        for action in shortlist:
            if node.action_visits.get(action, 0) == 0:
                return action

        span = hi - lo
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        best, best_score = shortlist[0], -math.inf
        for action in shortlist:
            q = node.mean(action)
            q_norm = (q - lo) / span if span > 1e-12 else 0.5
            bonus = self.exploration * math.sqrt(log_n / node.action_visits[action])
            score = q_norm + bonus
            if score > best_score:
                best, best_score = action, score
        return best
