"""Phase-1 reference policies: the behaviour real services default to, plus a floor."""

from __future__ import annotations

import numpy as np

from ..city import Call
from .base import Policy


class NearestIdleHomeBase(Policy):
    """Send the closest available unit; idle units always return to their home base.

    This is baseline B1: myopically optimal per call, blind to coverage. Every later
    method has to beat it.
    """

    name = "nearest-idle/home-base"

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        return sim.ambulances[amb_id].home_base


class RandomReposition(Policy):
    """Nearest-idle dispatch, but idle units park at a uniformly random station.

    This exists to measure *headroom*: the gap between it and the best known
    repositioning rule is how much the repositioning choice is worth at all in a given
    configuration. If that gap is small, no method can win there, however clever -- which
    is the hypothesis the regime sweep is testing. Holding dispatch fixed is what makes
    the gap attributable to repositioning alone.
    """

    name = "random-reposition"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        return int(self.rng.choice(sim.city.bases))


class RandomPolicy(Policy):
    """Uniformly random choices. A sanity floor -- any real method must beat this."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return int(self.rng.choice([a.id for a in available]))

    def reposition(self, sim, amb_id: int) -> int:
        return int(self.rng.choice(sim.city.bases))
