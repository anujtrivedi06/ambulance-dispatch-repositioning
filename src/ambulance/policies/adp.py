"""Greedy policy over a learned post-decision value function."""

from __future__ import annotations

import numpy as np

from ..city import Call
from ..features import FeatureExtractor
from .base import Policy


class ADPPolicy(Policy):
    """Nearest-idle dispatch; repositioning by a linear value function.

    The decision is one dot product per candidate station. Compare that with Phase 3,
    where the same choice cost hundreds of simulated futures and still could not separate
    the candidates: the sampling has been moved offline into the fitted weights.
    """

    def __init__(
        self,
        theta: np.ndarray,
        extractor: FeatureExtractor,
        label: str = "adp",
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.extractor = extractor
        self.label = label
        self.sims_run = 0  # keeps the Phase 3 cost table shape; always zero here

    @property
    def name(self) -> str:
        return self.label

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        me = sim.ambulances[amb_id]
        best_station, best_value, best_tt = sim.city.bases[0], -np.inf, np.inf
        for station in sim.city.bases:
            value = float(self.theta @ self.extractor.post_decision(sim, amb_id, station))
            tt = sim.travel_time_to(me, station)
            # Ties go to the nearer station, as in DMEXCLP, so the learned policy does
            # not pay for pointless driving when the value function is indifferent.
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12 and tt < best_tt
            ):
                best_station, best_value, best_tt = station, value, tt
        return best_station
