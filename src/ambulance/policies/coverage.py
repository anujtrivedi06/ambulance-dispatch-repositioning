"""Coverage-aware dynamic repositioning (DMEXCLP).

Baseline B1 sends a freed unit back to its home station, which ignores where the rest of
the fleet currently is. DMEXCLP instead sends it wherever it adds the most *expected*
coverage given the units that are still available -- so if two units are already busy in
the east, the freed unit drifts east to fill the hole rather than parking at home.

Jagtenberg, Bhulai & van der Mei (2015), "An efficient heuristic for real-time ambulance
redeployment", use exactly this marginal-MEXCLP rule and report it beating static
redeployment on real Dutch EMS data.
"""

from __future__ import annotations

import numpy as np

from ..city import Call
from ..location import TARGET_MINUTES, coverage_matrix
from .base import Policy


class DMEXCLP(Policy):
    """Nearest-idle dispatch, marginal-expected-coverage repositioning."""

    name = "dmexclp"

    def __init__(
        self,
        busy_prob: float = 0.35,
        target: float = TARGET_MINUTES,
        candidates: list[int] | None = None,
    ) -> None:
        self.busy_prob = busy_prob
        self.target = target
        self.candidates = candidates
        self._cov: np.ndarray | None = None
        self._city_id: int | None = None

    def _coverage(self, city) -> np.ndarray:
        """Cache the coverage matrix per city; it is fixed for a whole experiment."""
        if self._cov is None or self._city_id != id(city):
            self._cov = coverage_matrix(city, self.target)
            self._city_id = id(city)
        return self._cov

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def gains(self, sim, amb_id: int) -> dict[int, float]:
        """Marginal expected covered demand from sending unit `amb_id` to each station.

        Exposed separately so a lookahead policy can use this ranking to shortlist
        candidate stations instead of simulating futures for every one of them.
        """
        city = sim.city
        cov = self._coverage(city)
        q = self.busy_prob

        # How many *other* available units already cover each demand node. A relocating
        # unit counts at its destination, since that is where it will end up waiting.
        counts = np.zeros(len(city.nodes), dtype=int)
        for amb in sim.available():
            if amb.id != amb_id:
                counts += cov[city.index_of(amb.destination)]

        # Marginal expected demand newly covered by putting one more unit at a site:
        # for each node it reaches, the chance every unit already covering it is busy.
        gain_scale = city.demand_weights * (q**counts) * (1.0 - q)
        bases = self.candidates if self.candidates is not None else city.bases
        return {b: float(gain_scale @ cov[city.index_of(b)]) for b in bases}

    def reposition(self, sim, amb_id: int) -> int:
        gains = self.gains(sim, amb_id)
        me = sim.ambulances[amb_id]
        bases = list(gains)
        best_base, best_gain, best_tt = bases[0], -np.inf, np.inf
        for base in bases:
            gain = gains[base]
            tt = sim.travel_time_to(me, base)
            # Prefer the closer station when coverage gain ties, to avoid pointless
            # driving (and the fuel/crew cost a real service would object to).
            if gain > best_gain + 1e-12 or (abs(gain - best_gain) <= 1e-12 and tt < best_tt):
                best_base, best_gain, best_tt = base, gain, tt
        return best_base
