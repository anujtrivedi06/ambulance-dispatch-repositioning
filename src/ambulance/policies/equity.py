"""Trading city-wide coverage against the worst-served zone.

DMEXCLP maximises expected covered demand summed over the whole city. That objective has
no term for the worst-off zone, which Phases 2 and 4 both caught it sacrificing: on
hand-placed stations it lifted city-wide coverage while lowering worst-zone coverage, and
a learner given the same aggregate signal abandoned a sparse zone outright.

This policy puts a weight on that trade-off and makes it explicit. With `lam = 0` it is
DMEXCLP exactly, so the frontier it traces starts from the earlier phases' result rather
than from somewhere new.
"""

from __future__ import annotations

import numpy as np

from ..city import Call
from ..location import TARGET_MINUTES, coverage_matrix
from .base import Policy


class EquityDMEXCLP(Policy):
    """Nearest-idle dispatch; repositioning by a mixture of total and worst-zone gain.

    The repositioning score for a candidate station is

        (1 - lam) * marginal expected demand covered city-wide
      +      lam  * marginal expected demand covered in the currently worst zone

    where "currently worst" is recomputed at every decision from the units still
    available, so the target of the equity term moves as the fleet does. Sweeping `lam`
    from 0 to 1 traces the attainable frontier between the two objectives.
    """

    def __init__(
        self,
        lam: float = 0.0,
        busy_prob: float = 0.35,
        target: float = TARGET_MINUTES,
    ) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = lam
        self.busy_prob = busy_prob
        self.target = target
        self._cov: np.ndarray | None = None
        self._city_id: int | None = None
        self._zone_masks: list[np.ndarray] | None = None

    @property
    def name(self) -> str:
        return f"equity(lam={self.lam:g})"

    def _prepare(self, city) -> tuple[np.ndarray, list[np.ndarray]]:
        if self._cov is None or self._city_id != id(city):
            self._cov = coverage_matrix(city, self.target)
            zones = np.array([city.zone_of(n) for n in city.nodes])
            self._zone_masks = [zones == z for z in sorted(set(zones.tolist()))]
            self._city_id = id(city)
        return self._cov, self._zone_masks

    def dispatch(self, sim, call: Call) -> int | None:
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        city = sim.city
        cov, zone_masks = self._prepare(city)
        q = self.busy_prob

        counts = np.zeros(len(city.nodes), dtype=int)
        for amb in sim.available():
            if amb.id != amb_id:
                counts += cov[city.index_of(amb.destination)]

        covered = 1.0 - q**counts
        demand = city.demand_weights
        gain_scale = demand * (q**counts) * (1.0 - q)

        # The zone with the least expected coverage right now is the one the equity term
        # protects. Picked per decision rather than once, so the policy chases whichever
        # zone the current fleet layout has left exposed.
        zone_scores = [
            float(demand[m] @ covered[m]) / max(float(demand[m].sum()), 1e-12)
            for m in zone_masks
        ]
        worst = zone_masks[int(np.argmin(zone_scores))]
        worst_scale = np.where(worst, gain_scale, 0.0)

        me = sim.ambulances[amb_id]
        best_station, best_score, best_tt = city.bases[0], -np.inf, np.inf
        for station in city.bases:
            reach = cov[city.index_of(station)]
            total_gain = float(gain_scale @ reach)
            worst_gain = float(worst_scale @ reach)
            score = (1.0 - self.lam) * total_gain + self.lam * worst_gain
            tt = sim.travel_time_to(me, station)
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and tt < best_tt
            ):
                best_station, best_score, best_tt = station, score, tt
        return best_station
