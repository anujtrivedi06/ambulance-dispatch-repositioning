"""Post-decision state features for approximate dynamic programming.

Phase 3 established that estimating an action's value by simulating futures at decision
time is too noisy to be worth its cost. The alternative is to pay that cost once, offline,
and compress it into a value function. This module provides the state description that
function is defined over.

The features describe a *post-decision* state: the fleet configuration immediately after a
repositioning choice is made, before the next call arrives. That framing matters. The
post-decision state is a deterministic function of (state, action), so a policy can rank
its options by evaluating one cheap feature vector per candidate station -- no sampling,
no simulation, no variance. The randomness is pushed into the value function, which is
where learning can average it out properly.

Features are deliberately few and interpretable. With a linear value function the risk is
not underfitting the simulator, it is fitting noise in a few thousand transitions; and a
small feature set is one whose learned weights can be read and sanity-checked against what
the heuristics already say.
"""

from __future__ import annotations

import numpy as np

from .city import City
from .location import TARGET_MINUTES, coverage_matrix

FEATURE_NAMES = (
    "bias",
    "available_fraction",
    "time_sin",
    "time_cos",
    "expected_covered_demand",
    "worst_zone_coverage",
    "demand_weighted_distance",
)

# This set is what survived two rounds of diagnosis on collected data.
#
# Dropped as exactly redundant (design matrix came back rank 9 of 12):
#   busy_fraction   = bias - available_fraction, exactly
#   queue_length    = 0 always, since a unit is only asked where to wait after the queue
#                     has been served, so the feature never varies at all
#
# Dropped as near-degenerate: the four per-zone coverages. Only three features here can be
# changed by a repositioning action, so they are the only ones that affect the argmax --
# and the per-zone ones were correlated at 0.97 (zones 0,1) and 1.00 (zones 2,3), giving
# the action-varying block a condition number of 246. Their individual weights were not
# identified, only their sum, which is why the fit returned confidently negative weights
# on coverage. Replaced by the total (linear in the zones it summarises) and the worst
# zone (a min, so genuinely independent information about equity).
#
# Ridge regression hides this rather than fixing it: the value *prediction* stays fine,
# because the collinear directions cancel in-distribution. The greedy policy does not,
# because an action moves along exactly the direction the data could not identify.


class FeatureExtractor:
    """Builds post-decision feature vectors, with the city-derived arrays cached."""

    def __init__(
        self,
        city: City,
        busy_prob: float = 0.35,
        target: float = TARGET_MINUTES,
        n_zones: int = 4,
    ) -> None:
        self.city = city
        self.busy_prob = busy_prob
        self.target = target
        self.n_zones = n_zones
        self._cov = coverage_matrix(city, target)
        self._tt = city.travel_matrix()
        self._demand = city.demand_weights
        zones = np.array([city.zone_of(n) for n in city.nodes])
        self._zone_masks = [zones == z for z in range(n_zones)]
        self._zone_demand = np.array(
            [max(self._demand[m].sum(), 1e-12) for m in self._zone_masks]
        )

    @property
    def size(self) -> int:
        return len(FEATURE_NAMES)

    def post_decision(self, sim, amb_id: int, station: int) -> np.ndarray:
        """Features of the state that follows sending `amb_id` to `station`.

        Every other available unit is counted at its own destination, since that is where
        it will be waiting when the next call lands.
        """
        destinations = [station]
        for amb in sim.available():
            if amb.id != amb_id:
                destinations.append(amb.destination)
        return self._build(sim, destinations)

    def current(self, sim) -> np.ndarray:
        """Features of the fleet as it stands, with no action applied."""
        return self._build(sim, [a.destination for a in sim.available()])

    def _build(self, sim, destinations: list[int]) -> np.ndarray:
        city = self.city
        q = self.busy_prob
        n_units = len(sim.ambulances)

        if destinations:
            rows = [city.index_of(d) for d in destinations]
            counts = self._cov[rows].sum(axis=0)
            nearest = self._tt[rows].min(axis=0)
        else:
            counts = np.zeros(len(city.nodes), dtype=int)
            nearest = np.full(len(city.nodes), self._tt.max())

        # Probability at least one covering unit is free, per demand node.
        covered = 1.0 - q**counts
        zone_cov = [
            float(self._demand[mask] @ covered[mask]) / self._zone_demand[z]
            for z, mask in enumerate(self._zone_masks)
        ]

        minutes = sim.time % 1440.0
        angle = 2.0 * np.pi * minutes / 1440.0

        return np.array(
            [
                1.0,
                len(destinations) / n_units,
                np.sin(angle),
                np.cos(angle),
                float(self._demand @ covered),
                min(zone_cov),
                # Mean distance a call would travel to reach it. Coverage is blind to
                # improvements inside the target; this is not, which is the Phase 3
                # lesson written into the state description.
                float(self._demand @ nearest) / 10.0,
            ],
            dtype=float,
        )
