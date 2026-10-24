"""City model: road graph, travel times, demand distribution, call generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

# Hourly multipliers on the base call rate, normalised to mean 1.0.
# Low overnight, morning peak, evening peak -- the usual EMS demand shape.
HOURLY_PROFILE = np.array(
    [0.45, 0.35, 0.30, 0.30, 0.35, 0.50, 0.75, 1.05, 1.30, 1.35, 1.30, 1.25,
     1.25, 1.25, 1.20, 1.20, 1.30, 1.45, 1.50, 1.35, 1.15, 0.95, 0.75, 0.55]
)


@dataclass(frozen=True)
class Call:
    """A single emergency request."""

    id: int
    node: int
    arrival_time: float
    priority: int = 1  # 1 = highest urgency
    on_scene_time: float = 10.0
    transport_required: bool = True


@dataclass
class City:
    """Road network plus the demand and travel-time models defined on it.

    Travel times are in minutes and are precomputed as a dense all-pairs matrix, since
    policies query them in their inner loops and the graphs here stay small.
    """

    graph: nx.Graph
    bases: list[int]
    hospitals: list[int]
    demand_weights: np.ndarray  # per-node probability of being a call's location
    base_call_rate: float = 2.0  # mean calls/hour, before the hourly profile
    _tt: np.ndarray = field(init=False, repr=False)
    _index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = sorted(self.graph.nodes)
        self._index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        self._tt = np.full((n, n), np.inf)
        paths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="time"))
        for u, dists in paths.items():
            for v, d in dists.items():
                self._tt[self._index[u], self._index[v]] = d
        if not np.isfinite(self._tt).all():
            raise ValueError("city graph is not connected")

    @property
    def nodes(self) -> list[int]:
        return sorted(self.graph.nodes)

    def travel_time(self, u: int, v: int) -> float:
        """Free-flow travel time in minutes from node u to node v."""
        return float(self._tt[self._index[u], self._index[v]])

    def travel_matrix(self) -> np.ndarray:
        return self._tt

    def arrival_rate(self, t: float) -> float:
        """Instantaneous call rate (calls/hour) at simulation time t (minutes)."""
        hour = int((t / 60.0) % 24)
        return self.base_call_rate * float(HOURLY_PROFILE[hour])

    def sample_call_node(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.nodes, p=self.demand_weights))

    def zone_of(self, node: int) -> int:
        """Coarse zone label, used for the equity metrics."""
        return self.graph.nodes[node].get("zone", 0)


def grid_city(
    width: int = 8,
    height: int = 8,
    block_minutes: float = 1.5,
    n_bases: int = 4,
    n_hospitals: int = 2,
    hotspots: int = 3,
    base_call_rate: float = 4.0,
    seed: int = 0,
) -> City:
    """Build a synthetic Manhattan-grid city with clustered demand hotspots.

    Demand is a mixture of a uniform background and Gaussian hotspots in grid space, which
    gives the spatial imbalance that makes repositioning matter at all.
    """
    rng = np.random.default_rng(seed)
    g = nx.grid_2d_graph(width, height)
    # Relabel (x, y) -> flat int ids so policies can index arrays directly.
    mapping = {(x, y): y * width + x for x, y in g.nodes}
    nx.relabel_nodes(g, mapping, copy=False)
    for node in g.nodes:
        x, y = node % width, node // width
        g.nodes[node]["pos"] = (x, y)
        # Quadrant zones for equity reporting.
        g.nodes[node]["zone"] = (0 if x < width / 2 else 1) + (0 if y < height / 2 else 2)
    for u, v in g.edges:
        # Mild congestion jitter so ties between symmetric routes get broken.
        g.edges[u, v]["time"] = block_minutes * float(rng.uniform(0.85, 1.25))

    coords = np.array([g.nodes[n]["pos"] for n in sorted(g.nodes)], dtype=float)
    weights = np.full(len(coords), 0.35 / len(coords))
    centres = rng.uniform([0, 0], [width - 1, height - 1], size=(hotspots, 2))
    for centre in centres:
        d2 = ((coords - centre) ** 2).sum(axis=1)
        bump = np.exp(-d2 / (2 * (0.18 * min(width, height)) ** 2))
        weights += 0.65 / hotspots * bump / bump.sum()
    weights /= weights.sum()

    # Bases spread by k-means-ish greedy max-min spacing over the grid.
    nodes = sorted(g.nodes)
    bases = [int(np.argmax(weights))]
    while len(bases) < n_bases:
        d = np.min([[np.hypot(*(coords[i] - coords[b])) for b in bases] for i in range(len(nodes))], axis=1)
        bases.append(int(np.argmax(d * (0.5 + weights / weights.max()))))
    hospitals = [nodes[0], nodes[-1]][:n_hospitals]

    return City(
        graph=g,
        bases=[nodes[b] for b in bases],
        hospitals=hospitals,
        demand_weights=weights,
        base_call_rate=base_call_rate,
    )


def generate_calls(
    city: City, horizon_minutes: float, rng: np.random.Generator
) -> list[Call]:
    """Sample a non-homogeneous Poisson call stream by thinning.

    Generating the whole stream up front (rather than inside the simulator) means every
    policy can be evaluated on an identical set of calls -- paired comparison, far lower
    variance than independent runs.
    """
    lam_max = city.base_call_rate * float(HOURLY_PROFILE.max()) / 60.0  # per minute
    calls: list[Call] = []
    t = 0.0
    next_id = 0
    while True:
        t += rng.exponential(1.0 / lam_max)
        if t >= horizon_minutes:
            break
        if rng.random() <= city.arrival_rate(t) / 60.0 / lam_max:
            priority = 1 if rng.random() < 0.6 else 2
            calls.append(
                Call(
                    id=next_id,
                    node=city.sample_call_node(rng),
                    arrival_time=t,
                    priority=priority,
                    on_scene_time=float(rng.lognormal(np.log(9.0), 0.35)),
                    transport_required=bool(rng.random() < 0.75),
                )
            )
            next_id += 1
    return calls
