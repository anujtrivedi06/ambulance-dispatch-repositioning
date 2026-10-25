"""Neural fitted value iteration over post-decision states.

The linear ADP of `adp.py` failed for a diagnosed reason: least-squares TD fits value
*prediction*, and the directions an action can move the state contribute almost nothing to
prediction error, so the fit has no incentive to get them right. Two things could rescue
it -- a richer function class, or a training signal that cares about action ranking. This
module tries the first, keeping everything else identical so the comparison is clean.

The network is a value function over post-decision states, not a Q-network over (state,
action). Those are the same thing here: the post-decision state is a deterministic
function of state and action, so `max_a Q(s, a)` is `max_a V(s^a)`, and the value form
needs no output head per station -- which matters because the station set changes with the
city and a per-station head would not transfer.

It is given a richer input than the linear model: the per-zone coverages are restored.
They were dropped from the linear features because they were collinear (0.97 and 1.00
pairwise), which makes a linear fit unidentifiable -- but a network with weight decay can
hold correlated inputs without its parameters becoming arbitrary, so this is exactly the
place to put them back and see whether the extra information was worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .adp import DEFAULT_BETA, rewards_between
from .city import City, generate_calls
from .experiment import CALL_SEED_BASE, SIM_SEED_BASE
from .features import FeatureExtractor
from .policies.base import Policy
from .policies.coverage import DMEXCLP
from .sim import Simulator


def rich_features(extractor: FeatureExtractor, sim, amb_id: int, station: int) -> np.ndarray:
    """The linear feature vector plus the per-zone coverages the linear model could not use."""
    base = extractor.post_decision(sim, amb_id, station)
    destinations = [station] + [
        a.destination for a in sim.available() if a.id != amb_id
    ]
    rows = [extractor.city.index_of(d) for d in destinations]
    counts = extractor._cov[rows].sum(axis=0)
    covered = 1.0 - extractor.busy_prob**counts
    zone_cov = [
        float(extractor._demand[m] @ covered[m]) / extractor._zone_demand[z]
        for z, m in enumerate(extractor._zone_masks)
    ]
    return np.concatenate([base, zone_cov])


class ValueNet(nn.Module):
    """Small MLP. The input is 11 features and the data is tens of thousands of
    transitions, so capacity is not the binding constraint and a wider net only
    overfits the exogenous noise faster."""

    def __init__(self, d_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NFQTransition:
    phi: np.ndarray
    reward: float
    discount: float
    next_candidates: np.ndarray  # [n_stations, d]: every action open at the next epoch


class NFQCollector(Policy):
    """Epsilon-greedy over the current network, recording every candidate it considered.

    The candidate set has to be stored, not just the chosen action: the TD target is
    `r + gamma * max over next actions`, so training needs the features of actions the
    behaviour policy did not take.
    """

    name = "nfq-collector"

    def __init__(
        self,
        extractor: FeatureExtractor,
        net: ValueNet | None = None,
        epsilon: float = 0.3,
        seed: int = 0,
        fallback: Policy | None = None,
    ) -> None:
        self.extractor = extractor
        self.net = net
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fallback = fallback if fallback is not None else DMEXCLP()
        self.decisions: list[tuple[float, np.ndarray, np.ndarray]] = []

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.decisions = []

    def dispatch(self, sim, call):
        return self.fallback.dispatch(sim, call)

    def reposition(self, sim, amb_id: int) -> int:
        stations = sim.city.bases
        candidates = np.array(
            [rich_features(self.extractor, sim, amb_id, s) for s in stations]
        )
        if self.net is None or self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(len(stations)))
        else:
            with torch.no_grad():
                values = self.net(torch.as_tensor(candidates, dtype=torch.float32))
            idx = int(torch.argmax(values))
        self.decisions.append((sim.time, candidates[idx], candidates))
        return stations[idx]


class DeepValuePolicy(Policy):
    """Greedy over the learned network. One forward pass per decision, no simulation."""

    def __init__(self, net: ValueNet, extractor: FeatureExtractor, label: str = "deep-rl") -> None:
        self.net = net
        self.extractor = extractor
        self.label = label
        self.sims_run = 0

    @property
    def name(self) -> str:
        return self.label

    def dispatch(self, sim, call):
        available = sim.available()
        if not available:
            return None
        return min(available, key=lambda a: sim.travel_time_to(a, call.node)).id

    def reposition(self, sim, amb_id: int) -> int:
        stations = sim.city.bases
        candidates = np.array(
            [rich_features(self.extractor, sim, amb_id, s) for s in stations]
        )
        with torch.no_grad():
            values = self.net(torch.as_tensor(candidates, dtype=torch.float32))
        return stations[int(torch.argmax(values))]


def collect_nfq(
    city: City,
    collector: NFQCollector,
    episodes: int,
    days: int = 3,
    n_ambulances: int = 6,
    home_bases: list[int] | None = None,
    reward_mode: str = "coverage",
    target: float = 8.0,
    beta: float = DEFAULT_BETA,
    seed_offset: int = 0,
) -> list[NFQTransition]:
    out: list[NFQTransition] = []
    for ep in range(episodes):
        calls = generate_calls(
            city, days * 24 * 60,
            np.random.default_rng(CALL_SEED_BASE + 20_000 + seed_offset + ep),
        )
        collector.reset()
        sim = Simulator(
            city, calls, n_ambulances=n_ambulances,
            seed=SIM_SEED_BASE + 20_000 + seed_offset + ep, home_bases=home_bases,
        )
        records = sim.run(collector)
        if len(collector.decisions) < 2:
            continue
        times = np.array([t for t, _, _ in collector.decisions])
        rewards = rewards_between(records, times, reward_mode, target)
        gaps = np.diff(times)
        for k in range(len(rewards)):
            out.append(
                NFQTransition(
                    phi=collector.decisions[k][1],
                    reward=float(rewards[k]),
                    discount=float(np.exp(-beta * gaps[k])),
                    next_candidates=collector.decisions[k + 1][2],
                )
            )
    return out


def train_nfq(
    city: City,
    extractor: FeatureExtractor,
    rounds: int = 6,
    episodes: int = 20,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epsilon: float = 0.3,
    reward_mode: str = "coverage",
    seed: int = 0,
    device: str = "cpu",
    fresh_buffer: bool = False,
    **collect_kw,
) -> tuple[ValueNet, list[dict[str, float]]]:
    """Alternate collecting under the current greedy policy and refitting to TD targets.

    Targets are computed from a frozen copy of the network, refreshed once per round.
    Bootstrapping from the network being trained is the standard way this diverges, and
    with a value signal this weak it would be impossible to tell divergence from noise.
    """
    torch.manual_seed(seed)
    d_in = extractor.size + extractor.n_zones
    net = ValueNet(d_in).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    buffer: list[NFQTransition] = []

    for rnd in range(rounds):
        collector = NFQCollector(
            extractor,
            net=None if rnd == 0 else net,
            epsilon=epsilon,
            seed=seed + rnd,
        )
        if fresh_buffer:
            # Keeping only the newest data makes the fit on-policy at the cost of
            # sample efficiency; worth testing when stale off-policy data may be what
            # is poisoning the targets.
            buffer = []
        buffer.extend(
            collect_nfq(
                city, collector, episodes=episodes, reward_mode=reward_mode,
                seed_offset=1000 * rnd, **collect_kw,
            )
        )

        phi = torch.as_tensor(np.array([t.phi for t in buffer]), dtype=torch.float32, device=device)
        rew = torch.as_tensor(np.array([t.reward for t in buffer]), dtype=torch.float32, device=device)
        disc = torch.as_tensor(np.array([t.discount for t in buffer]), dtype=torch.float32, device=device)
        nxt = torch.as_tensor(
            np.array([t.next_candidates for t in buffer]), dtype=torch.float32, device=device
        )

        target_net = ValueNet(d_in).to(device)
        target_net.load_state_dict(net.state_dict())
        with torch.no_grad():
            n, a, d = nxt.shape
            best_next = target_net(nxt.reshape(n * a, d)).reshape(n, a).max(dim=1).values
            targets = rew + disc * best_next

        losses = []
        for _ in range(epochs):
            idx = torch.randperm(len(buffer), device=device)[:batch_size]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(net(phi[idx]), targets[idx])
            loss.backward()
            optimiser.step()
            losses.append(loss.detach().item())
        history.append(
            {"round": rnd + 1, "transitions": len(buffer), "loss": float(np.mean(losses))}
        )
    return net, history
