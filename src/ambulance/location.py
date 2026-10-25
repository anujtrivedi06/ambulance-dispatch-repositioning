"""Static base-location models: p-median and MEXCLP.

These answer the *offline* half of the problem -- where to put stations and how many
ambulances to home at each -- assuming nothing about the live fleet state. Phase 2 uses
them as the starting configuration that the dynamic policies then improve on.

Both are NP-hard, so both are solved by greedy construction followed by a swap-based
local search. On the graph sizes here that lands on or very near the optimum, and it
avoids an ILP solver dependency.
"""

from __future__ import annotations

import numpy as np

from .city import City

TARGET_MINUTES = 8.0


def coverage_matrix(city: City, target: float = TARGET_MINUTES) -> np.ndarray:
    """Boolean [node, node] matrix: can a unit at i reach j within `target` minutes."""
    return city.travel_matrix() <= target


# ---------------------------------------------------------------------- p-median


def p_median_cost(city: City, sites: list[int]) -> float:
    """Demand-weighted mean travel time from the nearest chosen site."""
    idx = [city.index_of(s) for s in sites]
    nearest = city.travel_matrix()[idx, :].min(axis=0)
    return float(city.demand_weights @ nearest)


def p_median(
    city: City,
    p: int,
    candidates: list[int] | None = None,
    max_passes: int = 20,
) -> list[int]:
    """Choose `p` sites minimising demand-weighted mean response distance.

    Minimising the *mean* makes p-median indifferent to whether a node is inside or
    outside the response-time target, which is exactly where MEXCLP differs.
    """
    cands = list(city.nodes if candidates is None else candidates)
    tt = city.travel_matrix()
    d = city.demand_weights

    # Greedy: repeatedly add the site that most reduces the weighted mean distance.
    sites: list[int] = []
    best = np.full(len(city.nodes), np.inf)
    while len(sites) < p:
        gains = []
        for c in cands:
            if c in sites:
                gains.append(np.inf)
                continue
            gains.append(float(d @ np.minimum(best, tt[city.index_of(c)])))
        pick = cands[int(np.argmin(gains))]
        sites.append(pick)
        best = np.minimum(best, tt[city.index_of(pick)])

    return _swap_local_search(city, sites, cands, p_median_cost, maximise=False,
                              max_passes=max_passes)


# ------------------------------------------------------------------------ MEXCLP


def expected_covered_demand(
    city: City,
    allocation: list[int],
    busy_prob: float,
    target: float = TARGET_MINUTES,
) -> float:
    """MEXCLP objective: expected fraction of demand covered within `target`.

    With `n` units able to reach a node and each independently busy with probability q,
    the chance at least one is free is `1 - q**n`. Summing that over demand gives the
    expected covered demand -- the closed form of MEXCLP's usual
    `sum_k (1-q) q**(k-1)` series.

    The independence assumption is wrong (nearby units go busy together, and q is not
    uniform), which is precisely why the simulator, not this number, is the judge.
    """
    cov = coverage_matrix(city, target)
    counts = np.zeros(len(city.nodes), dtype=int)
    for site in allocation:
        counts += cov[city.index_of(site)]
    return float(city.demand_weights @ (1.0 - busy_prob**counts))


def mexclp(
    city: City,
    n_servers: int,
    busy_prob: float = 0.35,
    target: float = TARGET_MINUTES,
    candidates: list[int] | None = None,
    max_sites: int | None = None,
    max_passes: int = 20,
) -> tuple[list[int], list[int]]:
    """Place `n_servers` ambulances to maximise expected covered demand.

    Returns `(sites, allocation)` where `sites` are the distinct stations used and
    `allocation` has one entry per ambulance, so several units may share a station --
    MEXCLP's whole point is that doubling up on a hotspot can beat spreading out, since
    the second unit covers the hotspot when the first is busy.

    `max_sites` caps how many distinct stations may be opened, for comparability with a
    fixed-station-count baseline.
    """
    cands = list(city.nodes if candidates is None else candidates)
    cov = coverage_matrix(city, target)
    d = city.demand_weights

    allocation: list[int] = []
    counts = np.zeros(len(city.nodes), dtype=int)
    for _ in range(n_servers):
        gain_scale = d * (busy_prob**counts) * (1.0 - busy_prob)
        open_sites = set(allocation)
        best_site, best_gain = None, -np.inf
        for c in cands:
            if max_sites is not None and c not in open_sites and len(open_sites) >= max_sites:
                continue
            gain = float(gain_scale @ cov[city.index_of(c)])
            if gain > best_gain:
                best_site, best_gain = c, gain
        if best_site is None:
            raise ValueError("no candidate site available under max_sites")
        allocation.append(best_site)
        counts += cov[city.index_of(best_site)]

    def objective(city_: City, alloc: list[int]) -> float:
        return expected_covered_demand(city_, alloc, busy_prob, target)

    allocation = _swap_local_search(
        city, allocation, cands, objective, maximise=True,
        max_passes=max_passes, max_sites=max_sites,
    )
    return sorted(set(allocation)), allocation


# ------------------------------------------------------------------ local search


def _swap_local_search(
    city: City,
    solution: list[int],
    candidates: list[int],
    objective,
    maximise: bool,
    max_passes: int = 20,
    max_sites: int | None = None,
) -> list[int]:
    """First-improvement swap search: move one server at a time until nothing helps."""
    current = list(solution)
    score = objective(city, current)
    for _ in range(max_passes):
        improved = False
        for i in range(len(current)):
            for c in candidates:
                if c == current[i]:
                    continue
                trial = list(current)
                trial[i] = c
                if max_sites is not None and len(set(trial)) > max_sites:
                    continue
                trial_score = objective(city, trial)
                if (trial_score > score) if maximise else (trial_score < score):
                    current, score, improved = trial, trial_score, True
                    break
            if improved:
                break
        if not improved:
            break
    return current
