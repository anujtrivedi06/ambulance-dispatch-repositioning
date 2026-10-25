import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import grid_city
from ambulance.location import (
    coverage_matrix,
    expected_covered_demand,
    mexclp,
    p_median,
    p_median_cost,
)


@pytest.fixture(scope="module")
def city():
    return grid_city(width=6, height=6, base_call_rate=4.0, seed=3)


def test_coverage_matrix_is_reflexive_and_symmetric(city):
    cov = coverage_matrix(city, target=8.0)
    assert cov.diagonal().all()  # a unit always covers its own node
    assert (cov == cov.T).all()
    assert not coverage_matrix(city, target=0.0)[0, 1]


def test_p_median_returns_distinct_sites_and_beats_the_hand_placed_ones(city):
    sites = p_median(city, 4)
    assert len(set(sites)) == 4
    assert set(sites) <= set(city.nodes)
    assert p_median_cost(city, sites) <= p_median_cost(city, city.bases)


def test_p_median_cost_falls_as_stations_are_added(city):
    costs = [p_median_cost(city, p_median(city, p)) for p in (1, 2, 4)]
    assert costs[0] > costs[1] > costs[2]


def test_expected_covered_demand_matches_the_closed_form(city):
    """Two units at one site must give exactly 1 - q**2 coverage on what they reach."""
    q = 0.4
    site = city.nodes[0]
    cov = coverage_matrix(city)[city.index_of(site)]
    expected = float(city.demand_weights[cov].sum() * (1 - q**2))
    assert expected_covered_demand(city, [site, site], q) == pytest.approx(expected)


def test_expected_covered_demand_is_monotone_in_fleet_size(city):
    q = 0.35
    sites = p_median(city, 4)
    values = [
        expected_covered_demand(city, sites[: k + 1], q) for k in range(len(sites))
    ]
    assert values == sorted(values)


def test_mexclp_allocates_every_unit_and_respects_the_station_cap(city):
    sites, alloc = mexclp(city, 6, busy_prob=0.35, max_sites=4)
    assert len(alloc) == 6
    assert set(sites) == set(alloc)
    assert len(sites) <= 4


def test_mexclp_beats_the_hand_placed_layout_on_its_own_objective(city):
    q = 0.35
    _, alloc = mexclp(city, 6, busy_prob=q, max_sites=len(city.bases))
    hand = [city.bases[i % len(city.bases)] for i in range(6)]
    assert expected_covered_demand(city, alloc, q) > expected_covered_demand(
        city, hand, q
    )


def test_mexclp_doubles_up_when_units_outnumber_useful_sites(city):
    """With one candidate site, all units must pile onto it rather than error."""
    only = [city.nodes[10]]
    sites, alloc = mexclp(city, 3, candidates=only)
    assert sites == only
    assert alloc == only * 3


def test_p_median_and_mexclp_optimise_different_things(city):
    """p-median minimises mean distance; MEXCLP maximises covered demand.

    Each should win on its own objective, which is why the project reports both.
    """
    q = 0.35
    pm = p_median(city, 4)
    _, mx_alloc = mexclp(city, 4, busy_prob=q, max_sites=4)
    assert p_median_cost(city, pm) <= p_median_cost(city, sorted(set(mx_alloc)))
    assert expected_covered_demand(city, mx_alloc, q) >= expected_covered_demand(
        city, pm, q
    )
