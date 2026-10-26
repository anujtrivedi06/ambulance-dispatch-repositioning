import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city, mexclp, summarise
from ambulance.city import Call
from ambulance.metrics import summarise as summarise_metrics
from ambulance.policies import DMEXCLP, EquityDMEXCLP, SelectiveDispatch


@pytest.fixture(scope="module")
def setup():
    city = grid_city(width=6, height=6, base_call_rate=4.0, seed=3)
    sites, alloc = mexclp(city, 6, busy_prob=0.35, max_sites=4)
    return city.with_bases(sites), alloc


def run(city, policy, alloc, seed=1, days=3, n_ambulances=6):
    calls = generate_calls(city, days * 24 * 60, np.random.default_rng(seed))
    sim = Simulator(city, calls, n_ambulances=n_ambulances, seed=seed, home_bases=alloc)
    return sim.run(policy), sim


# -------------------------------------------------------------------- metrics


def test_priority_metrics_split_the_calls_correctly(setup):
    city, alloc = setup
    records, _ = run(city, DMEXCLP(), alloc)
    m = summarise(records, city)
    assert m["n_calls_p1"] + m["n_calls_p2"] == m["served"]
    assert 0.0 <= m["coverage_p1"] <= 1.0
    assert 0.0 <= m["coverage_p2"] <= 1.0


def test_zone_gap_is_consistent_with_worst_zone(setup):
    city, alloc = setup
    records, _ = run(city, DMEXCLP(), alloc)
    m = summarise(records, city)
    assert m["zone_coverage_gap"] >= 0.0
    assert m["worst_zone_coverage"] + m["zone_coverage_gap"] <= 1.0 + 1e-9


def test_priority_metrics_are_empty_safe():
    assert summarise_metrics([], grid_city(width=4, height=4, seed=0))["coverage_p1"] == 0.0


# --------------------------------------------------------------------- equity


def test_equity_at_lambda_zero_is_exactly_dmexclp(setup):
    """The frontier must start at the earlier phases' result, not near it."""
    city, alloc = setup
    a, _ = run(city, DMEXCLP(), alloc)
    b, _ = run(city, EquityDMEXCLP(lam=0.0), alloc)
    ma, mb = summarise(a, city), summarise(b, city)
    for key in ("coverage", "mean_response", "worst_zone_coverage"):
        assert ma[key] == pytest.approx(mb[key], abs=1e-12), key


def test_equity_rejects_a_weight_outside_the_unit_interval():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            EquityDMEXCLP(lam=bad)


def test_equity_only_ever_picks_stations(setup):
    city, alloc = setup
    _, sim = run(city, EquityDMEXCLP(lam=0.5), alloc)
    assert all(a.destination in city.bases or a.node in city.bases
               for a in sim.ambulances)


# ------------------------------------------------------------------ selective


def test_selective_withholds_from_routine_calls_when_cover_is_thin(setup):
    """With one unit left and others busy, a routine call must wait."""
    city, alloc = setup
    _, sim = run(city, DMEXCLP(), alloc, seed=4)
    # Force the fleet to one available unit.
    from ambulance.sim import Status

    for amb in sim.ambulances[1:]:
        amb.status = Status.TO_SCENE
    sim.ambulances[0].status = Status.IDLE

    policy = SelectiveDispatch(DMEXCLP(), reserve=1)
    routine = Call(id=-1, node=city.nodes[5], arrival_time=sim.time, priority=2)
    urgent = Call(id=-2, node=city.nodes[5], arrival_time=sim.time, priority=1)
    assert policy.dispatch(sim, routine) is None
    assert policy.dispatch(sim, urgent) == 0


def test_selective_never_deadlocks_with_an_idle_fleet(setup):
    """The guard that matters.

    If every unit is idle, nothing will ever become free, no BECOME_FREE event will
    fire, and a withheld call would wait for the rest of the simulation. The policy must
    dispatch instead of holding.
    """
    city, alloc = setup
    calls = generate_calls(city, 60.0, np.random.default_rng(9))
    sim = Simulator(city, calls, n_ambulances=6, seed=9, home_bases=alloc)
    policy = SelectiveDispatch(DMEXCLP(), reserve=6)  # reserve >= fleet size
    routine = Call(id=-1, node=city.nodes[3], arrival_time=0.0, priority=2)
    assert policy.dispatch(sim, routine) is not None


def test_selective_serves_every_call_despite_withholding(setup):
    city, alloc = setup
    records, _ = run(city, SelectiveDispatch(DMEXCLP(), reserve=3), alloc)
    assert all(r.reached_at is not None for r in records)
    assert all(r.response_time >= 0 for r in records)


def test_selective_helps_urgent_calls_and_costs_routine_ones(setup):
    city, alloc = setup
    base, _ = run(city, DMEXCLP(), alloc, seed=7)
    held, _ = run(city, SelectiveDispatch(DMEXCLP(), reserve=2), alloc, seed=7)
    mb, mh = summarise(base, city), summarise(held, city)
    assert mh["coverage_p1"] > mb["coverage_p1"]
    assert mh["coverage_p2"] < mb["coverage_p2"]
    assert mh["mean_response_p2"] > mb["mean_response_p2"]


def test_selective_with_zero_reserve_changes_nothing(setup):
    city, alloc = setup
    a, _ = run(city, DMEXCLP(), alloc, seed=5)
    b, _ = run(city, SelectiveDispatch(DMEXCLP(), reserve=0), alloc, seed=5)
    assert [r.response_time for r in a] == [r.response_time for r in b]


def test_selective_rejects_a_negative_reserve():
    with pytest.raises(ValueError):
        SelectiveDispatch(DMEXCLP(), reserve=-1)
