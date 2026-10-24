import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city, summarise
from ambulance.city import Call
from ambulance.policies import NearestIdleHomeBase, RandomPolicy
from ambulance.sim import DISPATCHABLE, Status


@pytest.fixture(scope="module")
def city():
    return grid_city(width=6, height=6, base_call_rate=4.0, seed=3)


def test_graph_is_connected_and_travel_times_symmetric(city):
    tt = city.travel_matrix()
    assert np.isfinite(tt).all()
    assert np.allclose(tt, tt.T)
    assert np.allclose(np.diag(tt), 0.0)


def test_demand_weights_are_a_distribution(city):
    assert city.demand_weights.shape == (len(city.nodes),)
    assert pytest.approx(1.0) == city.demand_weights.sum()
    assert (city.demand_weights >= 0).all()


def test_call_stream_is_time_ordered_and_rate_is_plausible(city):
    calls = generate_calls(city, 7 * 24 * 60, np.random.default_rng(0))
    times = [c.arrival_time for c in calls]
    assert times == sorted(times)
    per_hour = len(calls) / (7 * 24)
    assert 0.6 * city.base_call_rate < per_hour < 1.4 * city.base_call_rate


def test_every_call_is_eventually_served(city):
    calls = generate_calls(city, 2 * 24 * 60, np.random.default_rng(1))
    sim = Simulator(city, calls, n_ambulances=6, seed=1)
    records = sim.run(NearestIdleHomeBase())
    assert len(records) == len(calls)
    assert all(r.reached_at is not None for r in records)
    assert all(r.response_time >= 0 for r in records)


def test_no_ambulance_serves_two_calls_at_once(city):
    """The invariant that stale ARRIVE_BASE events used to break.

    A dispatched unit must not be marked idle until its mission has finished, so the
    busy intervals of calls sharing an ambulance may never overlap.
    """
    calls = generate_calls(city, 3 * 24 * 60, np.random.default_rng(2))
    sim = Simulator(city, calls, n_ambulances=3, seed=2)  # small fleet -> heavy reuse
    records = sim.run(RandomPolicy(seed=0))  # random dispatch picks relocating units
    by_amb: dict[int, list[tuple[float, float]]] = {}
    for r in records:
        by_amb.setdefault(r.amb_id, []).append((r.assigned_at, r.reached_at))
    for amb_id, spans in by_amb.items():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert start >= end - 1e-9, f"ambulance {amb_id} double-booked"


def test_nearest_idle_beats_random(city):
    calls = generate_calls(city, 5 * 24 * 60, np.random.default_rng(3))
    scores = {}
    for pol in (RandomPolicy(seed=0), NearestIdleHomeBase()):
        sim = Simulator(city, calls, n_ambulances=6, seed=5)
        scores[pol.name] = summarise(sim.run(pol), city)
    assert scores["nearest-idle/home-base"]["coverage"] > scores["random"]["coverage"]
    assert (
        scores["nearest-idle/home-base"]["mean_response"]
        < scores["random"]["mean_response"]
    )


def test_calls_queue_when_the_whole_fleet_is_busy(city):
    """Simultaneous calls with one unit: the second must wait, not vanish."""
    burst = [
        Call(id=i, node=city.nodes[-1], arrival_time=1.0 + i * 0.01, on_scene_time=20.0)
        for i in range(3)
    ]
    sim = Simulator(city, burst, n_ambulances=1, seed=0)
    records = sim.run(NearestIdleHomeBase())
    waits = sorted(r.wait_to_assign for r in records)
    assert waits[0] == pytest.approx(0.0, abs=1e-6)
    assert waits[1] > 0 and waits[2] > waits[1]


def test_all_units_end_dispatchable(city):
    calls = generate_calls(city, 24 * 60, np.random.default_rng(4))
    sim = Simulator(city, calls, n_ambulances=4, seed=4)
    sim.run(NearestIdleHomeBase())
    assert all(a.status in DISPATCHABLE for a in sim.ambulances)
    assert all(a.call is None for a in sim.ambulances)
    assert all(a.status is not Status.AT_SCENE for a in sim.ambulances)
