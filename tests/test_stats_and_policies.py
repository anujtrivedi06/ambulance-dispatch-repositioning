import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import grid_city, mexclp
from ambulance.experiment import column, evaluate
from ambulance.policies import DMEXCLP, NearestIdleHomeBase
from ambulance.sim import Simulator, Status
from ambulance.stats import paired_test


@pytest.fixture(scope="module")
def city():
    return grid_city(width=6, height=6, base_call_rate=4.0, seed=3)


# ------------------------------------------------------------------------- stats


def test_identical_samples_are_not_significant():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = paired_test(x, x.copy(), metric="coverage")
    assert r.mean_diff == 0.0
    assert not r.significant


def test_a_consistent_shift_is_significant_with_the_right_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, size=30)
    b = a + 0.5  # same scenarios, uniformly better
    r = paired_test(a, b, metric="mean_response")
    assert r.mean_diff == pytest.approx(0.5)
    assert r.ci_low > 0 and r.significant


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=30)
    b = rng.normal(0.0, 1.0, size=30)
    assert not paired_test(a, b, metric="coverage").significant


def test_mismatched_sample_lengths_are_rejected():
    with pytest.raises(ValueError):
        paired_test(np.zeros(5), np.zeros(4))


# ---------------------------------------------------------------------- policies


def test_dmexclp_only_ever_repositions_to_a_station(city):
    calls_city = city
    sim = Simulator(calls_city, [], n_ambulances=4, seed=0)
    pol = DMEXCLP()
    for amb in sim.ambulances:
        assert pol.reposition(sim, amb.id) in calls_city.bases


def test_dmexclp_fills_the_gap_left_by_busy_units(city):
    """With every other unit committed to one corner, the freed unit should not
    park next to them -- it should cover the station furthest from the busy cluster."""
    sim = Simulator(city, [], n_ambulances=4, seed=0)
    corner = min(city.bases, key=lambda b: b)
    for amb in sim.ambulances[1:]:
        amb.node = corner
        amb.status = Status.TO_SCENE  # busy: excluded from the coverage count
    choice = DMEXCLP().reposition(sim, 0)
    assert choice != corner


def test_dmexclp_beats_static_home_base_on_paired_runs(city):
    sites, alloc = mexclp(city, 6, busy_prob=0.35, max_sites=len(city.bases))
    mx_city = city.with_bases(sites)
    kw = dict(days=2, reps=6, n_ambulances=6, home_bases=alloc)
    static = evaluate(NearestIdleHomeBase(), mx_city, **kw)
    dynamic = evaluate(DMEXCLP(busy_prob=0.35), mx_city, **kw)
    assert column(dynamic, "coverage").mean() > column(static, "coverage").mean()
    assert column(dynamic, "mean_response").mean() < column(
        static, "mean_response"
    ).mean()


def test_paired_replications_are_reproducible(city):
    kw = dict(days=1, reps=3, n_ambulances=5)
    first = evaluate(NearestIdleHomeBase(), city, **kw)
    second = evaluate(NearestIdleHomeBase(), city, **kw)
    assert column(first, "coverage").tolist() == column(second, "coverage").tolist()


def test_policies_see_the_same_call_streams(city):
    """The pairing that the significance tests assume: identical call counts per rep."""
    kw = dict(days=2, reps=4, n_ambulances=6)
    a = evaluate(NearestIdleHomeBase(), city, **kw)
    b = evaluate(DMEXCLP(), city, **kw)
    assert column(a, "n_calls").tolist() == column(b, "n_calls").tolist()
