import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city, mexclp, summarise
from ambulance.adp import (
    CollectingPolicy,
    bellman_residual,
    collect,
    fit_lstd,
    rewards_between,
)
from ambulance.features import FEATURE_NAMES, FeatureExtractor
from ambulance.policies import ADPPolicy, DMEXCLP
from ambulance.policy_search import ACTION_FEATURES, expand


@pytest.fixture(scope="module")
def setup():
    city = grid_city(width=6, height=6, base_call_rate=4.0, seed=3)
    sites, alloc = mexclp(city, 6, busy_prob=0.35, max_sites=4)
    run_city = city.with_bases(sites)
    return run_city, alloc, FeatureExtractor(run_city)


@pytest.fixture
def mid_run(setup):
    run_city, alloc, _ = setup
    calls = generate_calls(run_city, 2 * 24 * 60, np.random.default_rng(1))
    sim = Simulator(run_city, calls, n_ambulances=6, seed=1, home_bases=alloc)
    sim.run(DMEXCLP(), horizon=600)
    return sim


# ------------------------------------------------------------------- features


def test_feature_vector_shape_and_finiteness(setup, mid_run):
    run_city, _, ex = setup
    phi = ex.post_decision(mid_run, 0, run_city.bases[0])
    assert phi.shape == (len(FEATURE_NAMES),) == (ex.size,)
    assert np.isfinite(phi).all()
    assert phi[0] == 1.0  # bias


def test_features_are_not_collinear(setup):
    """The reason three features were dropped. Guard it so they cannot creep back."""
    run_city, alloc, ex = setup
    transitions = collect(
        run_city,
        CollectingPolicy(ex, epsilon=0.4, seed=0),
        episodes=4,
        days=2,
        home_bases=alloc,
    )
    X = np.array([t.phi for t in transitions])
    assert np.linalg.matrix_rank(X) == X.shape[1], "design matrix is rank deficient"


def test_action_varying_features_actually_vary(setup, mid_run):
    """If an action could not change these, the learned policy could not do anything."""
    run_city, _, ex = setup
    idx = [FEATURE_NAMES.index(n) for n in ACTION_FEATURES]
    vectors = np.array(
        [ex.post_decision(mid_run, 0, s) for s in run_city.bases]
    )
    assert vectors[:, idx].std(axis=0).max() > 1e-6
    # And the rest must be identical across actions, or the argmax is not well posed.
    other = [k for k in range(len(FEATURE_NAMES)) if k not in idx]
    assert np.allclose(vectors[:, other], vectors[0, other])


def test_coverage_features_improve_when_a_unit_is_added(setup, mid_run):
    run_city, _, ex = setup
    i_ecd = FEATURE_NAMES.index("expected_covered_demand")
    with_unit = ex.post_decision(mid_run, 0, run_city.bases[0])[i_ecd]
    without = ex.current(mid_run)[i_ecd]
    assert with_unit >= without - 1e-9


# ------------------------------------------------------------------------ adp


def test_rewards_land_in_the_right_interval(setup):
    class FakeRecord:
        def __init__(self, reached_at, response_time):
            self.reached_at = reached_at
            self.response_time = response_time

    records = [FakeRecord(5.0, 4.0), FakeRecord(15.0, 12.0), FakeRecord(25.0, 2.0)]
    times = np.array([0.0, 10.0, 20.0, 30.0])
    got = rewards_between(records, times, "coverage", target=8.0)
    assert got.tolist() == [1.0, 0.0, 1.0]  # second call missed the target


def test_rewards_reject_an_unknown_mode():
    with pytest.raises(ValueError):
        rewards_between([], np.array([0.0, 1.0]), "vibes", target=8.0)


def test_lstd_fit_is_deterministic_and_reduces_td_error(setup):
    run_city, alloc, ex = setup
    transitions = collect(
        run_city,
        CollectingPolicy(ex, epsilon=0.3, seed=0),
        episodes=4,
        days=2,
        home_bases=alloc,
    )
    a = fit_lstd(transitions)
    b = fit_lstd(transitions)
    assert np.allclose(a, b)
    zero = np.zeros_like(a)
    assert bellman_residual(transitions, a) < bellman_residual(transitions, zero)


def test_lstd_needs_data():
    with pytest.raises(ValueError):
        fit_lstd([])


def test_collection_is_reproducible(setup):
    run_city, alloc, ex = setup
    kw = dict(episodes=2, days=2, home_bases=alloc)
    first = collect(run_city, CollectingPolicy(ex, epsilon=0.3, seed=0), **kw)
    second = collect(run_city, CollectingPolicy(ex, epsilon=0.3, seed=0), **kw)
    assert len(first) == len(second)
    assert np.allclose([t.reward for t in first], [t.reward for t in second])


def test_discounts_are_in_range_and_shrink_with_the_gap(setup):
    run_city, alloc, ex = setup
    transitions = collect(
        run_city, CollectingPolicy(ex, seed=0), episodes=2, days=2, home_bases=alloc
    )
    discounts = np.array([t.discount for t in transitions])
    assert ((discounts > 0) & (discounts <= 1)).all()
    assert discounts.min() < discounts.max()  # irregular epochs, so they must differ


# -------------------------------------------------------------------- policy


def test_adp_policy_only_picks_stations(setup, mid_run):
    run_city, _, ex = setup
    theta = np.zeros(len(FEATURE_NAMES))
    theta[FEATURE_NAMES.index("expected_covered_demand")] = 1.0
    assert ADPPolicy(theta, ex).reposition(mid_run, 0) in run_city.bases


def test_greedy_on_coverage_reproduces_dmexclp(setup):
    """Greedy on expected covered demand is DMEXCLP's rule, written differently.

    This is what makes the Phase 4 comparison meaningful: the learners' hypothesis class
    provably contains the heuristic they are being compared against, so losing to it is a
    statement about the training signal and not about the features.
    """
    run_city, alloc, ex = setup
    theta = np.zeros(len(FEATURE_NAMES))
    theta[FEATURE_NAMES.index("expected_covered_demand")] = 1.0

    calls = generate_calls(run_city, 3 * 24 * 60, np.random.default_rng(11))
    scores = []
    for policy in (DMEXCLP(), ADPPolicy(theta, ex)):
        sim = Simulator(run_city, calls, n_ambulances=6, seed=11, home_bases=alloc)
        scores.append(summarise(sim.run(policy), run_city))
    assert scores[0]["coverage"] == pytest.approx(scores[1]["coverage"], abs=1e-9)
    assert scores[0]["mean_response"] == pytest.approx(scores[1]["mean_response"], abs=1e-9)


def test_expand_places_weights_on_the_right_features():
    theta = expand(np.array([1.0, 2.0, 3.0]))
    assert theta.shape == (len(FEATURE_NAMES),)
    for value, name in zip((1.0, 2.0, 3.0), ACTION_FEATURES):
        assert theta[FEATURE_NAMES.index(name)] == value
    assert theta[FEATURE_NAMES.index("bias")] == 0.0
