import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city
from ambulance.policies import DMEXCLP, MCTSPolicy, RolloutPolicy
from ambulance.policies.mcts import _Pause, _PausingPolicy
from ambulance.sim import _FORK_ID_BASE, DISPATCHABLE, Status


@pytest.fixture(scope="module")
def city():
    return grid_city(width=6, height=6, base_call_rate=4.0, seed=3)


@pytest.fixture
def mid_run(city):
    """A simulator advanced to the middle of a day, with work still in flight."""
    calls = generate_calls(city, 2 * 24 * 60, np.random.default_rng(1))
    sim = Simulator(city, calls, n_ambulances=6, seed=1)
    sim.run(DMEXCLP(), horizon=600)
    return sim


# ------------------------------------------------------------------ the big one


def test_a_fork_cannot_see_the_real_future(mid_run):
    """The whole validity of Phase 3 rests on this.

    If forks inherited the real pending arrivals, a lookahead policy would be choosing
    stations while knowing where the next calls land, and every number it produced would
    be unreachable in deployment.
    """
    fork = mid_run.fork(np.random.default_rng(0), horizon=240.0)
    real_future_ids = {
        c.id for c in mid_run.calls if c.arrival_time > mid_run.time
    }
    assert real_future_ids, "fixture should leave calls still to come"

    assert not (real_future_ids & set(fork.records)), "fork leaked real future calls"
    for ev in fork._events:
        if ev.call is not None:
            assert ev.call.id not in real_future_ids
    # Everything the fork invents is id-tagged so it can never be mistaken for real.
    assert all(cid >= _FORK_ID_BASE for cid in fork.sampled_ids)


def test_a_fork_samples_a_plausible_future(mid_run):
    fork = mid_run.fork(np.random.default_rng(0), horizon=240.0)
    assert fork.sampled_ids, "nothing sampled over four hours at 4 calls/h"
    arrivals = [fork.records[c].call.arrival_time for c in fork.sampled_ids]
    assert all(mid_run.time < t <= mid_run.time + 240.0 for t in arrivals)


def test_running_a_fork_leaves_the_real_simulator_untouched(mid_run):
    before_time = mid_run.time
    before_units = [
        (a.id, a.status, a.node, a.trip, list(a.route)) for a in mid_run.ambulances
    ]
    before_records = {
        cid: (r.assigned_at, r.reached_at, r.amb_id)
        for cid, r in mid_run.records.items()
    }
    before_events = len(mid_run._events)

    fork = mid_run.fork(np.random.default_rng(7), horizon=480.0)
    fork.resume(DMEXCLP(), until=mid_run.time + 480.0)

    assert mid_run.time == before_time
    assert [
        (a.id, a.status, a.node, a.trip, list(a.route)) for a in mid_run.ambulances
    ] == before_units
    assert {
        cid: (r.assigned_at, r.reached_at, r.amb_id)
        for cid, r in mid_run.records.items()
    } == before_records
    assert len(mid_run._events) == before_events


def test_a_fork_keeps_work_already_committed(mid_run):
    """Missions in flight must survive the fork, or the lookahead starts from a lie."""
    busy = [a for a in mid_run.ambulances if a.status not in DISPATCHABLE]
    assert busy, "fixture should leave units mid-mission"
    fork = mid_run.fork(np.random.default_rng(0), horizon=120.0)
    for unit in busy:
        twin = fork.ambulances[unit.id]
        assert twin.status == unit.status
        assert twin.node == unit.node
        assert twin.route == unit.route


def test_forks_with_the_same_seed_sample_the_same_future(mid_run):
    """Common random numbers: candidate actions must be compared on one future."""
    a = mid_run.fork(np.random.default_rng(42), horizon=240.0)
    b = mid_run.fork(np.random.default_rng(42), horizon=240.0)
    key = lambda f: sorted(
        (f.records[c].call.arrival_time, f.records[c].call.node) for c in f.sampled_ids
    )
    assert key(a) == key(b)
    c = mid_run.fork(np.random.default_rng(43), horizon=240.0)
    assert key(a) != key(c)


# -------------------------------------------------------------------- pausing


def test_the_pausing_policy_stops_at_a_decision_epoch(mid_run):
    fork = mid_run.fork(np.random.default_rng(0), horizon=480.0)
    with pytest.raises(_Pause) as caught:
        fork.resume(_PausingPolicy(DMEXCLP()), until=mid_run.time + 480.0)
    amb = fork.ambulances[caught.value.amb_id]
    # The unit must be left genuinely awaiting a station, not mid-mission.
    assert amb.status is Status.IDLE
    assert amb.route == [] and amb.call is None


# ------------------------------------------------------------------- policies


@pytest.mark.parametrize(
    "policy",
    [
        RolloutPolicy(n_sims=2, horizon=120.0, n_candidates=2),
        MCTSPolicy(n_sims=4, horizon=120.0, depth=2, n_candidates=2),
    ],
    ids=["rollout", "mcts"],
)
def test_lookahead_policies_only_choose_real_stations(city, policy):
    calls = generate_calls(city, 12 * 60, np.random.default_rng(5))
    sim = Simulator(city, calls, n_ambulances=4, seed=5)
    records = sim.run(policy)
    assert all(r.reached_at is not None for r in records)
    assert all(a.destination in city.bases or a.node in city.bases
               for a in sim.ambulances)
    assert policy.sims_run > 0


@pytest.mark.parametrize(
    "policy_cls", [RolloutPolicy, MCTSPolicy], ids=["rollout", "mcts"]
)
def test_a_single_candidate_spends_no_budget(mid_run, policy_cls):
    """With nothing to compare, both must fall straight through to the base policy."""
    policy = policy_cls(n_sims=8, horizon=120.0, n_candidates=1)
    choice = policy.reposition(mid_run, 0)
    assert policy.sims_run == 0
    assert choice == DMEXCLP().reposition(mid_run, 0)


def test_reset_makes_a_lookahead_policy_reproducible(city):
    calls = generate_calls(city, 8 * 60, np.random.default_rng(6))
    policy = RolloutPolicy(n_sims=2, horizon=120.0, n_candidates=2, seed=11)
    out = []
    for _ in range(2):
        policy.reset()
        sim = Simulator(city, calls, n_ambulances=4, seed=6)
        out.append([r.response_time for r in sim.run(policy)])
    assert out[0] == out[1]


def test_rollout_rejects_an_unknown_value_mode():
    with pytest.raises(ValueError):
        RolloutPolicy(value_mode="vibes")
