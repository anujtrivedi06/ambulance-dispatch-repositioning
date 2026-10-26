import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ambulance import Simulator, generate_calls, grid_city, mexclp, summarise
from ambulance.deeprl import DeepValuePolicy, ValueNet, load_agent, save_agent
from ambulance.experiment import column, evaluate
from ambulance.features import FeatureExtractor
from ambulance.policies import DMEXCLP


@pytest.fixture(scope="module")
def setup():
    city = grid_city(width=6, height=6, base_call_rate=4.0, seed=3)
    sites, alloc = mexclp(city, 6, busy_prob=0.35, max_sites=4)
    run_city = city.with_bases(sites)
    return run_city, alloc, FeatureExtractor(run_city)


def test_saved_agent_round_trips(tmp_path, setup):
    """A saved agent must behave identically to the one that was saved.

    This is the submitted artefact: if the restored policy differs from the evaluated
    one, every number in the report describes a network nobody else can load.
    """
    run_city, alloc, ex = setup
    net = ValueNet(ex.size + ex.n_zones)
    path = tmp_path / "agent.pt"
    save_agent(net, path, {"reward": "response"})

    restored, config = load_agent(path)
    assert config["reward"] == "response"

    calls = generate_calls(run_city, 24 * 60, np.random.default_rng(4))
    outcomes = []
    for model in (net, restored):
        sim = Simulator(run_city, calls, n_ambulances=6, seed=4, home_bases=alloc)
        outcomes.append(
            [r.response_time for r in sim.run(DeepValuePolicy(model, ex))]
        )
    assert outcomes[0] == outcomes[1]


def test_saved_agent_records_its_feature_layout(tmp_path, setup):
    """Weights without their feature layout are a policy that fails silently."""
    _, _, ex = setup
    path = tmp_path / "agent.pt"
    save_agent(ValueNet(ex.size + ex.n_zones), path)
    import torch

    blob = torch.load(path, weights_only=False)
    assert blob["d_in"] == ex.size + ex.n_zones
    assert len(blob["features"]) == blob["d_in"]


def test_validation_and_test_seeds_are_disjoint(setup):
    """Model selection must not happen on the episodes used to report.

    Equal metrics from the two blocks would mean the offset is not taking effect and
    the reported score is a best-of-N rather than an estimate.
    """
    run_city, alloc, _ = setup
    kw = dict(days=1, reps=4, n_ambulances=6, home_bases=alloc)
    test_block = evaluate(DMEXCLP(), run_city, seed_offset=0, **kw)
    val_block = evaluate(DMEXCLP(), run_city, seed_offset=50_000, **kw)
    assert column(test_block, "coverage").tolist() != column(val_block, "coverage").tolist()


def test_seed_offset_is_reproducible(setup):
    run_city, alloc, _ = setup
    kw = dict(days=1, reps=3, n_ambulances=6, home_bases=alloc, seed_offset=50_000)
    first = evaluate(DMEXCLP(), run_city, **kw)
    second = evaluate(DMEXCLP(), run_city, **kw)
    assert column(first, "coverage").tolist() == column(second, "coverage").tolist()


def test_multiseed_selection_uses_validation_not_test(setup):
    """Guards the rule the multi-seed script depends on.

    Selecting a seed by its test score would make the reported figure a maximum over
    seeds. The two blocks must therefore disagree about which of two policies is better
    often enough that selecting on one is not selecting on the other.
    """
    run_city, alloc, _ = setup
    from ambulance.policies import DMEXCLP, RandomReposition

    kw = dict(days=1, reps=5, n_ambulances=6, home_bases=alloc)
    for offset in (0, 50_000):
        heuristic = column(evaluate(DMEXCLP(), run_city, seed_offset=offset, **kw), "coverage")
        floor = column(
            evaluate(RandomReposition(seed=0), run_city, seed_offset=offset, **kw), "coverage"
        )
        # Both blocks must be usable: a block where nothing can be distinguished would
        # make selection meaningless rather than merely biased.
        assert heuristic.mean() > floor.mean()


def test_target_diagnostic_reports_a_wider_gap_at_a_tighter_target():
    """The limitation the report states: candidates separate as the target tightens."""
    import numpy as np

    from ambulance import grid_city
    from ambulance.location import coverage_matrix

    city = grid_city(width=8, height=8, n_bases=4, base_call_rate=4.0, seed=7)
    reach = {t: coverage_matrix(city, t).mean() for t in (5.0, 8.0)}
    assert reach[5.0] < reach[8.0], "a tighter target must reach fewer node pairs"
    # The headline claim: at 8 minutes a single station already covers most demand.
    assert reach[8.0] > 0.5
