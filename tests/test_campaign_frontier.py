import dataclasses

import pytest

from examples.libero.speed_baselines.campaign_frontier import Episode
from examples.libero.speed_baselines.campaign_frontier import compare_frontier

PINS = {
    "suite": "libero_10",
    "task_id": 0,
    "bddl_sha256": "b" * 64,
    "policy_sha256": "a" * 64,
    "controller_sha256": "c" * 64,
    "environment_sha256": "e" * 64,
    "reset_protocol_sha256": "f" * 64,
    "control_hz": 20,
    "chunk_size": 10,
}


def episode(index, steps, *, success=True, wall=1.0):
    return Episode(index, success, steps, *(["a" * 64] * 4), loop_wall_seconds=wall)


def compare(a, b, **kw):
    return compare_frontier(a, b, reference_pins=PINS, candidate_pins=PINS, **kw)


def test_requires_strict_improvement_without_reliability_loss():
    base = [episode(0, 100), episode(1, 100)]
    assert compare(base, base)["status"] == "NO_WIN"
    fast = [episode(0, 90), episode(1, 90)]
    assert compare(base, fast)["status"] == "OBSERVED_PARETO_WIN"
    failed = [episode(0, 50), episode(1, 520, success=False)]
    report = compare(base, failed)
    assert report["status"] == "NO_WIN"
    assert report["candidate"]["mean_success_seconds"] == 2.5
    assert report["candidate"]["spent_seconds_per_success"] == 28.5


def test_failure_cost_can_veto_faster_successful_subset():
    a = [episode(0, 100), episode(1, 50, success=False)]
    b = [episode(0, 90), episode(1, 520, success=False)]
    assert compare(a, b)["status"] == "NO_WIN"


def test_equal_success_counts_allow_swapped_failures_but_report_them():
    a = [episode(0, 100), episode(1, 520, success=False)]
    b = [episode(0, 520, success=False), episode(1, 90)]
    report = compare(a, b)
    assert report["status"] == "OBSERVED_PARETO_WIN"
    assert report["candidate_only_successes"] == [1]
    assert report["reference_only_successes"] == [0]
    assert report["shared_successes"] == 0


def test_wall_time_is_preregistered_not_silently_ignored():
    a = [episode(0, 100, wall=1)]
    b = [episode(0, 90, wall=2)]
    assert compare(a, b)["status"] == "OBSERVED_PARETO_WIN"
    assert compare(a, b, require_loop_wall=True)["status"] == "NO_WIN"
    with pytest.raises(ValueError, match="every episode"):
        compare(a, [episode(0, 90, wall=None)], require_loop_wall=True)


def test_missing_successful_times_cannot_manufacture_win():
    assert compare([episode(0, 520, success=False)], [episode(0, 500, success=False)])["status"] == "NO_WIN"


@pytest.mark.parametrize(
    "field", ["initial_observation_sha256", "first_request_sha256", "first_actions_sha256", "first_rng_sha256"]
)
def test_episode_id_alone_never_establishes_matched_state(field):
    a = episode(0, 100)
    with pytest.raises(ValueError, match="mismatch"):
        compare([a], [dataclasses.replace(a, **{field: "b" * 64})])


def test_order_duplicates_and_suite_collision_rejected():
    a = [episode(0, 100), episode(1, 100)]
    with pytest.raises(ValueError, match="ordered"):
        compare(a, list(reversed(a)))
    with pytest.raises(ValueError, match="Duplicate"):
        compare([a[0], a[0]], [a[0], a[0]])
    with pytest.raises(ValueError, match="suite"):
        compare_frontier(a, a, reference_pins=PINS, candidate_pins=dict(PINS, suite="libero_spatial"))
