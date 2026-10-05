import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.libero.speed_baselines.actions import SupActionComposer, sail_precision_slices, uniform_slices
from examples.libero.speed_baselines.result_utils import atomic_write_json, summarize_episodes


@pytest.fixture
def composer():
    return SupActionComposer(
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.ones(6),
        output_max=np.ones(6),
    )


def test_sup_merge_sums_translation_left_composes_rotation_and_keeps_last_gripper(composer):
    actions = np.array(
        [
            [0.1, 0.2, 0.3, 0.2, 0.0, 0.0, -1.0],
            [0.4, -0.1, 0.2, 0.0, 0.3, 0.0, 1.0],
        ]
    )
    merged = composer.merge(actions)
    expected_rotation = Rotation.from_rotvec(actions[1, 3:6]) * Rotation.from_rotvec(actions[0, 3:6])
    np.testing.assert_allclose(merged[:3], [0.5, 0.1, 0.5])
    np.testing.assert_allclose(merged[3:6], expected_rotation.as_rotvec())
    assert merged[6] == 1.0


def test_uniform_stride_two_keeps_odd_tail(composer):
    actions = np.zeros((5, 7))
    actions[:, 0] = 0.1
    slices = uniform_slices(actions, stride=2, composer=composer)
    assert [item.source_indices for item in slices] == [(0, 1), (2, 3), (4,)]
    np.testing.assert_allclose([item.action[0] for item in slices], [0.2, 0.2, 0.1])


def test_composer_broadcasts_scalar_input_range():
    composer = SupActionComposer(
        input_min=-1,
        input_max=1,
        output_min=np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
        output_max=np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
    )
    merged = composer.merge(np.zeros((2, 7)))
    np.testing.assert_allclose(merged, np.zeros(7))


def test_sail_never_swallows_critical_second_action(composer):
    actions = np.zeros((4, 7))
    actions[:, 6] = -1.0
    slices = sail_precision_slices(
        actions,
        np.array([0.0, 0.9, 0.0, 0.0]),
        threshold=0.5,
        fast_stride=2,
        composer=composer,
    )
    assert [item.source_indices for item in slices] == [(0,), (1,), (2, 3)]


def test_sail_protects_both_sides_of_gripper_transition(composer):
    actions = np.zeros((4, 7))
    actions[:, 6] = [-1.0, -1.0, 1.0, 1.0]
    slices = sail_precision_slices(
        actions,
        np.zeros(4),
        threshold=0.5,
        fast_stride=2,
        composer=composer,
    )
    assert [item.source_indices for item in slices] == [(0,), (1,), (2,), (3,)]


def test_sail_rejects_nan_scores(composer):
    with pytest.raises(ValueError, match="finite"):
        sail_precision_slices(
            np.zeros((2, 7)),
            np.array([0.0, np.nan]),
            threshold=0.5,
            fast_stride=2,
            composer=composer,
        )


def test_summary_is_episode_weighted():
    summary = summarize_episodes(
        [
            {"success": True, "env_steps": 10, "source_actions_consumed": 20},
            {"success": False, "env_steps": 30, "source_actions_consumed": 60},
            {"success": True, "env_steps": 20, "source_actions_consumed": 40},
        ]
    )
    assert summary == {
        "total_episodes": 3,
        "total_successes": 2,
        "success_rate": 2 / 3,
        "mean_success_env_steps": 15,
        "mean_success_source_actions": 30,
    }


def test_atomic_json_replaces_complete_payload(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"status": "running"})
    atomic_write_json(path, {"status": "complete", "count": 1})
    assert json.loads(path.read_text()) == {"status": "complete", "count": 1}
