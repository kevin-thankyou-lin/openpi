import json
import collections

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.libero.speed_baselines.actions import SupActionComposer, sail_precision_slices, uniform_slices
from examples.libero.speed_baselines.result_utils import atomic_write_json, summarize_episodes, summarize_tasks
from examples.libero.speed_baselines.telemetry import record_from_obs, write_episode
from examples.libero.speed_baselines.strider_client import (
    causal_history,
    load_candidate_schedule,
    pop_strider_slice,
    proprio_from_obs,
    validate_strider_server_metadata,
)


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


def test_task_summaries_are_episode_weighted_and_sorted():
    episodes = [
        {
            "task_id": 3,
            "task_description": "task three",
            "success": True,
            "env_steps": 10,
            "source_actions_consumed": 20,
        },
        {
            "task_id": 1,
            "task_description": "task one",
            "success": False,
            "env_steps": 40,
            "source_actions_consumed": 80,
        },
        {
            "task_id": 3,
            "task_description": "task three",
            "success": False,
            "env_steps": 30,
            "source_actions_consumed": 60,
        },
        {
            "task_id": 1,
            "task_description": "task one",
            "success": True,
            "env_steps": 20,
            "source_actions_consumed": 40,
        },
    ]

    assert summarize_tasks(episodes) == [
        {
            "task_id": 1,
            "task_description": "task one",
            "total_episodes": 2,
            "total_successes": 1,
            "success_rate": 0.5,
            "mean_success_env_steps": 20,
            "mean_success_source_actions": 40,
        },
        {
            "task_id": 3,
            "task_description": "task three",
            "total_episodes": 2,
            "total_successes": 1,
            "success_rate": 0.5,
            "mean_success_env_steps": 10,
            "mean_success_source_actions": 20,
        },
    ]


def test_task_summaries_reject_inconsistent_descriptions():
    with pytest.raises(ValueError, match="inconsistent descriptions"):
        summarize_tasks(
            [
                {
                    "task_id": 0,
                    "task_description": "first",
                    "success": True,
                    "env_steps": 1,
                    "source_actions_consumed": 1,
                },
                {
                    "task_id": 0,
                    "task_description": "second",
                    "success": True,
                    "env_steps": 1,
                    "source_actions_consumed": 1,
                },
            ]
        )


def test_atomic_json_replaces_complete_payload(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"status": "running"})
    atomic_write_json(path, {"status": "complete", "count": 1})
    assert json.loads(path.read_text()) == {"status": "complete", "count": 1}


def test_strider_telemetry_record_is_robot_only_and_copied():
    obs = {
        "robot0_joint_pos": np.arange(7, dtype=np.float64),
        "robot0_gripper_qpos": np.array([-0.02, 0.02]),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "object_state": np.array([99.0]),
    }
    action = np.arange(7, dtype=np.float64)
    record = record_from_obs(obs, action=action, env_step=3, source_stride=2)
    assert set(record) == {
        "joint_pos",
        "gripper_pos",
        "eef_pos",
        "eef_quat",
        "proprio",
        "action",
        "env_step",
        "source_stride",
    }
    np.testing.assert_allclose(
        record["proprio"], np.concatenate([np.arange(7, dtype=np.float32), [-0.02, 0.02]])
    )
    assert record["env_step"] == 3
    assert record["source_stride"] == 2
    obs["robot0_joint_pos"][0] = 100.0
    assert record["joint_pos"][0] == 0.0


def test_strider_telemetry_episode_is_atomic_and_hash_pinned(tmp_path):
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"not-a-real-video-but-nonempty")
    records = []
    for step in range(2):
        records.append(
            record_from_obs(
                {
                    "robot0_joint_pos": np.arange(7) + step,
                    "robot0_gripper_qpos": np.array([-0.02, 0.02]),
                    "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
                    "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                },
                action=np.zeros(7),
                env_step=step,
                source_stride=1,
            )
        )
    episode = write_episode(
        tmp_path / "telemetry" / "task_00_episode_00",
        video_path=video,
        records=records,
        metadata={"task_id": 0, "episode_index": 0, "success": True},
    )
    metadata = json.loads((episode / "metadata.json").read_text())
    assert metadata["schema"] == "openpi_libero_strider_telemetry.v1"
    assert metadata["frame_count"] == 2
    assert metadata["array_shapes"]["proprio.npy"] == [2, 9]
    assert len(metadata["sha256"]) == 9
    assert (episode / "top_camera-images-rgb.mp4").stat().st_ino == video.stat().st_ino
    assert not (episode.parent / ".task_00_episode_00.partial").exists()


def test_strider_causal_history_is_left_padded_and_strided():
    values = [np.full(9, index, dtype=np.float32) for index in range(4)]
    history = causal_history(values, history=4, stride=2)
    np.testing.assert_array_equal(history[:, 0], [0, 0, 1, 3])


def test_strider_proprio_uses_only_nine_robot_features():
    obs = {
        "robot0_joint_pos": np.arange(7, dtype=np.float32),
        "robot0_gripper_qpos": np.array([-0.1, 0.1], dtype=np.float32),
        "object_state": np.array([99.0], dtype=np.float32),
    }
    np.testing.assert_array_equal(
        proprio_from_obs(obs),
        np.concatenate(
            [np.arange(7, dtype=np.float32), np.array([-0.1, 0.1], dtype=np.float32)]
        ),
    )


def test_strider_candidate_schedule_is_exact_and_non_authoritative(tmp_path):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema": "strider-libero-subtask-candidate-v1",
                "review_gate": "non-authoritative until every boundary is visually reviewed",
                "tasks": {"2": {"subtasks": [["approach", 2], ["contact", 1]]}},
            }
        )
    )
    speeds, task_ids = load_candidate_schedule(
        schedule,
        checkpoint_phases=("task_02:approach", "task_02:contact"),
        fast_stride=2,
    )
    assert speeds == {"task_02:approach": 2, "task_02:contact": 1}
    assert task_ids == {2}


def test_strider_candidate_schedule_accepts_registered_stride_three(tmp_path):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema": "strider-libero-subtask-candidate-v1",
                "review_gate": "non-authoritative until every boundary is visually reviewed",
                "tasks": {
                    "2": {"subtasks": [["approach", 3], ["contact", 1]]},
                    "3": {"subtasks": [["approach", 2], ["contact", 1]]},
                },
            }
        )
    )
    speeds, task_ids = load_candidate_schedule(
        schedule,
        checkpoint_phases=(
            "task_02:approach",
            "task_02:contact",
            "task_03:approach",
            "task_03:contact",
        ),
        fast_stride=3,
    )
    assert speeds == {
        "task_02:approach": 3,
        "task_02:contact": 1,
        "task_03:approach": 2,
        "task_03:contact": 1,
    }
    assert task_ids == {2, 3}


def test_strider_action_consumption_respects_speed_and_odd_tail(composer):
    plan = collections.deque(np.full(7, value, dtype=np.float64) for value in (0.1, 0.2, 0.3))
    first = pop_strider_slice(plan, stride=2, composer=composer)
    second = pop_strider_slice(plan, stride=2, composer=composer)
    assert first.stride == 2
    assert second.stride == 1
    assert len(plan) == 0
    np.testing.assert_allclose(first.action[:3], [0.3, 0.3, 0.3])
    np.testing.assert_allclose(second.action, np.full(7, 0.3))


def test_strider_action_consumption_supports_stride_three_and_tail(composer):
    plan = collections.deque(np.full(7, value, dtype=np.float64) for value in (0.1, 0.2, 0.3, 0.4))
    first = pop_strider_slice(plan, stride=3, composer=composer)
    second = pop_strider_slice(plan, stride=3, composer=composer)
    assert first.stride == 3
    assert second.stride == 1
    assert len(plan) == 0
    np.testing.assert_allclose(first.action[:3], [0.6, 0.6, 0.6])
    np.testing.assert_allclose(second.action, np.full(7, 0.4))


def test_strider_server_metadata_is_exact_and_candidate_only():
    metadata = {
        "method": "strider_phase_candidate",
        "base_policy": "pi05_libero",
        "base_checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "phase_checkpoint_sha256": "checkpoint",
        "phase_schedule_sha256": "schedule",
        "evaluator_commit": "commit",
        "fast_stride": 2,
        "authority": "AI_CANDIDATE_NOT_HUMAN_ANNOTATION",
        "requires_human_confirmation": True,
        "speed_schedule_status": "candidate_not_promoted",
    }
    validate_strider_server_metadata(
        metadata,
        checkpoint_sha256="checkpoint",
        schedule_sha256="schedule",
        evaluator_commit="commit",
        fast_stride=2,
    )
    metadata["fast_stride"] = 1
    with pytest.raises(ValueError, match="fast_stride"):
        validate_strider_server_metadata(
            metadata,
            checkpoint_sha256="checkpoint",
            schedule_sha256="schedule",
            evaluator_commit="commit",
            fast_stride=2,
        )
