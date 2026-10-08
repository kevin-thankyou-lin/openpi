from __future__ import annotations

import numpy as np
import pytest

from examples.libero.speed_baselines.prepare_task_subtasks import _boundaries
from examples.libero.speed_baselines.prepare_task_subtasks import _labels
from examples.libero.speed_baselines.prepare_task_subtasks import _split


def _commands(*runs: tuple[int, float]) -> np.ndarray:
    return np.concatenate([np.full(length, value, dtype=np.float32) for length, value in runs])


def test_two_object_boundaries_skip_short_retry_and_use_final_grasp() -> None:
    commands = _commands(
        (40, -1),
        (4, 1),
        (20, -1),
        (45, 1),
        (60, -1),
        (15, 1),
        (12, -1),
        (50, 1),
    )
    assert _boundaries(commands, "two_object") == (64, 109, 196)


def test_two_object_boundaries_debounce_command_chatter() -> None:
    commands = _commands(
        (50, -1),
        (70, 1),
        (4, -1),
        (6, 1),
        (60, -1),
        (55, 1),
    )
    assert _boundaries(commands, "two_object") == (50, 130, 190)


def test_one_object_then_close_uses_sustained_release() -> None:
    commands = _commands((30, -1), (40, 1), (3, -1), (4, 1), (25, -1))
    assert _boundaries(commands, "one_object_then_close") == (30, 77)


def test_labels_cover_every_frame_with_strict_phases() -> None:
    labels = _labels(12, (3, 8))
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="strictly ordered"):
        _labels(12, (3, 3))


def test_split_is_deterministic_and_disjoint() -> None:
    episode_ids = [f"task_00_episode_{index:02d}" for index in range(20)]
    first = _split(0, episode_ids, 1)
    second = _split(0, list(reversed(episode_ids)), 1)
    assert first == second
    assert list(first.values()).count("train") == 12
    assert list(first.values()).count("validation") == 4
    assert list(first.values()).count("test") == 4
