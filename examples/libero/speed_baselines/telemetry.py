"""Frame-aligned, robot-only telemetry for Strider phase modeling."""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Any

import numpy as np

from .result_utils import atomic_write_json


SCHEMA = "openpi_libero_strider_telemetry.v1"


def record_from_obs(
    obs: dict[str, Any], *, action: np.ndarray, env_step: int, source_stride: int
) -> dict[str, np.ndarray | int]:
    """Copy the pre-action observation and applied controller action for one frame."""
    joint_pos = _finite_vector(obs, "robot0_joint_pos")
    gripper_pos = _finite_vector(obs, "robot0_gripper_qpos")
    eef_pos = _finite_vector(obs, "robot0_eef_pos")
    eef_quat = _finite_vector(obs, "robot0_eef_quat")
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 7 or not np.isfinite(action).all():
        raise ValueError(f"action must contain seven finite values, got shape {action.shape}")
    if env_step < 0 or source_stride < 1:
        raise ValueError("env_step must be nonnegative and source_stride must be positive")
    return {
        "joint_pos": joint_pos,
        "gripper_pos": gripper_pos,
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "proprio": np.concatenate([joint_pos, gripper_pos]),
        "action": action.copy(),
        "env_step": int(env_step),
        "source_stride": int(source_stride),
    }


def write_episode(
    episode_dir: pathlib.Path,
    *,
    video_path: pathlib.Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> pathlib.Path:
    """Atomically publish one aligned episode without duplicating its MP4 bytes."""
    episode_dir = pathlib.Path(episode_dir)
    video_path = pathlib.Path(video_path).resolve()
    if episode_dir.exists():
        raise FileExistsError(f"Strider telemetry episode already exists: {episode_dir}")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"completed rollout video is missing or empty: {video_path}")
    if not records:
        raise ValueError("cannot publish an empty Strider telemetry episode")

    partial = episode_dir.with_name(f".{episode_dir.name}.partial")
    if partial.exists():
        raise FileExistsError(f"partial Strider telemetry episode requires review: {partial}")
    partial.mkdir(parents=True)

    arrays = {
        "robot0-joint_pos.npy": _stack(records, "joint_pos", np.float32),
        "robot0-gripper_pos.npy": _stack(records, "gripper_pos", np.float32),
        "robot0-eef_pos.npy": _stack(records, "eef_pos", np.float32),
        "robot0-eef_quat.npy": _stack(records, "eef_quat", np.float32),
        "proprio.npy": _stack(records, "proprio", np.float32),
        "action.npy": _stack(records, "action", np.float32),
        "env_step.npy": np.asarray([row["env_step"] for row in records], dtype=np.int64),
        "source_stride.npy": np.asarray([row["source_stride"] for row in records], dtype=np.int64),
    }
    for name, value in arrays.items():
        np.save(partial / name, value, allow_pickle=False)

    linked_video = partial / "top_camera-images-rgb.mp4"
    os.link(video_path, linked_video)
    joint_width = arrays["robot0-joint_pos.npy"].shape[1]
    gripper_width = arrays["robot0-gripper_pos.npy"].shape[1]
    payload = {
        "schema": SCHEMA,
        **metadata,
        "frame_count": len(records),
        "fps": 10,
        "alignment": "pre-action observation, rendered frame, and applied controller action at each env step",
        "proprio_feature_names": [f"robot0.joint.{index}" for index in range(joint_width)]
        + [f"robot0.gripper.{index}" for index in range(gripper_width)],
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "sha256": {name: _sha256(partial / name) for name in [*arrays, "top_camera-images-rgb.mp4"]},
        "source_video": str(video_path),
    }
    atomic_write_json(partial / "metadata.json", payload)
    partial.replace(episode_dir)
    return episode_dir


def _finite_vector(obs: dict[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"LIBERO observation missing required Strider telemetry key {key!r}")
    value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain finite values, got shape {value.shape}")
    return value.copy()


def _stack(records: list[dict[str, Any]], key: str, dtype: np.dtype) -> np.ndarray:
    values = [np.asarray(row[key], dtype=dtype) for row in records]
    try:
        return np.stack(values)
    except ValueError as error:
        raise ValueError(f"inconsistent {key} shapes across telemetry frames") from error


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
