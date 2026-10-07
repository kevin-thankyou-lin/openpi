from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Any, Callable

import numpy as np

from .actions import ActionSlice, SupActionComposer


STRIDER_AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"
STRIDER_SCHEDULE_SCHEMA = "strider-libero-subtask-candidate-v1"
STRIDER_FINE_SCHEDULE_SCHEMA = "strider-libero-task2-fine-phase-schedule-v1"
STRIDER_SERVER_METHOD = "strider_phase_candidate"


def causal_history(values: list[np.ndarray], *, history: int, stride: int) -> np.ndarray:
    if not values:
        raise ValueError("causal history requires at least one proprioception sample")
    if history < 1 or stride < 1:
        raise ValueError("history and stride must be positive")
    stacked = np.asarray(values, dtype=np.float32)
    offsets = np.arange(history - 1, -1, -1) * stride
    indices = np.maximum(len(stacked) - 1 - offsets, 0)
    return stacked[indices]


def proprio_from_obs(obs: dict[str, Any]) -> np.ndarray:
    values = []
    for key in ("robot0_joint_pos", "robot0_gripper_qpos"):
        if key not in obs:
            raise KeyError(f"LIBERO observation missing Strider feature {key!r}")
        value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite values, got shape {value.shape}")
        values.append(value)
    result = np.concatenate(values)
    if result.shape != (9,):
        raise ValueError(f"Strider checkpoint requires nine proprioception features, got {result.shape}")
    return result


def load_candidate_schedule(
    path: pathlib.Path, *, checkpoint_phases: tuple[str, ...], fast_stride: int
) -> tuple[dict[str, int], frozenset[int]]:
    path = pathlib.Path(path)
    payload = json.loads(path.read_text())
    schema = payload.get("schema")
    speeds: dict[str, int] = {}
    task_ids = set()
    entries: list[tuple[str, int]] = []
    if schema == STRIDER_SCHEDULE_SCHEMA:
        if payload.get("review_gate") != "non-authoritative until every boundary is visually reviewed":
            raise ValueError("Strider candidate schedule is missing its non-authoritative review gate")
        for task_text, task in payload.get("tasks", {}).items():
            task_id = int(task_text)
            task_ids.add(task_id)
            entries.extend(
                (f"task_{task_id:02d}:{phase_name}", speed)
                for phase_name, speed in task.get("subtasks", [])
            )
    elif schema == STRIDER_FINE_SCHEDULE_SCHEMA:
        if payload.get("authority") != STRIDER_AUTHORITY or payload.get("requires_human_confirmation") is not True:
            raise ValueError("fine-phase schedule is missing its candidate-only authority gate")
        if payload.get("status") != "AI_CANDIDATE_NOT_PROMOTED_OR_EVALUATED":
            raise ValueError("fine-phase schedule must remain unpromoted before evaluation")
        entries = payload.get("phase_schedule", [])
        if not entries:
            raise ValueError("fine-phase schedule has no registered phases")
    else:
        raise ValueError(f"unsupported Strider schedule schema: {schema!r}")

    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"invalid Strider phase entry: {entry!r}")
        key, speed = entry
        key = str(key)
        if not key.startswith("task_") or ":" not in key:
            raise ValueError(f"invalid Strider phase key: {key!r}")
        try:
            task_ids.add(int(key.split(":", 1)[0][5:]))
        except ValueError as exc:
            raise ValueError(f"invalid Strider phase key: {key!r}") from exc
        speed = int(speed)
        if speed < 1 or speed > fast_stride:
            raise ValueError(f"{key} uses unsupported candidate speed {speed}")
        if key in speeds:
            raise ValueError(f"duplicate Strider candidate phase {key}")
        speeds[key] = speed
    expected = set(checkpoint_phases)
    if set(speeds) != expected:
        missing = sorted(expected - set(speeds))
        extra = sorted(set(speeds) - expected)
        raise ValueError(f"checkpoint/schedule phase mismatch: missing={missing} extra={extra}")
    return speeds, frozenset(task_ids)


def pop_strider_slice(
    action_plan: collections.deque[np.ndarray], *, stride: int, composer: SupActionComposer
) -> ActionSlice:
    if not action_plan:
        raise ValueError("cannot consume an empty Strider action plan")
    if stride < 1:
        raise ValueError("Strider stride must be positive")
    count = min(stride, len(action_plan))
    actions = np.stack([np.asarray(action_plan.popleft(), dtype=np.float64) for _ in range(count)])
    action = actions[0].copy() if count == 1 else composer.merge(actions)
    return ActionSlice(action=action, source_indices=tuple(range(count)))


def validate_strider_server_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_sha256: str,
    schedule_sha256: str,
    evaluator_commit: str,
    fast_stride: int,
) -> None:
    expected = {
        "method": STRIDER_SERVER_METHOD,
        "base_policy": "pi05_libero",
        "base_checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "phase_checkpoint_sha256": checkpoint_sha256,
        "phase_schedule_sha256": schedule_sha256,
        "evaluator_commit": evaluator_commit,
        "fast_stride": fast_stride,
        "authority": STRIDER_AUTHORITY,
        "requires_human_confirmation": True,
        "speed_schedule_status": "candidate_not_promoted",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Strider server metadata mismatch: {mismatches}")


class StriderPhaseSelector:
    """Task-ID-free phase inference with a frozen candidate phase-to-speed table."""

    def __init__(
        self,
        *,
        checkpoint: pathlib.Path,
        phase_repo: pathlib.Path,
        schedule: pathlib.Path,
        fast_stride: int,
        device: str,
        predictor_factory: Callable[..., Any] | None = None,
    ) -> None:
        checkpoint = pathlib.Path(checkpoint)
        phase_repo = pathlib.Path(phase_repo)
        schedule = pathlib.Path(schedule)
        if predictor_factory is None:
            sys.path.insert(0, str(phase_repo))
            from phase_detector.rgb_inference import RGBPhasePredictor

            predictor_factory = RGBPhasePredictor
        self.predictor = predictor_factory(checkpoint, device=device)
        self.phase_speeds, self.affected_task_ids = load_candidate_schedule(
            schedule,
            checkpoint_phases=tuple(self.predictor.phases),
            fast_stride=fast_stride,
        )
        self.fast_stride = fast_stride
        self._proprio: list[np.ndarray] = []

    def reset_episode(self) -> None:
        self._proprio.clear()

    def select(self, image: np.ndarray, obs: dict[str, Any], *, task_id: int) -> tuple[int, dict[str, Any]]:
        if task_id not in self.affected_task_ids:
            return self.fast_stride, {
                "phase": None,
                "phase_confidence": None,
                "phase_task_match": None,
                "schedule_source": "registered_uniform_2x_unaffected_task",
            }
        self._proprio.append(proprio_from_obs(obs))
        history = causal_history(
            self._proprio,
            history=int(self.predictor.history),
            stride=int(self.predictor.inference_history_stride),
        )
        prediction = self.predictor.predict(np.asarray(image, dtype=np.uint8), history)
        phase = str(prediction["phase"])
        if phase not in self.phase_speeds:
            raise ValueError(f"Strider predictor returned unregistered phase {phase!r}")
        probability = np.asarray(prediction["probability"], dtype=np.float64)
        return self.phase_speeds[phase], {
            "phase": phase,
            "phase_confidence": float(np.max(probability)),
            "phase_task_match": phase.startswith(f"task_{task_id:02d}:"),
            "schedule_source": "ai_candidate_phase_schedule_not_promoted",
        }
