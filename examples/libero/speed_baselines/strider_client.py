from __future__ import annotations

import collections
from collections.abc import Callable
import importlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

from .actions import ActionSlice
from .actions import SupActionComposer

STRIDER_AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"
STRIDER_SCHEDULE_SCHEMA = "strider-libero-subtask-candidate-v1"
STRIDER_FINE_SCHEDULE_SCHEMA = "strider-libero-task2-fine-phase-schedule-v1"
STRIDER_SERVER_METHOD = "strider_phase_candidate"
STRIDER_ALLOWED_SPEEDS = frozenset({1.0, 1.5, 2.0, 3.0})


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
) -> tuple[dict[str, float], frozenset[int]]:
    path = pathlib.Path(path)
    payload = json.loads(path.read_text())
    schema = payload.get("schema")
    speeds: dict[str, float] = {}
    task_ids = set()
    entries: list[tuple[str, float]] = []
    if schema == STRIDER_SCHEDULE_SCHEMA:
        if payload.get("review_gate") != "non-authoritative until every boundary is visually reviewed":
            raise ValueError("Strider candidate schedule is missing its non-authoritative review gate")
        for task_text, task in payload.get("tasks", {}).items():
            task_id = int(task_text)
            task_ids.add(task_id)
            entries.extend(
                (f"task_{task_id:02d}:{phase_name}", speed) for phase_name, speed in task.get("subtasks", [])
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
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:  # noqa: UP038
            raise ValueError(f"invalid Strider phase entry: {entry!r}")
        key, speed = entry
        key = str(key)
        if not key.startswith("task_") or ":" not in key:
            raise ValueError(f"invalid Strider phase key: {key!r}")
        try:
            task_ids.add(int(key.split(":", 1)[0][5:]))
        except ValueError as exc:
            raise ValueError(f"invalid Strider phase key: {key!r}") from exc
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):  # noqa: UP038
            raise ValueError(f"{key} uses non-numeric candidate speed {speed!r}")
        speed = float(speed)
        if speed not in STRIDER_ALLOWED_SPEEDS or speed > fast_stride:
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
    """Horizon-aligned phase inference with a frozen candidate speed table."""

    def __init__(
        self,
        *,
        checkpoint: pathlib.Path,
        phase_repo: pathlib.Path,
        schedule: pathlib.Path,
        fast_stride: int,
        device: str,
        predictor_factory: Callable[..., Any] | None = None,
        composer: SupActionComposer | None = None,
    ) -> None:
        checkpoint = pathlib.Path(checkpoint)
        phase_repo = pathlib.Path(phase_repo)
        schedule = pathlib.Path(schedule)
        sys.path.insert(0, str(phase_repo / "src"))
        strider = importlib.import_module("strider")
        if predictor_factory is None:
            model_module = importlib.import_module("strider.subtask_model")
            predictor_factory = model_module.TorchObservationHistorySubtaskPredictor.from_checkpoint
        self.predictor = predictor_factory(checkpoint, device=device)
        checkpoint_phases = tuple(f"task_02:{name}" for name in self.predictor.class_names)
        self.phase_speeds, self.affected_task_ids = load_candidate_schedule(
            schedule,
            checkpoint_phases=checkpoint_phases,
            fast_stride=fast_stride,
        )
        if self.affected_task_ids != {2}:
            raise ValueError("horizon Task-2 evaluator requires exactly task 2 in the schedule")
        self.fast_stride = fast_stride
        self._observations: list[dict[str, np.ndarray]] = []
        self._composer = composer or SupActionComposer.from_robosuite()

        phase_speeds = {key.split(":", 1)[1]: float(speed) for key, speed in self.phase_speeds.items()}

        libero_composer = self._composer

        class LiberoActionComposer:
            def compose(self, actions):
                return libero_composer.merge(np.asarray(actions, dtype=np.float64))

        self.runtime = strider.StriderRuntime(
            predictor=self.predictor,
            schedule=strider.SubtaskSpeedSchedule(
                phase_speeds,
                fallback_speed=1.0,
                minimum_confidence=0.0,
            ),
            pipeline=strider.PlanPipeline(
                retimer=strider.BoundaryAwareCadenceRetimer(
                    LiberoActionComposer(),
                    allowed_speeds=tuple(sorted(STRIDER_ALLOWED_SPEEDS)),
                )
            ),
        )

    def reset_episode(self) -> None:
        self._observations.clear()

    def observe(self, image: np.ndarray, obs: dict[str, Any]) -> None:
        self._observations.append({"rgb": np.asarray(image, dtype=np.uint8), "proprio": proprio_from_obs(obs)})
        self._observations = self._observations[-int(self.predictor.history_length) :]

    def plan(self, actions: np.ndarray, *, task_id: int) -> tuple[list[ActionSlice], dict[str, Any]]:
        if task_id != 2:
            raise ValueError(f"horizon Task-2 evaluator received task {task_id}")
        result = self.runtime.plan(
            self._observations,
            tuple(np.asarray(action, dtype=np.float64) for action in actions),
            metadata={
                "task_id": task_id,
                "schedule_source": "horizon_ai_candidate_not_promoted",
            },
        )
        phases = tuple(None if label is None else f"task_02:{label}" for label in result.metadata["subtask_labels"])
        confidence = result.metadata["subtask_confidence"]
        task_match = tuple(phase in self.phase_speeds for phase in phases)
        scheduled = []
        for action, provenance in zip(  # noqa: B905
            result.scheduled_actions,
            result.transformed.scheduled.provenance,
        ):
            indices = tuple(provenance["source_indices"])
            scheduled.append(
                ActionSlice(
                    action=np.asarray(action, dtype=np.float64),
                    source_indices=indices,
                    metadata={
                        "requested_stride": provenance["requested_speed"],
                        "actual_stride": provenance["actual_stride"],
                        "boundary_limited": provenance["boundary_limited"],
                        "truncated_at_horizon": provenance["truncated_at_horizon"],
                        "phases": tuple(phases[index] for index in indices),
                        "phase_confidence": tuple(confidence[index] for index in indices),
                        "phase_task_match": tuple(task_match[index] for index in indices),
                    },
                )
            )
        return scheduled, {
            "selected_k": None,
            "slice_strides": [item.stride for item in scheduled],
            "predicted_phases": list(phases),
            "phase_confidence": list(confidence),
            "phase_task_match": list(task_match),
            "speed_factors": list(result.speed.factors),
            "used_fallback": result.speed.used_fallback,
            "fallback_indices": list(result.metadata["fallback_indices"]),
            "coverage_steps": result.coverage_steps,
            "schedule_source": result.metadata["schedule_source"],
            "pipeline_stages": list(result.transformed.receipt.stage_order),
        }
