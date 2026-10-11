from __future__ import annotations

import collections
from collections.abc import Callable
import hashlib
import importlib
import io
import json
import pathlib
import sys
from typing import Any

import numpy as np

from .actions import ActionSlice
from .actions import SupActionComposer
from .phase_preprocessing import PREPROCESSING
from .phase_preprocessing import bind_phase_inputs

STRIDER_AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"
STRIDER_SCHEDULE_SCHEMA = "strider-libero-subtask-candidate-v1"
STRIDER_FINE_SCHEDULE_SCHEMA = "strider-libero-task2-fine-phase-schedule-v1"
STRIDER_TASK_FINE_SCHEDULE_SCHEMA = "strider-libero-task-fine-phase-schedule-v1"
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
    elif schema in (STRIDER_FINE_SCHEDULE_SCHEMA, STRIDER_TASK_FINE_SCHEDULE_SCHEMA):
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
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
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
        phase_input_binding: dict[str, Any] | None = None,
        legacy_input_mode: bool = False,
    ) -> None:
        checkpoint = pathlib.Path(checkpoint)
        phase_repo = pathlib.Path(phase_repo)
        schedule = pathlib.Path(schedule)
        if (phase_input_binding is None) == (not legacy_input_mode):
            raise ValueError("choose checked phase inputs or explicit historical legacy_input_mode")
        self.input_contract = None
        self.phase_input_proof = None
        sys.path.insert(0, str(phase_repo / "src"))
        strider = importlib.import_module("strider")
        if phase_input_binding is not None:
            if predictor_factory is not None:
                raise ValueError("checked phase inputs cannot override predictor loading")
            model_module = importlib.import_module("strider.subtask_model")
            source_root = (phase_repo / "src").resolve()
            for name, module in tuple(sys.modules.items()):
                if name != "strider" and not name.startswith("strider."):
                    continue
                if not pathlib.Path(module.__file__).resolve().is_relative_to(source_root):
                    raise ValueError("imported STRIDER module differs from pinned phase source root")
            # Deserialize exactly the bytes whose SHA was checked. Never reopen
            # the checkpoint through from_checkpoint after validating its metadata.
            raw = checkpoint.read_bytes()
            if hashlib.sha256(raw).hexdigest() != phase_input_binding["expected_checkpoint_sha256"]:
                raise ValueError("phase checkpoint SHA mismatch before deserialization")
            torch = importlib.import_module("torch")

            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            self.input_contract = bind_phase_inputs(
                checkpoint_path=checkpoint,
                checkpoint_bytes=raw,
                checkpoint_payload=payload,
                **phase_input_binding,
            )
            config = model_module.SubtaskModelConfig(**payload["model_config"])
            model = model_module.ObservationHistorySubtaskModel(config)
            model.load_state_dict(payload["model_state_dict"])
            normalization = payload["normalization"]
            self.predictor = model_module.TorchObservationHistorySubtaskPredictor(
                model=model,
                class_names=payload["class_names"],
                image_mean=normalization["image_mean"],
                image_std=normalization["image_std"],
                proprio_mean=normalization["proprio_mean"],
                proprio_std=normalization["proprio_std"],
                device=device,
                checkpoint_metadata=payload["metadata"],
            )
            self.phase_input_proof = {
                "schema": "strider-verified-phase-inputs-v1",
                "dataset_manifest_sha256": self.input_contract.dataset_manifest_sha256,
                "checkpoint_sha256": self.input_contract.checkpoint_sha256,
                "identity": phase_input_binding["identity"],
                "class_names": list(self.input_contract.class_names),
                "history_length": self.input_contract.history_length,
                "horizon": self.input_contract.horizon,
                "preprocessing": PREPROCESSING,
                "checkpoint_load": "same SHA-verified in-memory bytes; constructed from verified payload",
            }
        else:
            if predictor_factory is None:
                model_module = importlib.import_module("strider.subtask_model")
                predictor_factory = model_module.TorchObservationHistorySubtaskPredictor.from_checkpoint
            self.predictor = predictor_factory(checkpoint, device=device)
        schedule_payload = json.loads(schedule.read_text())
        phase_entries = schedule_payload.get("phase_schedule", [])
        task_ids = {
            int(str(entry[0]).split(":", 1)[0][5:])
            for entry in phase_entries
            if isinstance(entry, (list, tuple)) and len(entry) == 2
        }
        if len(task_ids) != 1:
            raise ValueError("horizon evaluator requires exactly one task in the schedule")
        self.task_id = next(iter(task_ids))
        if phase_input_binding is not None and self.task_id != phase_input_binding["identity"]["task_id"]:
            raise ValueError("schedule task differs from verified phase dataset")
        checkpoint_phases = tuple(f"task_{self.task_id:02d}:{name}" for name in self.predictor.class_names)
        self.phase_speeds, self.affected_task_ids = load_candidate_schedule(
            schedule,
            checkpoint_phases=checkpoint_phases,
            fast_stride=fast_stride,
        )
        if self.affected_task_ids != {self.task_id}:
            raise ValueError("horizon evaluator schedule task identity is inconsistent")
        if phase_input_binding is not None and any(v not in (1, 2, 3) for v in self.phase_speeds.values()):
            raise ValueError("checked campaign phase speeds must be integers 1, 2 or 3")
        self.fast_stride = fast_stride
        self._observations: list[dict[str, np.ndarray]] = []
        self._composer = composer or SupActionComposer.from_robosuite()

        phase_speeds = {key.split(":", 1)[1]: float(speed) for key, speed in self.phase_speeds.items()}

        libero_composer = self._composer

        class LiberoActionComposer:
            def compose(self, actions):
                return libero_composer.merge(np.asarray(actions, dtype=np.float64))

        retimer = (
            strider.BoundaryAwareStrideRetimer(LiberoActionComposer(), allowed_speeds=(1, 2, 3))
            if self.input_contract is not None
            else strider.BoundaryAwareCadenceRetimer(
                LiberoActionComposer(), allowed_speeds=tuple(sorted(STRIDER_ALLOWED_SPEEDS))
            )
        )
        self.runtime = strider.StriderRuntime(
            predictor=self.predictor,
            schedule=strider.SubtaskSpeedSchedule(
                phase_speeds,
                fallback_speed=1.0,
                minimum_confidence=0.0,
            ),
            pipeline=strider.PlanPipeline(retimer=retimer),
        )

    def reset_episode(self) -> None:
        self._observations.clear()

    def observe(self, image: np.ndarray, obs: dict[str, Any]) -> None:
        phase_image = (
            self.input_contract.image(image) if self.input_contract is not None else np.asarray(image, dtype=np.uint8)
        )
        self._observations.append({"rgb": phase_image, "proprio": proprio_from_obs(obs)})
        self._observations = self._observations[-int(self.predictor.history_length) :]

    def plan(self, actions: np.ndarray, *, task_id: int) -> tuple[list[ActionSlice], dict[str, Any]]:
        if task_id != self.task_id:
            raise ValueError(f"horizon evaluator expected task {self.task_id}, received task {task_id}")
        result = self.runtime.plan(
            self._observations,
            tuple(np.asarray(action, dtype=np.float64) for action in actions),
            metadata={
                "task_id": task_id,
                "schedule_source": "horizon_ai_candidate_not_promoted",
            },
        )
        phases = tuple(
            None if label is None else f"task_{self.task_id:02d}:{label}" for label in result.metadata["subtask_labels"]
        )
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
