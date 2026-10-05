from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class ActionSlice:
    """One environment action and the source-policy indices it consumes."""

    action: np.ndarray
    source_indices: tuple[int, ...]

    @property
    def stride(self) -> int:
        return len(self.source_indices)


class SupActionComposer:
    """Exact LIBERO delta-action composition used by the supplied SuP code.

    Arm actions are first mapped from the normalized policy range into the
    robosuite controller output range. Translation deltas are summed and
    rotations are left-composed because LIBERO uses the global frame. The last
    gripper command is retained unchanged. Gripper acceleration is provided by
    the controller patch, so it must not also be multiplied here.
    """

    def __init__(
        self,
        *,
        input_min: np.ndarray,
        input_max: np.ndarray,
        output_min: np.ndarray,
        output_max: np.ndarray,
    ) -> None:
        self.input_min = _as_six(input_min, "input_min")
        self.input_max = _as_six(input_max, "input_max")
        self.output_min = _as_six(output_min, "output_min")
        self.output_max = _as_six(output_max, "output_max")
        expected = (6,)
        for name, value in (
            ("input_min", self.input_min),
            ("input_max", self.input_max),
            ("output_min", self.output_min),
            ("output_max", self.output_max),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        denominator = self.input_max - self.input_min
        if np.any(denominator == 0):
            raise ValueError("controller input range must be nonzero")
        self._scale = np.abs(self.output_max - self.output_min) / np.abs(denominator)
        self._input_center = (self.input_max + self.input_min) / 2.0
        self._output_center = (self.output_max + self.output_min) / 2.0

    @classmethod
    def from_robosuite(cls) -> "SupActionComposer":
        from robosuite import load_controller_config

        config = load_controller_config(default_controller="OSC_POSE")
        return cls(
            input_min=np.asarray(config["input_min"]),
            input_max=np.asarray(config["input_max"]),
            output_min=np.asarray(config["output_min"]),
            output_max=np.asarray(config["output_max"]),
        )

    def to_controller_space(self, actions: np.ndarray) -> np.ndarray:
        actions = _validate_actions(actions)
        result = actions.astype(np.float64, copy=True)
        result[:, :6] = (result[:, :6] - self._input_center) * self._scale + self._output_center
        return result

    def from_controller_space(self, actions: np.ndarray) -> np.ndarray:
        actions = _validate_actions(actions)
        result = actions.astype(np.float64, copy=True)
        result[:, :6] = (result[:, :6] - self._output_center) / self._scale + self._input_center
        return result

    def merge(self, actions: np.ndarray) -> np.ndarray:
        raw = self.to_controller_space(actions)
        delta_position = np.sum(raw[:, :3], axis=0)
        merged_rotation = Rotation.identity()
        for rotation in Rotation.from_rotvec(raw[:, 3:6]):
            merged_rotation = rotation * merged_rotation
        merged_raw = np.concatenate(
            [delta_position, merged_rotation.as_rotvec(), np.asarray([raw[-1, 6]])]
        )
        return self.from_controller_space(merged_raw[None, :])[0]


def native_slices(actions: np.ndarray) -> list[ActionSlice]:
    actions = _validate_actions(actions)
    return [ActionSlice(action.copy(), (index,)) for index, action in enumerate(actions)]


def uniform_slices(
    actions: np.ndarray,
    *,
    stride: int,
    composer: SupActionComposer,
) -> list[ActionSlice]:
    actions = _validate_actions(actions)
    _validate_stride(stride)
    result: list[ActionSlice] = []
    for start in range(0, len(actions), stride):
        stop = min(start + stride, len(actions))
        indices = tuple(range(start, stop))
        group = actions[start:stop]
        merged = group[0].copy() if len(group) == 1 else composer.merge(group)
        result.append(ActionSlice(merged, indices))
    return result


def sail_precision_slices(
    actions: np.ndarray,
    precision_scores: np.ndarray,
    *,
    threshold: float,
    fast_stride: int,
    composer: SupActionComposer,
) -> list[ActionSlice]:
    """Merge only all-fast groups, preserving critical and gripper-edge actions.

    Every proposed action in a candidate group is examined. Both sides of a
    binary gripper transition are forced to stride one so a fast group cannot
    swallow the transition or the action immediately preceding it.
    """

    actions = _validate_actions(actions)
    scores = np.asarray(precision_scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(actions):
        raise ValueError(f"precision score count {len(scores)} != action count {len(actions)}")
    if not np.isfinite(scores).all():
        raise ValueError("precision scores must be finite")
    if not np.isfinite(threshold):
        raise ValueError("precision threshold must be finite")
    _validate_stride(fast_stride)

    critical = scores >= threshold
    gripper_binary = actions[:, 6] >= 0.0
    transition_indices = np.flatnonzero(gripper_binary[1:] != gripper_binary[:-1]) + 1
    for index in transition_indices:
        critical[index - 1 : index + 1] = True

    result: list[ActionSlice] = []
    index = 0
    while index < len(actions):
        stop = min(index + fast_stride, len(actions))
        can_merge = stop - index == fast_stride and not critical[index:stop].any()
        if can_merge:
            indices = tuple(range(index, stop))
            result.append(ActionSlice(composer.merge(actions[index:stop]), indices))
            index = stop
        else:
            result.append(ActionSlice(actions[index].copy(), (index,)))
            index += 1
    return result


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] == 0:
        raise ValueError(f"actions must have shape (N, 7) with N > 0, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("actions must be finite")
    return value


def _validate_stride(stride: int) -> None:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")


def _as_six(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return np.full(6, float(array), dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} must be scalar or shape (6,), got {array.shape}")
    return array
