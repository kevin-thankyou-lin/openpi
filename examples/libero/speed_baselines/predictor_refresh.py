"""Controller-clock predictor refresh without requesting new policy actions."""
from collections import deque
import math


def refresh_interval(min_hz: float, control_hz: int = 20) -> int | None:
    if not math.isfinite(min_hz) or min_hz < 0 or min_hz > control_hz:
        raise ValueError("predictor minimum Hz must be finite and between 0 and control Hz")
    return None if min_hz == 0 else max(1, math.floor(control_hz / min_hz))


class RefreshableActionQueue:
    """Retain native actions and consume each source index exactly once.

    Fresh predictions start at row zero relative to the latest observation.
    They are applied only to the unconsumed source suffix, never to composed
    controller commands or already executed actions.
    """

    def __init__(self, min_hz: float):
        self.interval = refresh_interval(min_hz)
        self.pending = deque()
        self.actions = None
        self.offset = 0
        self.last_prediction_step = None

    def __bool__(self):
        return bool(self.pending)

    def start(self, actions, selector, *, task_id, env_step):
        if self.pending:
            raise ValueError("cannot replace an unconsumed policy chunk")
        self.actions = actions.copy()
        self.offset = 0
        slices, decision = selector.plan(self.actions, task_id=task_id)
        self.pending.extend(slices)
        self.last_prediction_step = env_step
        return decision

    def refresh(self, selector, *, task_id, env_step):
        if (not self.pending or self.interval is None
                or env_step - self.last_prediction_step < self.interval):
            return None
        slices, decision = selector.plan_suffix(
            self.actions[self.offset:], task_id=task_id, source_offset=self.offset)
        self.pending = deque(slices)
        self.last_prediction_step = env_step
        return decision

    def popleft(self):
        item = self.pending.popleft()
        expected = tuple(range(self.offset, self.offset + item.stride))
        if tuple(item.source_indices) != expected:
            raise ValueError("refreshed plan skipped or repeated source actions")
        self.offset += item.stride
        if self.offset > len(self.actions) or (not self.pending and self.offset != len(self.actions)):
            raise ValueError("refreshed plan must consume the entire source chunk")
        return item


class PrefixPredictor:
    """Use the fresh full prediction's prefix for a shorter remaining chunk."""

    def __init__(self, predictor, horizon):
        if not 0 < horizon <= predictor.horizon:
            raise ValueError("remaining horizon must fit the trained predictor")
        self.predictor = predictor
        self.horizon = horizon
        self.class_names = predictor.class_names

    def predict(self, observation_history):
        from dataclasses import replace
        from strider.subtask_prediction import predict_subtasks
        full = predict_subtasks(self.predictor, observation_history,
                                expected_horizon=self.predictor.horizon)
        return replace(full, probabilities=full.probabilities[:self.horizon],
                       valid=full.valid[:self.horizon])
