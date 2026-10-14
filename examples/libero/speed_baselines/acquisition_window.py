"""Opt-in acquisition slowdown using only the current policy action forecast.

No episode clock, simulator contact/object state, or future observations enter
this gate. Source-row lookahead never crosses the available policy chunk.
"""
from dataclasses import replace

import numpy as np


def validate_window(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) != {"kind", "lead_source_steps"}:
        raise ValueError("acquisition_window requires kind and lead_source_steps")
    if config["kind"] != "close_preview_v1":
        raise ValueError("unknown acquisition window kind")
    lead = config["lead_source_steps"]
    if type(lead) is not int or not 0 <= lead <= 9:
        raise ValueError("lead_source_steps must be an integer in 0..9")
    return dict(config)


def window_speeds(actions, labels, factors, fallback_indices, config):
    """Raise early acquisition 1 to 2; keep current/imminent close at 1.

    Positive gripper actions are close commands under the frozen LIBERO
    convention. A current closed command keeps acquisition protected. Opening
    actions far from any forecast close are eligible for 2, including recovery.
    Other phases and fallback rows are never changed.
    """
    config = validate_window(config)
    raw = np.asarray(actions, dtype=np.float64)
    if raw.shape != (10, 7) or not np.isfinite(raw).all():
        raise ValueError("window requires finite ten-by-seven policy actions")
    if len(labels) != 10 or len(factors) != 10:
        raise ValueError("window requires aligned ten-row phases and speeds")
    lead = config["lead_source_steps"]
    near = tuple(bool(np.any(raw[i:min(i + lead + 1, 10), -1] > 0)) for i in range(10))
    fallback = set(fallback_indices)
    out = tuple(2.0 if label == "acquire" and speed == 1 and not near[i] and i not in fallback else float(speed)
                for i, (label, speed) in enumerate(zip(labels, factors, strict=True)))
    return out, near


class ClosePreviewSchedule:
    def __init__(self, base, config):
        self.base = base
        self.config = validate_window(config)
        self.actions = None

    def decide(self, prediction):
        base = self.base.decide(prediction)
        factors, near = window_speeds(self.actions, prediction.labels, base.factors,
                                     base.metadata["fallback_indices"], self.config)
        return replace(base, factors=factors, metadata={
            **base.metadata, "resolved_speeds": factors,
            "acquisition_window": {"config": self.config, "base_speed_factors": base.factors,
                                   "close_within_lead": near},
        })
