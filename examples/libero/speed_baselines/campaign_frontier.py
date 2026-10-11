"""Observed DS2 dominance on a matched bank, without changing reliability criteria.

Selection evidence is not population certification. The caller must separately
record whether these episodes were used for fitting, adaptive search or validation.
"""

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Episode:
    episode: int
    success: bool
    steps: int
    initial_observation_sha256: str
    first_request_sha256: str
    first_actions_sha256: str
    first_rng_sha256: str
    loop_wall_seconds: float | None = None

    def __post_init__(self):
        if type(self.episode) is not int or self.episode < 0 or type(self.success) is not bool:
            raise ValueError("Invalid episode identity or success flag")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("Nonnegative executed control steps required")
        for value in self.initial_identity:
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Actual initial observations, request, actions and RNG hashes required")
        wall = self.loop_wall_seconds
        if wall is not None and (isinstance(wall, bool) or not math.isfinite(wall) or wall < 0):
            raise ValueError("Measured loop wall time must be finite and nonnegative")

    @property
    def initial_identity(self):
        return (
            self.initial_observation_sha256,
            self.first_request_sha256,
            self.first_actions_sha256,
            self.first_rng_sha256,
        )


SHARED_PINS = (
    "suite",
    "task_id",
    "bddl_sha256",
    "policy_sha256",
    "inference_cache_sha256",
    "controller_sha256",
    "environment_sha256",
    "reset_protocol_sha256",
    "control_hz",
    "chunk_size",
)


def _metrics(episodes, control_hz):
    successes = [e for e in episodes if e.success]
    n = len(successes)
    total = sum(e.steps for e in episodes)
    mean = Fraction(sum(e.steps for e in successes), n) if n else None
    cost = Fraction(total, n) if n else None
    wall_available = all(e.loop_wall_seconds is not None for e in episodes)
    return {
        "trials": len(episodes),
        "successes": n,
        "success_rate": Fraction(n, len(episodes)),
        "mean_success_steps": mean,
        "mean_success_seconds": mean / control_hz if n else None,
        "spent_steps_per_success": cost,
        "spent_seconds_per_success": cost / control_hz if n else None,
        "total_steps": total,
        "mean_success_loop_wall_seconds": (
            sum(e.loop_wall_seconds for e in successes) / n if n and wall_available else None
        ),
        "spent_loop_wall_seconds_per_success": (
            sum(e.loop_wall_seconds for e in episodes) / n if n and wall_available else None
        ),
        "failures": [e.episode for e in episodes if not e.success],
    }


def compare_frontier(reference, candidate, *, reference_pins, candidate_pins, require_loop_wall=False):
    """Return a descriptive win only if SR and every selected time metric dominate.

    Failure-inclusive cost prevents a faster successful subset from hiding losses.
    Exact fractions are used for control-step metrics, without a tunable tolerance.
    Measured wall time is optional and must be preregistered as an acceptance metric;
    it includes instrumentation and first-use effects and is not robot motion time.
    """
    reference, candidate = tuple(reference), tuple(candidate)
    if not reference or [e.episode for e in reference] != [e.episode for e in candidate]:
        raise ValueError("Identical complete ordered episode banks required")
    if len({e.episode for e in reference}) != len(reference):
        raise ValueError("Duplicate episode identities in comparison bank")
    for name in SHARED_PINS:
        if name not in reference_pins or reference_pins[name] in (None, ""):
            raise ValueError(f"Missing shared contract pin: {name}")
        if reference_pins[name] != candidate_pins.get(name):
            raise ValueError(f"Unmatched contract pin: {name}")
    hz = reference_pins["control_hz"]
    if type(hz) is not int or hz <= 0:
        raise ValueError("Positive integer control frequency required")
    for a, b in zip(reference, candidate, strict=True):
        if a.initial_identity != b.initial_identity:
            raise ValueError(f"Initial scene or policy input mismatch for episode {a.episode}")
    r, c = _metrics(reference, hz), _metrics(candidate, hz)
    fields = ["mean_success_steps", "spent_steps_per_success"]
    if require_loop_wall:
        fields += ["mean_success_loop_wall_seconds", "spent_loop_wall_seconds_per_success"]
    comparable = all(r[k] is not None and c[k] is not None for k in fields)
    if require_loop_wall and any(e.loop_wall_seconds is None for e in reference + candidate):
        raise ValueError("Registered wall-time comparison requires every episode, including failures")
    checks = {"success_rate_not_lower": c["successes"] >= r["successes"]}
    checks.update({k + "_not_higher": comparable and c[k] <= r[k] for k in fields})
    strict = []
    if c["successes"] > r["successes"]:
        strict.append("success_rate")
    if comparable:
        strict.extend(k for k in fields if c[k] < r[k])
    win = all(checks.values()) and bool(strict)
    shared = [(a, b) for a, b in zip(reference, candidate, strict=True) if a.success and b.success]
    report = {
        "schema": "strider-observed-ds2-frontier-v1",
        "status": "OBSERVED_PARETO_WIN" if win else "NO_WIN",
        "comparable_successful_times": comparable,
        "checks": checks,
        "strict_improvements": strict,
        "reference": r,
        "candidate": c,
        "ordered_bank": [e.episode for e in reference],
        "shared_successes": len(shared),
        "mean_shared_success_step_delta": (
            Fraction(sum(b.steps - a.steps for a, b in shared), len(shared)) if shared else None
        ),
        "candidate_only_successes": [
            b.episode for a, b in zip(reference, candidate, strict=True) if b.success and not a.success
        ],
        "reference_only_successes": [
            a.episode for a, b in zip(reference, candidate, strict=True) if a.success and not b.success
        ],
        "require_loop_wall": require_loop_wall,
        "limits": [
            "Observed finite-bank comparison; no population reliability or statistical certification claim.",
            "Success-only times and failure-inclusive cost are separate; all failed runs remain counted.",
            "Loop wall time includes inference and instrumentation; it is not physical robot completion time.",
        ],
    }

    def serializable(value):
        if isinstance(value, Fraction):
            return float(value)
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [serializable(v) for v in value]
        return value

    return serializable(report)
