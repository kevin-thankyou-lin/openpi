"""Portable LIBERO scene replay at the boundary before settling controls.

LIBERO saved initial states omit fixture model transforms. Capture immediately
following the official ``env.reset(); env.set_init_state(initial_state)`` and
before settling controls; restore at the same boundary. This module does not
step physics, seed new scenes, change control laws, or render policy inputs.
The evaluator must verify subsequent observations and no-op rollout parity.

Snapshot files contain JSON and compressed typed arrays, never pickle. Caller
identity pins bind the scene to a suite, task, pose, BDDL, policy and controller.
Only MuJoCo 2.3.7 and concrete portable OSC runtime state are supported.
"""

# Runtime snapshots require robosuite internals; defer MuJoCo import for CPU tooling.
# ruff: noqa: SLF001, PLC0415

from __future__ import annotations

import base64
import collections
import dataclasses
import hashlib
import json
import pathlib
import random
import re
from typing import Any
import zlib

import numpy as np

SUPPORTED_MUJOCO_VERSION = "2.3.7"
SCENE_SCHEMA = "libero-scene-replay-v1"
SUPPORTED_SUITES = frozenset({"libero_spatial", "libero_object", "libero_goal", "libero_10"})


class SceneReplayError(ValueError):
    """The snapshot cannot establish the requested identical scene."""


@dataclasses.dataclass(frozen=True)
class SceneIdentity:
    suite: str
    task_id: int
    episode_index: int
    bddl_sha256: str
    initial_state_sha256: str
    policy_sha256: str
    controller_sha256: str

    def __post_init__(self):
        if self.suite not in SUPPORTED_SUITES:
            raise SceneReplayError(f"Unsupported LIBERO suite: {self.suite}")
        if type(self.task_id) is not int or self.task_id < 0:
            raise SceneReplayError("task_id must be a nonnegative integer")
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise SceneReplayError("episode_index must be a nonnegative integer")
        for name in ("bddl_sha256", "initial_state_sha256", "policy_sha256", "controller_sha256"):
            _check_sha256(getattr(self, name), name)


def _check_sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SceneReplayError(f"{name} must be a lowercase SHA256")


def _check_mujoco_version(mujoco):
    if mujoco.__version__ != SUPPORTED_MUJOCO_VERSION:
        raise SceneReplayError(f"Unvalidated MuJoCo version: {mujoco.__version__}")
    if int(mujoco.mjtState.mjSTATE_INTEGRATION) != 4095:
        raise SceneReplayError("Unvalidated integration state specification")


def array_sha256(value):
    """Hash exact contiguous array bytes, matching existing initial-state pins."""
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype.kind == "O":
        raise SceneReplayError("Object arrays are not portable replay state")
    return hashlib.sha256(a.tobytes()).hexdigest()


def _validate_env_identity(env, identity, initial_state=None):
    base = _base(env)
    if hashlib.sha256(pathlib.Path(base.bddl_file_name).read_bytes()).hexdigest() != identity.bddl_sha256:
        raise SceneReplayError("Environment BDDL differs from the pinned scene")
    state = np.asarray(base.sim.get_state().flatten())
    if initial_state is not None:
        original = np.asarray(initial_state)
        model = base.sim.model._model
        expected_shape = (1 + model.nq + model.nv,)
        if original.dtype != np.dtype("float64") or original.shape != expected_shape or not np.isfinite(original).all():
            raise SceneReplayError("Original initial state must be finite float64 time/qpos/qvel")
        if array_sha256(original) != identity.initial_state_sha256:
            raise SceneReplayError("Original initial state differs from its pinned SHA256")
        if state.dtype != original.dtype or state.shape != original.shape:
            raise SceneReplayError("Environment initial state layout differs from the original")
    if array_sha256(state) == identity.initial_state_sha256:
        return
    if initial_state is not None:
        import mujoco

        _check_mujoco_version(mujoco)
        # LIBERO set_init_state calls mj_forward once. That normalizes free/ball
        # joint quaternions in qpos, even when their norm differs only by an ULP.
        # Normalize a COPY of the pinned original, never the environment state.
        # One normalization is intentional: it need not be byte-idempotent.
        expected = original.copy()
        mujoco.mj_normalizeQuat(model, expected[1 : 1 + model.nq])
        if state.tobytes() == expected.tobytes():
            return
    raise SceneReplayError("Environment initial pose differs; restore only immediately after set_init_state")


def _canonical_sha256(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pack(x):
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        if a.dtype.kind not in "biufcSU":
            raise SceneReplayError(str(a.dtype))
        return {
            "_array_": True,
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "data": base64.b64encode(zlib.compress(a.tobytes())).decode(),
        }
    if isinstance(x, np.generic):
        return _pack(x.item())
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, dict):
        return {"_dict_": [[_pack(k), _pack(v)] for k, v in x.items()]}
    if isinstance(x, tuple):
        return {"_tuple_": [_pack(v) for v in x]}
    if isinstance(x, list):
        return {"_list_": [_pack(v) for v in x]}
    if isinstance(x, collections.deque):
        return {"_deque_": [_pack(v) for v in x], "maxlen": x.maxlen}
    if type(x).__module__ == "robosuite.utils.buffers" and type(x).__name__ in {"DeltaBuffer", "RingBuffer"}:
        return {"_buffer_": type(x).__module__ + "." + type(x).__name__, "attrs": _pack(vars(x))}
    raise TypeError("Unsupported replay state " + str(type(x)))


def _unpack(x):
    if not isinstance(x, dict):
        return x
    if "_array_" in x:
        return (
            np.frombuffer(zlib.decompress(base64.b64decode(x["data"])), dtype=np.dtype(x["dtype"]))
            .reshape(x["shape"])
            .copy()
        )
    if "_dict_" in x:
        return {_unpack(k): _unpack(v) for k, v in x["_dict_"]}
    if "_tuple_" in x:
        return tuple(_unpack(v) for v in x["_tuple_"])
    if "_list_" in x:
        return [_unpack(v) for v in x["_list_"]]
    if "_deque_" in x:
        return collections.deque((_unpack(v) for v in x["_deque_"]), maxlen=x["maxlen"])
    if "_buffer_" in x:
        raise TypeError("Restore buffer into existing object, never construct new runtime class")
    raise TypeError("Unknown replay encoding")


def _attrs(obj, exclude=()):
    out = {}
    for k, v in vars(obj).items():
        if k in exclude or callable(v):
            continue
        out[k] = _pack(v)
    return out


def _restore_attrs(obj, attrs):
    for k, v in attrs.items():
        if not hasattr(obj, k):
            raise SceneReplayError(k)
        if isinstance(v, dict) and "_buffer_" in v:
            current = getattr(obj, k)
            if type(current).__module__ + "." + type(current).__name__ != v["_buffer_"]:
                raise SceneReplayError("Scene replay state validation failed")
            current.__dict__.clear()
            current.__dict__.update(_unpack(v["attrs"]))
        else:
            setattr(obj, k, _unpack(v))


def _simple_attrs(obj):
    return {
        k: _pack(v)
        for k, v in vars(obj).items()
        if v is None or isinstance(v, (str, bool, int, float, np.ndarray, np.generic)) or k.startswith("recent_")
    }


def _base(env):
    while getattr(env, "env", None) is not None:
        env = env.env
    if not ("sim" in vars(env) and "_observables" in vars(env)):
        raise SceneReplayError("Scene replay state validation failed")
    return env


def model_fingerprints(model):
    return {
        k: field_fingerprint(getattr(model, k))
        for k in dir(model)
        if not k.startswith("_")
        and isinstance(getattr(model, k), np.ndarray)
        and (getattr(model, k).dtype.kind in "biufc")
    }


def field_fingerprint(v):
    if isinstance(v, str):
        return {"type": "str", "sha256": hashlib.sha256(v.encode()).hexdigest()}
    a = np.ascontiguousarray(np.asarray(v))
    if a.dtype.kind == "O":
        raise SceneReplayError("Scene replay state validation failed")
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def observation_fingerprints(obs):
    return {k: field_fingerprint(v) for k, v in sorted(obs.items())}


def _capture_snapshot(env, obs, episode):
    import mujoco

    _check_mujoco_version(mujoco)
    b = _base(env)
    m = b.sim.model._model
    d = b.sim.data._data
    bits = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, bits), dtype=np.float64)
    mujoco.mj_getState(m, d, state, bits)
    robots = []
    for r in b.robots:
        if not (r.controller.interpolator_pos is None and r.controller.interpolator_ori is None):
            raise SceneReplayError("Scene replay state validation failed")
        robots.append(
            {
                "controller": _attrs(r.controller, exclude=("sim", "_sim")),
                "runtime": _simple_attrs(r),
                "gripper_current_action": _pack(r.gripper.current_action),
            }
        )
    payload = {
        "episode": int(episode),
        "mujoco_version": mujoco.__version__,
        "state_spec": int(bits),
        "model_body_pos": _pack(m.body_pos),
        "model_body_quat": _pack(m.body_quat),
        "model_fingerprints": model_fingerprints(m),
        "integration_state": _pack(state),
        "qacc": _pack(d.qacc),
        "robots": robots,
        "environment_scalars": _simple_attrs(b),
        "obs_cache": _pack(b._obs_cache),
        "observables": {k: _attrs(v) for k, v in b._observables.items()},
        "observation": _pack(obs),
        "numpy_rng": _pack(np.random.get_state()),
        "python_rng": _pack(random.getstate()),
    }
    return {"schema": "libero-prewarmup-state-v1", "payload": payload, "sha256": _canonical_sha256(payload)}


def _restore_snapshot(env, snapshot):
    import mujoco

    _check_mujoco_version(mujoco)
    p = snapshot["payload"]
    if snapshot["sha256"] != _canonical_sha256(p):
        raise SceneReplayError("Scene replay state validation failed")
    b = _base(env)
    m = b.sim.model._model
    d = b.sim.data._data
    if not (p["mujoco_version"] == mujoco.__version__ and p["state_spec"] == int(mujoco.mjtState.mjSTATE_INTEGRATION)):
        raise SceneReplayError("Scene replay state validation failed")
    for name in ["body_pos", "body_quat"]:
        a = _unpack(p["model_" + name])
        target = getattr(m, name)
        if not (target.shape == a.shape and target.dtype == a.dtype):
            raise SceneReplayError("Scene replay state validation failed")
        target[:] = a
    state = _unpack(p["integration_state"])
    if state.shape != (mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION),):
        raise SceneReplayError("Scene replay state validation failed")
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    b.sim.forward()
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    d.qacc[:] = _unpack(p["qacc"])
    if model_fingerprints(m) != p["model_fingerprints"]:
        raise SceneReplayError("Unrestored model array differs")
    if len(b.robots) != len(p["robots"]):
        raise SceneReplayError("Scene replay state validation failed")
    for r, s in zip(b.robots, p["robots"], strict=True):
        _restore_attrs(r.controller, s["controller"])
        _restore_attrs(r, s["runtime"])
        r.gripper.current_action = _unpack(s["gripper_current_action"])
    _restore_attrs(b, p["environment_scalars"])
    b._obs_cache = _unpack(p["obs_cache"])
    if set(b._observables) != set(p["observables"]):
        raise SceneReplayError("Scene replay state validation failed")
    for k, v in p["observables"].items():
        _restore_attrs(b._observables[k], v)
    np.random.set_state(_unpack(p["numpy_rng"]))
    random.setstate(_unpack(p["python_rng"]))
    obs = b._get_observations()
    expected = _unpack(p["observation"])
    if observation_fingerprints(obs) != observation_fingerprints(expected):
        raise SceneReplayError("Scene replay state validation failed")
    check = _capture_snapshot(env, obs, p["episode"])
    if check["sha256"] != snapshot["sha256"]:
        raise SceneReplayError("Restored pre-warmup state fingerprint mismatch")
    return obs


def capture_scene(env, observations, *, identity: SceneIdentity, initial_state=None) -> dict[str, Any]:
    """Capture before settling; supply pinned raw state to allow one quaternion normalization."""
    _validate_env_identity(env, identity, initial_state)
    return bind_snapshot(_capture_snapshot(env, observations, identity.episode_index), identity=identity)


def bind_snapshot(snapshot: dict, *, identity: SceneIdentity) -> dict[str, Any]:
    """Attach identity to a captured snapshot, including a previously audited one.

    This is a provenance operation, not a claim that an old capture used the
    supplied policy. The caller must establish that association from run receipts.
    The legacy physics payload and its digest remain untouched.
    """
    if snapshot.get("sha256") != _canonical_sha256(snapshot["payload"]):
        raise SceneReplayError("Snapshot payload digest mismatch")
    if snapshot["payload"]["episode"] != identity.episode_index:
        raise SceneReplayError("Snapshot episode differs from requested identity")
    if snapshot["payload"]["mujoco_version"] != SUPPORTED_MUJOCO_VERSION or snapshot["payload"]["state_spec"] != 4095:
        raise SceneReplayError("Unsupported captured physics version")
    binding = dataclasses.asdict(identity)
    result = {"schema": SCENE_SCHEMA, "identity": binding, "snapshot": snapshot}
    result["sha256"] = _canonical_sha256(result)
    return result


def _validate_scene(scene, expected_identity):
    if scene.get("schema") != SCENE_SCHEMA:
        raise SceneReplayError("Unknown scene envelope")
    SceneIdentity(**scene["identity"])
    if scene["identity"] != dataclasses.asdict(expected_identity):
        raise SceneReplayError("Scene identity differs: suite/task/pose/BDDL/policy/controller must all match")
    fields = {k: v for k, v in scene.items() if k != "sha256"}
    if set(fields) != {"schema", "identity", "snapshot"} or scene.get("sha256") != _canonical_sha256(fields):
        raise SceneReplayError("Scene envelope digest mismatch")
    rebound = bind_snapshot(scene["snapshot"], identity=expected_identity)
    if rebound != scene:
        raise SceneReplayError("Scene snapshot does not match its identity envelope")


def write_scene(path, scene, *, expected_identity: SceneIdentity) -> str:
    """Write one new snapshot file; never silently replace existing evidence."""
    _validate_scene(scene, expected_identity)
    data = (json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with pathlib.Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def load_scene(path, *, expected_sha256: str, expected_identity: SceneIdentity) -> dict[str, Any]:
    """Load an explicit path only after validating its externally pinned hash."""
    _check_sha256(expected_sha256, "expected_sha256")
    data = pathlib.Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise SceneReplayError("Snapshot file SHA256 mismatch")
    scene = json.loads(data)
    _validate_scene(scene, expected_identity)
    return scene


def restore_scene(env, scene, *, expected_identity: SceneIdentity, initial_state=None):
    """Restore initialization only; never advances physics or reseeds a new pose.

    The current environment must already contain the pinned official init state.
    Scene and topology checks happen before mutation. A restoration exception is
    fatal for that episode: do not continue stepping a partially restored state.
    """
    _validate_scene(scene, expected_identity)
    _validate_env_identity(env, expected_identity, initial_state)
    model = _base(env).sim.model._model
    actual = model_fingerprints(model)
    expected = scene["snapshot"]["payload"]["model_fingerprints"]
    if set(actual) != set(expected) or any(
        actual[k] != expected[k] for k in actual if k not in {"body_pos", "body_quat"}
    ):
        raise SceneReplayError("Model topology/geometry differs beyond restorable fixture transforms")
    return _restore_snapshot(env, scene["snapshot"])
