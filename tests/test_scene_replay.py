# Replay codec and fake robosuite internals are intentionally exercised directly.
# ruff: noqa: SLF001
"""No MuJoCo simulation: test portable state and the fail-closed replay boundary."""

import copy
import dataclasses
import hashlib
import json
import random
import sys
import types

import numpy as np
import pytest

from examples.libero.speed_baselines import scene_replay as replay


class DeltaBuffer:
    __module__ = "robosuite.utils.buffers"

    def __init__(self):
        self.last = np.zeros(3)
        self.current = np.ones(3)


class Observable:
    def __init__(self, value):
        self._current_observed_value = value
        self._time_since_last_sample = 0.0
        self._sensor = lambda cache: value
        self._enabled = True


class Model:
    def __init__(self):
        self.body_pos = np.array([[0.0, 0.0, 0.0], [0.01, 0.34, 0.0]])
        self.body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        self.geometry = np.array([1, 2], dtype=np.int32)


class Data:
    def __init__(self):
        self.integration = np.arange(16, dtype=float)
        self.qacc = np.arange(3, dtype=float)


class Sim:
    def __init__(self):
        self.model = types.SimpleNamespace(_model=Model())
        self.data = types.SimpleNamespace(_data=Data())

    def forward(self):
        self.data._data.qacc[:] = -5

    def get_state(self):
        return types.SimpleNamespace(flatten=self.data._data.integration[:9].copy)


class Base:
    def __init__(self, bddl_file):
        self.bddl_file_name = str(bddl_file)
        self.sim = Sim()
        self.timestep = 0
        self.done = False
        self.cur_time = 0.0
        self._obs_cache = {"a": np.ones(3)}
        self._observables = {
            "a": Observable(np.ones(3)),
            "image": Observable(np.arange(192, dtype=np.uint8).reshape(8, 8, 3)),
        }
        controller = types.SimpleNamespace(
            sim=self.sim,
            interpolator_pos=None,
            interpolator_ori=None,
            initial_joint=np.arange(7, dtype=float),
            goal_pos=np.ones(3),
            control_freq=20,
        )
        self.robots = [
            types.SimpleNamespace(
                controller=controller,
                gripper=types.SimpleNamespace(current_action=np.array([0.0])),
                recent_qpos=DeltaBuffer(),
                base_pos=np.zeros(3),
            )
        ]

    def _get_observations(self):
        return {key: value._current_observed_value for key, value in self._observables.items()}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    fake = types.ModuleType("mujoco")
    fake.__version__ = "2.3.7"
    fake.mjtState = types.SimpleNamespace(mjSTATE_INTEGRATION=4095)
    fake.mj_stateSize = lambda model, bits: 16
    fake.mj_getState = lambda model, data, state, bits: state.__setitem__(slice(None), data.integration)
    fake.mj_setState = lambda model, data, state, bits: data.integration.__setitem__(slice(None), state)
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    bddl = tmp_path / "task.bddl"
    bddl.write_text("(define (problem portable-replay))\n")
    env = Base(bddl)
    identity = replay.SceneIdentity(
        suite="libero_10",
        task_id=9,
        episode_index=7,
        bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
        initial_state_sha256=replay.array_sha256(env.sim.get_state().flatten()),
        policy_sha256="a" * 64,
        controller_sha256="b" * 64,
    )
    # Preserve test-process RNG even when restoration changes it.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    yield env, identity
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def capture(environment):
    env, identity = environment
    return replay.capture_scene(env, env._get_observations(), identity=identity)


def test_capture_preserves_rng_and_observations(environment):
    env, identity = environment
    obs = env._get_observations()
    before = replay.observation_fingerprints(obs)
    numpy_state = replay._pack(np.random.get_state())
    python_state = replay._pack(random.getstate())
    first = replay.capture_scene(env, obs, identity=identity)
    assert replay.capture_scene(env, obs, identity=identity) == first
    assert replay.observation_fingerprints(obs) == before
    assert replay._pack(np.random.get_state()) == numpy_state
    assert replay._pack(random.getstate()) == python_state


def test_restore_preserves_full_portable_state(environment, tmp_path):
    env, identity = environment
    scene = capture(environment)
    path = tmp_path / "scene.json"
    digest = replay.write_scene(path, scene, expected_identity=identity)
    loaded = replay.load_scene(path, expected_sha256=digest, expected_identity=identity)
    assert loaded == scene
    # Official saved pose is unchanged, but fixture, controller and warmstart differ.
    env.sim.model._model.body_pos[1] = [7, 8, 9]
    env.sim.model._model.body_quat[1] = [0, 1, 0, 0]
    env.sim.data._data.integration[9:] = -1
    env.sim.data._data.qacc[:] = -2
    env.robots[0].controller.initial_joint[:] = 40
    env.robots[0].recent_qpos.current[:] = 5
    env.robots[0].gripper.current_action[:] = 1
    env._observables["a"]._current_observed_value[:] = -3
    env._obs_cache = {}
    np.random.random(7)
    random.random()
    observations = replay.restore_scene(env, loaded, expected_identity=identity)
    assert replay.capture_scene(env, observations, identity=identity) == scene
    # Existing evidence is never replaced.
    with pytest.raises(FileExistsError):
        replay.write_scene(path, loaded, expected_identity=identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("suite", "libero_goal"),
        ("task_id", 2),
        ("episode_index", 1),
        ("bddl_sha256", "1" * 64),
        ("initial_state_sha256", "2" * 64),
        ("policy_sha256", "3" * 64),
        ("controller_sha256", "4" * 64),
    ],
)
def test_wrong_binding_rejected_before_mutation(environment, field, replacement):
    env, identity = environment
    scene = capture(environment)
    body_before = env.sim.model._model.body_pos.copy()
    wrong = dataclasses.replace(identity, **{field: replacement})
    with pytest.raises(replay.SceneReplayError, match="Scene identity differs"):
        replay.restore_scene(env, scene, expected_identity=wrong)
    np.testing.assert_array_equal(env.sim.model._model.body_pos, body_before)


def test_wrong_actual_initial_pose_rejected(environment):
    env, identity = environment
    scene = capture(environment)
    env.sim.data._data.integration[1] += 0.01
    with pytest.raises(replay.SceneReplayError, match="initial pose differs"):
        replay.restore_scene(env, scene, expected_identity=identity)


def test_wrong_actual_bddl_rejected(environment, tmp_path):
    env, identity = environment
    scene = capture(environment)
    other = tmp_path / "other.bddl"
    other.write_text("different task")
    env.bddl_file_name = str(other)
    with pytest.raises(replay.SceneReplayError, match="BDDL differs"):
        replay.restore_scene(env, scene, expected_identity=identity)


def test_wrong_model_geometry_rejected_before_fixture_mutation(environment):
    env, identity = environment
    scene = capture(environment)
    env.sim.model._model.body_pos[1] = [7, 8, 9]
    before = env.sim.model._model.body_pos.copy()
    env.sim.model._model.geometry[0] = 100
    with pytest.raises(replay.SceneReplayError, match="topology/geometry"):
        replay.restore_scene(env, scene, expected_identity=identity)
    np.testing.assert_array_equal(env.sim.model._model.body_pos, before)


def test_corrupt_file_and_payload_rejected(environment, tmp_path):
    _, identity = environment
    scene = capture(environment)
    path = tmp_path / "scene.json"
    digest = replay.write_scene(path, scene, expected_identity=identity)
    with pytest.raises(replay.SceneReplayError, match="file SHA256"):
        replay.load_scene(path, expected_sha256="0" * 64, expected_identity=identity)
    changed = json.loads(path.read_text())
    changed["snapshot"]["payload"]["episode"] = 1
    path.write_text(json.dumps(changed))
    with pytest.raises(replay.SceneReplayError, match="file SHA256"):
        replay.load_scene(path, expected_sha256=digest, expected_identity=identity)
    # Even a matching newly supplied file hash does not bypass envelope integrity.
    with pytest.raises(replay.SceneReplayError, match="envelope digest"):
        replay.load_scene(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected_identity=identity
        )


def test_unsupported_version_rejected(environment, monkeypatch):
    monkeypatch.setattr(sys.modules["mujoco"], "__version__", "3.0.0")
    with pytest.raises(replay.SceneReplayError, match="Unvalidated MuJoCo"):
        capture(environment)


def test_pointers_and_object_arrays_are_not_serialized(environment):
    env, _ = environment
    env.robots[0].controller.unexpected_pointer = types.SimpleNamespace(value=1)
    with pytest.raises(TypeError, match="Unsupported replay state"):
        capture(environment)
    with pytest.raises(replay.SceneReplayError):
        replay._pack(np.array([object()], dtype=object))


def test_legacy_payload_is_wrapped_without_modification(environment):
    env, identity = environment
    legacy = capture(environment)["snapshot"]
    legacy["schema"] = "previous-audited-capture"
    before = copy.deepcopy(legacy)
    scene = replay.bind_snapshot(legacy, identity=identity)
    assert scene["snapshot"] == before
    observations = replay.restore_scene(env, scene, expected_identity=identity)
    assert replay.observation_fingerprints(observations) == replay.observation_fingerprints(env._get_observations())
    wrong = dataclasses.replace(identity, episode_index=1)
    with pytest.raises(replay.SceneReplayError, match="Snapshot episode differs"):
        replay.bind_snapshot(legacy, identity=wrong)


@pytest.mark.parametrize("suite", sorted(replay.SUPPORTED_SUITES))
def test_all_four_suite_identities_roundtrip(environment, suite):
    env, identity = environment
    identity = dataclasses.replace(identity, suite=suite)
    scene = replay.capture_scene(env, env._get_observations(), identity=identity)
    assert scene["identity"]["suite"] == suite
    replay.restore_scene(env, scene, expected_identity=identity)


def test_load_requires_exact_identity(environment, tmp_path):
    _, identity = environment
    scene = capture(environment)
    path = tmp_path / "scene.json"
    digest = replay.write_scene(path, scene, expected_identity=identity)
    with pytest.raises(replay.SceneReplayError, match="Scene identity differs"):
        replay.load_scene(path, expected_sha256=digest, expected_identity=dataclasses.replace(identity, task_id=0))


def test_only_known_portable_buffer_types_are_accepted():
    class UnknownBuffer:
        __module__ = "robosuite.utils.buffers"

    with pytest.raises(TypeError, match="Unsupported replay state"):
        replay._pack(UnknownBuffer())
    ring_type = type("RingBuffer", (), {"__module__": "robosuite.utils.buffers"})
    ring = ring_type()
    ring.buf = np.arange(12, dtype=float).reshape(4, 3)
    ring.ptr = 2
    packed = replay._pack(ring)
    target = ring_type()
    target.buf = np.zeros((4, 3))
    target.ptr = 0
    holder = types.SimpleNamespace(recent_values=target)
    replay._restore_attrs(holder, {"recent_values": packed})
    np.testing.assert_array_equal(target.buf, ring.buf)
    assert target.ptr == ring.ptr
    assert holder.recent_values is target


@pytest.fixture
def normalized_environment(environment):
    """Mock only the native normalization; test guard and replay without physics."""
    env, identity = environment
    model = env.sim.model._model
    model.nq, model.nv = 5, 3
    # Four quaternion components, one ordinary qpos, then three qvel.
    raw = np.array([0., -0.526621538951244, 0., 0., 0.8500998498486068, 7., 8., 9., 10.])
    identity = dataclasses.replace(identity, initial_state_sha256=replay.array_sha256(raw))

    def normalize(model, qpos):
        qpos[:4] /= np.linalg.norm(qpos[:4])

    sys.modules["mujoco"].mj_normalizeQuat = normalize
    expected = raw.copy()
    normalize(model, expected[1:6])
    assert expected.tobytes() != raw.tobytes()
    env.sim.data._data.integration[:9] = expected
    return env, identity, raw


def test_once_normalized_capture_restore_keeps_raw_identity(normalized_environment):
    env, identity, raw = normalized_environment
    original_bytes = raw.tobytes()
    integration = env.sim.data._data.integration.copy()
    scene = replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw)
    assert scene["identity"]["initial_state_sha256"] == replay.array_sha256(raw)
    assert raw.tobytes() == original_bytes
    assert env.sim.data._data.integration.tobytes() == integration.tobytes()

    # Restore's forward may normalize again. The existing second mj_setState
    # must preserve the captured integration bytes despite that non-idempotence.
    def forward():
        env.sim.data._data.integration[1] += 0.125
        env.sim.data._data.qacc[:] = -5

    env.sim.forward = forward
    obs = replay.restore_scene(env, scene, expected_identity=identity, initial_state=raw)
    assert env.sim.data._data.integration.tobytes() == integration.tobytes()
    assert replay.capture_scene(env, obs, identity=identity, initial_state=raw) == scene
    assert raw.tobytes() == original_bytes


def test_normalized_state_requires_original_pinned_bytes(normalized_environment):
    env, identity, raw = normalized_environment
    with pytest.raises(replay.SceneReplayError, match="initial pose differs"):
        replay.capture_scene(env, env._get_observations(), identity=identity)
    with pytest.raises(replay.SceneReplayError, match="pinned SHA256"):
        replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw + 1)
    # Even a byte-exact raw environment cannot bypass bad supplied evidence.
    env.sim.data._data.integration[:9] = raw
    with pytest.raises(replay.SceneReplayError, match="pinned SHA256"):
        replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw + 1)
    replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw)


@pytest.mark.parametrize("index", range(9))
def test_one_ulp_anywhere_beyond_expected_normalization_rejected(normalized_environment, index):
    env, identity, raw = normalized_environment
    scene = replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw)
    state = env.sim.data._data.integration
    state[index] = np.nextafter(state[index], np.inf)
    before = state.tobytes()
    with pytest.raises(replay.SceneReplayError, match="initial pose differs"):
        replay.restore_scene(env, scene, expected_identity=identity, initial_state=raw)
    assert state.tobytes() == before


@pytest.mark.parametrize(
    "bad", [lambda a: a.astype(np.float32), lambda a: a.reshape(3, 3), lambda a: a[:-1], lambda a: a * np.nan]
)
def test_original_state_layout_is_fail_closed(normalized_environment, bad):
    env, identity, raw = normalized_environment
    with pytest.raises(replay.SceneReplayError, match="finite float64"):
        replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=bad(raw))


def test_normalization_requires_pinned_mujoco(normalized_environment, monkeypatch):
    env, identity, raw = normalized_environment
    monkeypatch.setattr(sys.modules["mujoco"], "__version__", "3.0.0")
    with pytest.raises(replay.SceneReplayError, match="Unvalidated MuJoCo"):
        replay.capture_scene(env, env._get_observations(), identity=identity, initial_state=raw)
