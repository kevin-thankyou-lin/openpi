"""Exercise refusal/accounting and actual episode hooks without a GPU or simulator."""

import ast
import collections
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero.speed_baselines import campaign_policy as cp
from examples.libero.speed_baselines import campaign_routes as cr
from examples.libero.speed_baselines import campaign_session as cs
from examples.libero.speed_baselines.actions import SupActionComposer
from examples.libero.speed_baselines.actions import uniform_slices


def asset(path):
    return {"path": str(path), "sha256": cr.file_sha256(path)}


@pytest.fixture
def session_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "preflight", lambda *a, **kw: {})
    bddl = tmp_path / "task.bddl"
    bddl.write_text("test")
    routes = tmp_path / "routes.json"
    routes.write_text(
        json.dumps(
            {
                "schema": cr.SCHEMA,
                "routes": [
                    {
                        "suite": "libero_spatial",
                        "task_id": 0,
                        "method": "uniform",
                        "uniform_stride": 1,
                        "bddl": asset(bddl),
                    }
                ],
            }
        )
    )
    policy = tmp_path / "policy.json"
    (tmp_path / "run").mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"test weights")
    rows = [{"relative_path": "weights.bin", "size_bytes": weights.stat().st_size, "sha256": cr.file_sha256(weights)}]
    content_sha = cs.digest(rows)
    manifest = tmp_path / "policy-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "strider-base-policy-content-manifest-v1",
                "policy": "pi05_libero",
                "files": rows,
                "file_count": 1,
                "total_bytes": weights.stat().st_size,
                "policy_sha256": content_sha,
            }
        )
    )
    policy.write_text(
        json.dumps(
            {
                "schema": cp.SCHEMA,
                "route_registry": asset(routes),
                "suite": "libero_spatial",
                "task_id": 0,
                "arm": "uniform1",
                "ordered_episode_indices": [7, 1],
                "policy_seed": 0,
                "prompt": "test task",
                "run_dir": "run",
                "expected_evaluator_commit": "a" * 40,
                "policy_manifest": asset(manifest),
                "checkpoint_path": str(checkpoint),
                "policy_sha256": content_sha,
                "controller_sha256": cp.controller_identity()["sha256"],
            }
        )
    )
    source = Path(cs.__file__).parent
    spec = {
        "schema": cs.SCHEMA,
        "policy_spec": asset(policy),
        "purpose": "baseline",
        "baseline_cap": 4,
        "challenger_cap": 20,
        "implementation_sources": {
            k: asset(source / v)
            for k, v in {
                "session": "campaign_session.py",
                "evaluator": "main.py",
                "actions": "actions.py",
                "controller": "controller.py",
                "policy_adapter": "campaign_policy.py",
                "scene_replay": "scene_replay.py",
                "budget": "campaign_budget.py",
                "routes": "campaign_routes.py",
                "provenance": "campaign_provenance.py",
                "audit": "campaign_audit.py",
                "strider_client": "strider_client.py",
                "acquisition_window": "acquisition_window.py",
                "telemetry": "telemetry.py",
                "eval_entrypoint": "../../../scripts/eval_campaign_libero.py",
                "server_entrypoint": "../../../scripts/serve_campaign_libero.py",
            }.items()
        },
        "initial_states": {
            str(ep): {"sha256": cp.array_hash(np.array([ep], dtype=float)), "dtype": "float64", "shape": [1]}
            for ep in [7, 1]
        },
        "scene_mode": "capture",
        "source_pins": {"policy_sha256": content_sha, "controller_sha256": cp.controller_identity()["sha256"]},
        "unit_id": "first",
        "ledger_path": "budget.jsonl",
        "campaign_id": "test",
        "cap_authority": "user20",
        "baseline_authority": "registered4",
        "candidate_id": "uniform1",
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(spec))
    return path


def connect(session):
    session.connect({**cp.expected_metadata(session.policy_spec), "campaign_inference_canary_passed": True})


def test_failed_started_episode_counts_and_unused_reservation_releases(session_spec):
    s = cs.CampaignSession(session_spec, repo=Path(cs.__file__).parents[3])
    connect(s)
    s.before_episode(7, object())
    assert s.budget.snapshot()["total_starts"] == 1
    # Simulate a reset/restore/physics exception before any terminal result.
    s.close()
    final = s.budget.snapshot()
    assert final["by_purpose"]["baseline"] == 1
    assert final["limits"]["baselines"]["reserved"] == 0
    events = [json.loads(line)["event"] for line in s.budget.path.read_text().splitlines()]
    assert any(e.get("outcome") == "infrastructure_failure" for e in events)
    with pytest.raises(FileExistsError):
        connect(cs.CampaignSession(session_spec, repo=Path(cs.__file__).parents[3]))


def test_wrong_handshake_and_wrong_order_consume_no_start(session_spec):
    s = cs.CampaignSession(session_spec, repo=Path(cs.__file__).parents[3])
    with pytest.raises(ValueError, match="handshake"):
        s.connect({})
    assert not (session_spec.parent / "budget.jsonl").exists()
    connect(s)
    with pytest.raises(ValueError, match="order"):
        s.before_episode(1, object())
    assert s.budget.snapshot()["total_starts"] == 0
    s.close()


@pytest.mark.parametrize("mutation", ["bank", "cap", "candidate_capture", "source"])
def test_invalid_spec_rejected_before_simulation(session_spec, mutation):
    spec = json.loads(session_spec.read_text())
    if mutation == "bank":
        del spec["initial_states"]["1"]
    elif mutation == "cap":
        spec["challenger_cap"] = 21
    elif mutation == "candidate_capture":
        spec["purpose"] = "causal_replay"
    else:
        spec["implementation_sources"]["evaluator"] = spec["implementation_sources"]["session"]
    session_spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="bank|cap|restore|source pins"):
        cs.CampaignSession(session_spec, repo=Path(cs.__file__).parents[3])
    assert not (session_spec.parent / "budget.jsonl").exists()


def test_initial_state_bytes_not_only_episode_number_checked(session_spec):
    s = cs.CampaignSession(session_spec, repo=Path(cs.__file__).parents[3])
    task = SimpleNamespace(language="test task")
    states = {7: np.array([7.0]), 1: np.array([1.0])}
    s.validate_task(task, states, s.policy_spec.route.bddl.path)
    states[7][0] += 0.001
    with pytest.raises(ValueError, match="official initial state"):
        s.validate_task(task, states, s.policy_spec.route.bddl.path)


def test_real_episode_hook_boundaries_preserve_actions_and_settling(tmp_path):
    # Compile the actual episode function; dependencies are supplied below to
    # avoid importing LIBERO/JAX. This exercises executed source, not a copy.
    source = Path(cs.__file__).with_name("main.py")
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_episode")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    composer = SupActionComposer(
        input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6), output_max=np.ones(6)
    )

    def schedule(actions, *args):
        return uniform_slices(actions, stride=2, composer=composer), {}

    ns = {
        "np": np,
        "logging": logging,
        "collections": collections,
        "MAX_STEPS": {"libero_spatial": 10},
        "LIBERO_DUMMY_ACTION": [0] * 6 + [-1],
        "_schedule_actions": schedule,
        "_quat2axisangle": lambda q: np.zeros(3),
        "image_tools": SimpleNamespace(convert_to_uint8=lambda x: x, resize_with_pad=lambda x, *a: x),
        "imageio": SimpleNamespace(mimwrite=lambda *a, **kw: None),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)

    def observation():
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    class Environment:
        def __init__(self):
            self.actions, self.events = [], []

        def reset(self):
            self.events.append("reset")

        def set_init_state(self, state):
            self.events.append("set_init")
            self.original_state = state
            return observation()

        def step(self, action):
            self.actions.append(action)
            return observation(), 0, len(self.actions) == 12, {}

    class Hooks:
        def after_set_init_state(self, env, obs, *, initial_state):
            assert initial_state is env.original_state
            assert env.events == ["reset", "set_init"]
            assert not env.actions
            self.env = env
            return obs

        def after_warmup(self, env, obs):
            assert len(env.actions) == 10

        def before_action(self, env, obs, action, step, decision):
            assert len(env.actions) == 10 + step

        def after_loop(self, env, obs, success, steps):
            assert success
            assert steps == 2
            assert len(env.actions) == 12

    args = SimpleNamespace(
        task_suite_name="libero_spatial",
        num_steps_wait=10,
        resize_size=224,
        chunk_size=10,
        save_strider_telemetry=False,
    )
    policy = SimpleNamespace(infer=lambda obs: {"actions": np.full((10, 7), 0.1)})
    baseline, instrumented = Environment(), Environment()
    for env, hooks in [(baseline, None), (instrumented, Hooks())]:
        result = ns["_run_episode"](
            env=env,
            task_description="test",
            initial_state=np.zeros(1),
            task_id=0,
            episode_index=7,
            args=args,
            policy=policy,
            composer=composer,
            selector=None,
            strider=None,
            videos_dir=tmp_path,
            campaign=hooks,
        )
        assert result["success"]
        assert result["env_steps"] == 2
    assert baseline.actions == instrumented.actions


def test_phase_manifest_and_actual_helper_pin_required_before_scene_start(session_spec):
    spec = json.loads(session_spec.read_text())
    policy_path = Path(spec["policy_spec"]["path"])
    policy = json.loads(policy_path.read_text())
    route_path = Path(policy["route_registry"]["path"])
    registry = json.loads(route_path.read_text())
    entry = registry["routes"][0]
    identity = {"suite":entry["suite"], "task_id":entry["task_id"], "bddl_sha256":entry["bddl"]["sha256"]}
    checkpoint = session_spec.parent / "phase.pt"
    checkpoint.write_bytes(b"opaque phase checkpoint")
    ontology = session_spec.parent / "ontology.json"
    ontology.write_text(json.dumps({"task_id":0,"phases":["approach"]}))
    dataset = session_spec.parent / "phase-manifest.json"
    dataset.write_text(json.dumps({"schema":"strider-libero-prepared-phase-dataset-v1", "identity":identity,
        "preprocessing":cr.PREPROCESSING,"classes":[{"name":"approach"}]}))
    entry.pop("uniform_stride")
    entry.update(method="strider",phase_checkpoint=asset(checkpoint),ontology=asset(ontology),
        dataset_manifest=asset(dataset),phase_names=["approach"],asset_binding=identity)
    route_path.write_text(json.dumps(registry))
    schedule = session_spec.parent / "rendered.json"
    schedule.write_text(json.dumps({"schema":"strider-libero-task-fine-phase-schedule-v1",
        "status":"AI_CANDIDATE_NOT_PROMOTED_OR_EVALUATED", "authority":"AI_CANDIDATE_NOT_HUMAN_ANNOTATION",
        "requires_human_confirmation":True,"phase_schedule":[["task_00:approach",2]],
        "task_suite_name":identity["suite"],"task_id":0,"bddl_sha256":identity["bddl_sha256"]}))
    policy.update(arm="strider",route_registry=asset(route_path),rendered_schedule=asset(schedule))
    policy_path.write_text(json.dumps(policy))
    spec.update(policy_spec=asset(policy_path),purpose="search_candidate")
    session_spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="missing executable source pins"):
        cs.CampaignSession(session_spec,repo=Path(cs.__file__).parents[3])
    spec["implementation_sources"]["phase_preprocessing"] = asset(Path(cs.__file__).with_name("phase_preprocessing.py"))
    session_spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="challengers must restore"):
        cs.CampaignSession(session_spec,repo=Path(cs.__file__).parents[3])
    assert not (session_spec.parent / "budget.jsonl").exists()


def test_implicit_noncampaign_phase_mode_refused_before_connect():
    source = Path(cs.__file__).with_name("main.py")
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="eval_speed_baseline")
    namespace={"Args":object,"_validate_args":lambda args:None}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),namespace)
    with pytest.raises(ValueError,match="explicit strider_legacy_input_mode"):
        namespace["eval_speed_baseline"](SimpleNamespace(method="strider",strider_legacy_input_mode=False))


@pytest.mark.parametrize("mode", ["capture", "restore"])
def test_scene_hook_forwards_exact_official_state(tmp_path, monkeypatch, mode):
    from examples.libero.speed_baselines.scene_replay import SceneIdentity

    original = np.zeros(9)
    identity = SceneIdentity("libero_10", 5, 31, "a" * 64, "b" * 64, "c" * 64, "d" * 64)
    scene, observations = {}, {"proprio": np.zeros(9)}
    calls = []

    def capture(env, obs, *, identity, initial_state):
        assert initial_state is original
        assert obs is observations
        calls.append("capture")
        return scene

    def restore(env, saved, *, expected_identity, initial_state):
        assert initial_state is original
        assert saved is scene
        calls.append("restore")
        return observations

    monkeypatch.setattr(cs, "capture_scene", capture)
    monkeypatch.setattr(cs, "restore_scene", restore)
    monkeypatch.setattr(cs, "write_scene", lambda *a, **kw: "e" * 64)
    session = SimpleNamespace(
        episode=31, scene_mode=mode, scenes={31: scene}, output=tmp_path,
        scene_identity=lambda ep: identity, record=lambda *a, **kw: None,
    )
    result = cs.CampaignSession.after_set_init_state(
        session, SimpleNamespace(control_timestep=0.05), observations, initial_state=original
    )
    assert result is observations
    assert calls == [mode]
