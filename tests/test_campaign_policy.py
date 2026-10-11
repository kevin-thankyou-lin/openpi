from __future__ import annotations

import hashlib
import dataclasses
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from examples.libero.speed_baselines import campaign_policy as cp
from examples.libero.speed_baselines import campaign_routes as cr
from examples.libero.speed_baselines import campaign_provenance as provenance
from examples.libero.speed_baselines.strider_client import validate_strider_server_metadata
from examples.libero.speed_baselines.strider_search_runner import render_libero_schedule


def _asset(path):
    return {"path": str(path), "sha256": cr.file_sha256(path)}


@pytest.fixture
def spec(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("(problem test)")
    registry = tmp_path / "routes.json"
    registry.write_text(json.dumps({"schema": cr.SCHEMA, "routes": [{
        "suite": "libero_spatial", "task_id": 0, "bddl": _asset(bddl), "method": "uniform", "uniform_stride": 2,
    }]}))
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"test weights")
    rows = [{"relative_path": "weights.bin", "size_bytes": weights.stat().st_size, "sha256": cr.file_sha256(weights)}]
    content_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = tmp_path / "policy-manifest.json"
    manifest.write_text(json.dumps({"schema": provenance.POLICY_SCHEMA, "policy": "pi05_libero", "files": rows,
                                    "file_count": 1, "total_bytes": weights.stat().st_size, "policy_sha256": content_sha}))
    path = tmp_path / "policy-spec.json"
    path.write_text(json.dumps({
        "schema": cp.SCHEMA, "route_registry": _asset(registry), "suite": "libero_spatial", "task_id": 0,
        "arm": "uniform2", "ordered_episode_indices": [7, 1], "policy_seed": 0, "prompt": "pick up the bowl",
        "run_dir": str(run), "expected_evaluator_commit": "a" * 40,
        "policy_manifest": _asset(manifest), "checkpoint_path": str(checkpoint), "policy_sha256": content_sha,
        "controller_sha256": cp.controller_identity()["sha256"],
    }))
    return cp.load_policy_spec(path)


class FakePolicy:
    _is_pytorch_model = False
    metadata = {"original": "retained"}

    def __init__(self):
        self._rng = np.array([99, 0], dtype=np.uint32)
        self.requests = []
        self.fail = False

    def infer(self, obs):
        assert cp.TAG not in obs
        self.requests.append(dict(obs))
        step = int(self._rng[1])
        self._rng = self._rng + np.array([0, 1], dtype=np.uint32)
        if self.fail:
            raise RuntimeError("fake inference failure")
        return {"actions": np.full((10, 7), step + 1, dtype=np.float32)}


def _seeded(spec, fake=None):
    verified = provenance.verify_policy_manifest(spec.policy_manifest.path,
        expected_manifest_sha256=spec.policy_manifest.sha256, checkpoint_path=spec.checkpoint_path,
        expected_policy_sha256=spec.policy_sha256)
    return cp.SeededPolicy(fake or FakePolicy(), spec, verified_policy=verified,
                           rng_key=lambda seed: np.array([seed, 0], dtype=np.uint32), rng_data=lambda rng: rng)


def _obs(spec):
    return {"prompt": spec.prompt, "observation/state": np.zeros(8, dtype=np.float64)}


def _tag(spec, ep=7, plan=0):
    return {"suite": "libero_spatial", "task": 0, "bddl_sha256": spec.route.identity.bddl_sha256,
            "seed": 0, "episode": ep, "plan": plan}


def test_three_call_canary_rng_reset_and_receipts(spec):
    fake = FakePolicy()
    policy = _seeded(spec, fake)
    assert len(fake.requests) == 3
    assert len({cp.request_hash(r) for r in fake.requests}) == 1
    canary = json.loads((spec.run_dir / "inference-canary.json").read_text())
    assert canary["passed"] and canary["warmup_calls"] == 1 and canary["repeat_calls"] == 2
    client = cp.EpisodePolicy(policy, spec, 7)
    first = client.infer(_obs(spec))
    second = client.infer(_obs(spec))
    other = cp.EpisodePolicy(policy, spec, 1).infer(_obs(spec))
    assert np.array_equal(first["actions"], other["actions"])
    assert not np.array_equal(first["actions"], second["actions"])
    assert first[cp.RECEIPT]["rng_before"] == other[cp.RECEIPT]["rng_before"] == [0, 0]
    receipts = [json.loads(line) for line in (spec.run_dir / "inference-rng.jsonl").read_text().splitlines()]
    assert len(receipts) == 3
    assert receipts[0]["raw_actions"] == first["actions"].tolist()
    assert receipts[0]["request_fields"]["observation/state"]["shape"] == [8]
    cp.validate_metadata(policy.metadata, spec)
    assert policy.metadata["original"] == "retained"
    policy.close()


@pytest.mark.parametrize("field,value", [
    ("suite", "libero_goal"), ("task", 1), ("bddl_sha256", "0" * 64), ("seed", 1),
    ("episode", 1), ("plan", 1), ("plan", -1), ("episode", True), ("seed", False), ("task", False),
])
def test_wrong_identity_or_first_request_never_reaches_model(spec, field, value):
    policy = _seeded(spec)
    tag = _tag(spec)
    tag[field] = value
    with pytest.raises(ValueError):
        policy.infer({**_obs(spec), cp.TAG: tag})
    assert len(policy.policy.requests) == 3
    assert policy._seen == []
    policy.close()


def test_order_duplicate_plan_and_exhausted_bank_rejected(spec):
    policy = _seeded(spec)
    request = lambda ep, plan: {**_obs(spec), cp.TAG: _tag(spec, ep, plan)}
    policy.infer(request(7, 0))
    for ep, plan in [(7, 0), (7, 2), (1, 1), (99, 0)]:
        with pytest.raises(ValueError):
            policy.infer(request(ep, plan))
    assert len(policy.policy.requests) == 4
    policy.infer(request(1, 0))
    with pytest.raises(ValueError):
        policy.infer(request(7, 0))
    assert len(policy.policy.requests) == 5
    policy.close()


def test_missing_tag_wrong_prompt_and_double_wrap_rejected(spec):
    policy = _seeded(spec)
    with pytest.raises(ValueError, match="tag"):
        policy.infer(_obs(spec))
    with pytest.raises(ValueError, match="prompt"):
        policy.infer({"prompt": "different", cp.TAG: _tag(spec)})
    with pytest.raises(ValueError, match="already wrapped"):
        cp.EpisodePolicy(policy, spec, 7).infer({**_obs(spec), cp.TAG: _tag(spec)})
    assert len(policy.policy.requests) == 3
    policy.close()


def test_failure_prevents_ambiguous_rng_retry(spec):
    policy = _seeded(spec)
    policy.policy.fail = True
    request = {**_obs(spec), cp.TAG: _tag(spec)}
    with pytest.raises(RuntimeError, match="fake inference"):
        policy.infer(request)
    with pytest.raises(RuntimeError, match="ambiguous RNG"):
        policy.infer(request)
    assert len(policy.policy.requests) == 4
    policy.close()


def test_canary_failure_keeps_failed_receipt_and_no_metadata(spec):
    class Unrepeatable(FakePolicy):
        def infer(self, obs):
            out = super().infer(obs)
            out["actions"] += len(self.requests)
            return out

    with pytest.raises(ValueError, match="canary failed"):
        _seeded(spec, Unrepeatable())
    assert json.loads((spec.run_dir / "inference-canary.json").read_text())["passed"] is False


def test_outputs_are_exclusive(spec):
    policy = _seeded(spec)
    policy.close()
    with pytest.raises(FileExistsError):
        _seeded(spec)


def test_client_rejects_altered_raw_action_receipt(spec):
    policy = _seeded(spec)

    class Altering:
        def infer(self, obs):
            out = policy.infer(obs)
            out["actions"] = out["actions"] + 1
            return out

    with pytest.raises(ValueError, match="request/action receipt"):
        cp.EpisodePolicy(Altering(), spec, 7).infer(_obs(spec))
    policy.close()


def test_uniform_handshake_gate_precedes_environment_creation(spec):
    metadata = cp.expected_metadata(spec)
    metadata["campaign_inference_canary_passed"] = True
    events = []

    class Client:
        def get_server_metadata(self):
            return metadata

    main = types.SimpleNamespace(
        websocket_client_policy=types.SimpleNamespace(WebsocketClientPolicy=lambda *a, **kw: Client()),
        _get_libero_env=lambda *a, **kw: events.append("env"),
        _run_episode=lambda **kw: events.append("episode"), eval_speed_baseline=lambda args: events.append("eval"),
    )
    cp.install_evaluator_policy(main, spec)
    with pytest.raises(RuntimeError, match="before environment"):
        main._get_libero_env()
    metadata["campaign"]["task_identity"]["suite"] = "libero_goal"
    with pytest.raises(ValueError, match="handshake"):
        main.websocket_client_policy.WebsocketClientPolicy()
    assert events == []
    with pytest.raises(RuntimeError, match="before environment"):
        main._get_libero_env()
    metadata = {**cp.expected_metadata(spec), "campaign_inference_canary_passed": True}
    main.websocket_client_policy.WebsocketClientPolicy()
    main._get_libero_env()
    assert events == ["env"]
    args = types.SimpleNamespace(task_suite_name="libero_spatial", task_start=0, task_count=1,
                                 num_trials_per_task=2, seed=7, method="uniform", uniform_stride=2)
    main.eval_speed_baseline(args)
    args.uniform_stride = 3
    with pytest.raises(ValueError, match="contract"):
        main.eval_speed_baseline(args)
    args.uniform_stride = 2
    with pytest.raises(ValueError, match="order mismatch before reset"):
        main._run_episode(args=args, task_id=0, episode_index=0, policy=Client())
    assert events == ["env", "eval"]
    with pytest.raises(ValueError, match="already installed"):
        cp.install_evaluator_policy(main, spec)


def test_preflight_does_not_import_gpu_or_simulator_and_cli_is_executable(spec, monkeypatch):
    monkeypatch.setattr(cp.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    receipt = cp.preflight(spec, repo=spec.path.parent)
    assert receipt["inference_calls"] == receipt["physics_steps"] == 0
    assert not (spec.run_dir / "inference-rng.jsonl").exists()
    # Help must work with no environment, policy, GPU or simulator initialization.
    script = pathlib.Path(cp.__file__).resolve().parents[3] / "scripts/serve_campaign_libero.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--preflight" in result.stdout


def test_strider_metadata_remains_compatible_with_frozen_evaluator(spec):
    registry = json.loads(spec.route_registry.path.read_text())
    entry = registry["routes"][0]
    checkpoint = spec.path.parent / "predictor.pt"
    checkpoint.write_bytes(b"test checkpoint")
    ontology = spec.path.parent / "ontology.json"
    ontology.write_text(json.dumps({"task_id": 0, "phases": ["approach", "grasp"]}))
    dataset = spec.path.parent / "phase-dataset.json"
    dataset.write_text(json.dumps({"schema":"strider-libero-prepared-phase-dataset-v1",
        "identity":dataclasses.asdict(spec.route.identity), "preprocessing":cr.PREPROCESSING,
        "classes":[{"name":"approach"},{"name":"grasp"}]}))
    entry.pop("uniform_stride")
    entry.update(method="strider", phase_checkpoint=_asset(checkpoint), ontology=_asset(ontology),
                 dataset_manifest=_asset(dataset),
                 phase_names=["approach", "grasp"], asset_binding=dataclasses.asdict(spec.route.identity))
    spec.route_registry.path.write_text(json.dumps(registry))
    plain = spec.path.parent / "schedule.json"
    plain.write_text(json.dumps({"approach": 3, "grasp": 1}))
    rendered = spec.path.parent / "rendered.json"
    render_libero_schedule(plain, rendered, task_id=0, phases=("approach", "grasp"),
                           task_suite_name="libero_spatial", bddl_sha256=spec.route.identity.bddl_sha256)
    payload = json.loads(spec.path.read_text())
    payload.update(arm="strider", route_registry=_asset(spec.route_registry.path), rendered_schedule=_asset(rendered))
    spec.path.write_text(json.dumps(payload))
    spec = cp.load_policy_spec(spec.path)
    metadata = cp.expected_metadata(spec)
    validate_strider_server_metadata(metadata, checkpoint_sha256=cr.file_sha256(checkpoint),
                                     schedule_sha256=cr.file_sha256(rendered), evaluator_commit="a" * 40, fast_stride=3)
    policy = _seeded(spec)
    cp.validate_metadata(policy.metadata, spec)
    policy.close()


def test_wrap_episode_hook_checks_bank_and_existing_wrapper(spec):
    fake = FakePolicy()
    wrapped = cp.wrap_episode_policy(fake, spec, 7)
    assert wrapped.policy is fake and wrapped.episode == 7
    with pytest.raises(ValueError, match="outside"):
        cp.wrap_episode_policy(fake, spec, 0)
    with pytest.raises(ValueError, match="already wrapped"):
        cp.wrap_episode_policy(wrapped, spec, 7)


def test_serve_uses_verified_exact_checkpoint_and_pi05_config_with_mock_server_only(spec, monkeypatch):
    events = []
    monkeypatch.setattr(cp.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    for name in ["openpi", "openpi.serving", "openpi.policies", "openpi.training"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    config = types.ModuleType("openpi.training.config")
    config.get_config = lambda name: types.SimpleNamespace(name=name)
    factory = types.ModuleType("openpi.policies.policy_config")
    def loader(train_config, checkpoint):
        events.append(("create_trained_policy", train_config.name, checkpoint))
        return FakePolicy()
    factory.create_trained_policy = loader
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    fake_server = types.ModuleType("openpi.serving.websocket_policy_server")

    class Server:
        def __init__(self, *, policy, host, port, metadata):
            assert len(policy.policy.requests) == 3
            cp.validate_metadata(metadata, spec)
            events.append(("construct", host, port))

        def serve_forever(self):
            events.append("mock-serve-only")

    fake_server.WebsocketPolicyServer = Server
    monkeypatch.setitem(sys.modules, fake_server.__name__, fake_server)
    real_class = cp.SeededPolicy
    monkeypatch.setattr(cp, "SeededPolicy", lambda policy, s, **kw: real_class(
        policy, s, **kw, rng_key=lambda seed: np.array([seed, 0], dtype=np.uint32), rng_data=lambda rng: rng))
    cp.serve(spec, repo=spec.path.parent, port=8013)
    assert events == [("create_trained_policy", "pi05_libero", spec.checkpoint_path), ("construct", "127.0.0.1", 8013),
                      "mock-serve-only"]


def test_failing_canary_reference_blocks_serving(spec):
    reference = spec.path.parent / "reference.npy"
    np.save(reference, np.full((10, 7), 2, dtype=np.float32), allow_pickle=False)
    payload = json.loads(spec.path.read_text())
    payload["inference_canary_reference"] = _asset(reference)
    spec.path.write_text(json.dumps(payload))
    spec = cp.load_policy_spec(spec.path)
    with pytest.raises(ValueError, match="canary failed"):
        _seeded(spec)
    receipt = json.loads((spec.run_dir / "inference-canary.json").read_text())
    assert not receipt["passed"] and not receipt["cross_process_passed"]
    assert receipt["cross_process_tolerance"] == 0.0


def test_sub_micro_action_drift_fails_before_serving(spec):
    reference = spec.path.parent / "reference.npy"
    np.save(reference, np.full((10, 7), 1.0 + 5e-7, dtype=np.float32), allow_pickle=False)
    payload = json.loads(spec.path.read_text())
    payload["inference_canary_reference"] = _asset(reference)
    spec.path.write_text(json.dumps(payload))
    spec = cp.load_policy_spec(spec.path)
    with pytest.raises(ValueError, match="canary failed"):
        _seeded(spec)
    receipt = json.loads((spec.run_dir / "inference-canary.json").read_text())
    assert receipt["within_process_exact"] and not receipt["cross_process_exact"]
    assert 0 < receipt["cross_process_max_delta"] < 1e-6
    assert not (spec.run_dir / "inference-rng.jsonl").read_text()


def test_import_and_preflight_cli_forbid_gpu_and_simulator_modules(spec):
    repo = pathlib.Path(cp.__file__).resolve().parents[3]
    payload = json.loads(spec.path.read_text())
    payload["expected_evaluator_commit"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    spec.path.write_text(json.dumps(payload))
    script = repo / "scripts/serve_campaign_libero.py"
    code = '''
import builtins, runpy, sys
original = builtins.__import__
def checked(name, *a, **kw):
    if name.split('.')[0] in {'jax', 'torch', 'libero', 'serve_policy', 'openpi'}:
        raise RuntimeError('forbidden preflight import: ' + name)
    return original(name, *a, **kw)
builtins.__import__ = checked
sys.argv = [sys.argv[1], '--spec', sys.argv[2], '--preflight']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code, str(script), str(spec.path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["inference_calls"] == 0
    assert not (spec.run_dir / "inference-rng.jsonl").exists()


@pytest.mark.parametrize("field", ["policy_sha256", "controller_sha256"])
def test_fabricated_content_pin_rejected_at_load(spec, field):
    payload = json.loads(spec.path.read_text())
    payload[field] = "0" * 64
    spec.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        cp.load_policy_spec(spec.path)


def test_different_verified_checkpoint_rejected_before_canary(spec):
    verified = provenance.VerifiedPolicy(spec.path.parent / "other", spec.policy_sha256,
                                         spec.policy_manifest.sha256, 1, 12)
    fake = FakePolicy()
    with pytest.raises(ValueError, match="verification receipt"):
        cp.SeededPolicy(fake, spec, verified_policy=verified)
    assert not fake.requests


def test_load_is_lightweight_but_model_load_rejects_changed_weights(spec):
    (spec.checkpoint_path / "weights.bin").write_bytes(b"evil weights")
    cp.load_policy_spec(spec.path)  # No weight hashing during structural preflight.
    calls = []
    with pytest.raises(ValueError, match="bytes differ"):
        provenance.create_verified_policy(spec.policy_manifest.path,
            expected_manifest_sha256=spec.policy_manifest.sha256, checkpoint_path=spec.checkpoint_path,
            expected_policy_sha256=spec.policy_sha256, train_config=types.SimpleNamespace(name="pi05_libero"),
            loader=lambda *args: calls.append(args))
    assert not calls
