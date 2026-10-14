"""Suite-bound pi0.5 serving and episode RNG protocol, with no eager GPU imports."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import subprocess
import types
from typing import Any

import numpy as np

from .campaign_provenance import VerifiedPolicy, create_verified_policy, read_policy_manifest, verify_controller_identity
from .campaign_routes import CampaignRoute
from .campaign_routes import FileAsset
from .campaign_routes import file_sha256
from .campaign_routes import load_routes
from .campaign_routes import select_route

SCHEMA = "strider-libero-policy-spec-v1"
BASE_POLICY = "pi05_libero"
BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
CONTROLLER = "sup-supplement-unclip-panda-speed-0.02-mujoco-step"
TAG = "_diagnostic_episode"
RECEIPT = "_diagnostic_rng"


def array_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def request_hash(obs: dict) -> str:
    """Preserve the prior campaign's model-input hash, after stripping tags."""
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = obs[key]
        digest.update(key.encode())
        digest.update(value.encode() if isinstance(value, str) else np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    path: pathlib.Path
    sha256: str
    route_registry: FileAsset
    route: CampaignRoute
    ordered_episode_indices: tuple[int, ...]
    prompt: str
    run_dir: pathlib.Path
    expected_evaluator_commit: str
    policy_manifest: FileAsset
    checkpoint_path: pathlib.Path
    policy_sha256: str
    controller_sha256: str
    rendered_schedule: FileAsset | None = None
    inference_canary_reference: FileAsset | None = None

    def verify(self) -> None:
        FileAsset(self.path, self.sha256).verify()
        self.route_registry.verify()
        self.route.verify()
        manifest = read_policy_manifest(self.policy_manifest.path, expected_manifest_sha256=self.policy_manifest.sha256)
        if manifest["policy_sha256"] != self.policy_sha256:
            raise ValueError("policy SHA256 differs from manifest content identity")
        if not self.checkpoint_path.is_dir():
            raise ValueError("checkpoint must be an explicit local directory")
        controller_identity(expected_sha256=self.controller_sha256)
        for asset in (self.rendered_schedule, self.inference_canary_reference):
            if asset is not None:
                asset.verify()


def controller_identity(*, expected_sha256=None) -> dict:
    paths = {name: pathlib.Path(__file__).with_name(name + ".py") for name in ("actions", "controller")}
    return verify_controller_identity(
        files={name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()},
        actual_files=paths, control_hz=20, chunk_size=10, expected_sha256=expected_sha256)


def load_policy_spec(path: pathlib.Path) -> PolicySpec:
    path = pathlib.Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported policy spec schema")
    if type(payload.get("policy_seed")) is not int or payload["policy_seed"] != 0:
        raise ValueError("campaign requires policy_seed=0")
    bank = payload["ordered_episode_indices"]
    if (not isinstance(bank, list) or not bank or any(type(i) is not int or i < 0 for i in bank)
            or len(set(bank)) != len(bank)):
        raise ValueError("ordered bank must contain unique nonnegative integer episode IDs")
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("prompt must be a nonempty benchmark instruction")
    commit = payload["expected_evaluator_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("expected_evaluator_commit must be a full git SHA")

    def asset(value: dict) -> FileAsset:
        result = FileAsset((path.parent / value["path"]).resolve(), value["sha256"])
        result.verify()
        return result

    registry = asset(payload["route_registry"])
    route = select_route(load_routes(registry.path), suite=payload["suite"], task_id=payload["task_id"], arm=payload["arm"])
    schedule = asset(payload["rendered_schedule"]) if payload.get("rendered_schedule") else None
    if route.method in ("uniform", "sup") and schedule is not None:
        raise ValueError("uniform route must not require a phase schedule")
    if route.method == "strider":
        if schedule is None:
            raise ValueError("STRIDER route requires rendered_schedule")
        from .strider_client import load_candidate_schedule

        table = json.loads(schedule.path.read_text())
        if (table.get("task_suite_name") != route.identity.suite
                or table.get("task_id") != route.identity.task_id
                or table.get("bddl_sha256") != route.identity.bddl_sha256):
            raise ValueError("rendered schedule suite/task/BDDL binding mismatch")
        load_candidate_schedule(schedule.path, checkpoint_phases=tuple(
            f"task_{route.identity.task_id:02d}:{phase}" for phase in route.phase_names), fast_stride=3)
    reference = asset(payload["inference_canary_reference"]) if payload.get("inference_canary_reference") else None
    spec = PolicySpec(path=path, sha256=file_sha256(path), route_registry=registry, route=route,
                      ordered_episode_indices=tuple(bank), prompt=payload["prompt"],
                      run_dir=(path.parent / payload["run_dir"]).resolve(), expected_evaluator_commit=commit,
                      policy_manifest=asset(payload["policy_manifest"]),
                      checkpoint_path=(path.parent / payload["checkpoint_path"]).resolve(),
                      policy_sha256=payload["policy_sha256"], controller_sha256=payload["controller_sha256"],
                      rendered_schedule=schedule, inference_canary_reference=reference)
    spec.verify()
    return spec


def expected_metadata(spec: PolicySpec) -> dict:
    spec.verify()
    result = {
        "method": ("strider_phase_candidate" if spec.route.method == "strider" else
                   "sup_comparator" if spec.route.method == "sup" else "uniform_reference"),
        "base_policy": BASE_POLICY,
        "base_checkpoint": BASE_CHECKPOINT,
        "evaluator_commit": spec.expected_evaluator_commit,
        "campaign": {
            "schema": SCHEMA,
            "spec_sha256": spec.sha256,
            "route_registry_sha256": spec.route_registry.sha256,
            "task_identity": dataclasses.asdict(spec.route.identity),
            "arm": spec.route.arm,
            "ordered_episode_indices": list(spec.ordered_episode_indices),
            "policy_seed": 0,
            "evaluation_seed": 7,
            "adapter_sha256": file_sha256(pathlib.Path(__file__)),
            "controller_contract": CONTROLLER,
            "policy_manifest_sha256": spec.policy_manifest.sha256,
            "checkpoint_path": str(spec.checkpoint_path),
            "policy_sha256": spec.policy_sha256,
            "controller_sha256": spec.controller_sha256,
        },
    }
    if spec.route.method == "sup":
        from .sup_identity import expected_selector_metadata

        result["sup_selector"] = expected_selector_metadata(spec.route)
    if spec.route.method == "strider":
        result.update(phase_checkpoint_sha256=spec.route.phase_checkpoint.sha256,
                      phase_dataset_manifest_sha256=spec.route.dataset_manifest.sha256,
                      phase_schedule_sha256=spec.rendered_schedule.sha256, fast_stride=3,
                      authority="AI_CANDIDATE_NOT_HUMAN_ANNOTATION", requires_human_confirmation=True,
                      speed_schedule_status="candidate_not_promoted")
    return result


def validate_metadata(metadata: dict, spec: PolicySpec) -> None:
    expected = expected_metadata(spec)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches or metadata.get("campaign_inference_canary_passed") is not True:
        raise ValueError(f"campaign policy handshake mismatch: {mismatches}; requires passed inference canary")


def preflight(spec: PolicySpec, *, repo: pathlib.Path) -> dict:
    """File/commit validation only: no policy creation, server or inference."""
    metadata = expected_metadata(spec)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec.expected_evaluator_commit:
        raise ValueError("evaluator commit mismatch")
    return {"passed": True, "physics_steps": 0, "inference_calls": 0, "metadata_contract": metadata}


class SeededPolicy:
    def __init__(self, policy, spec: PolicySpec, *, verified_policy: VerifiedPolicy, rng_key=None, rng_data=None):
        spec.verify()
        manifest = read_policy_manifest(spec.policy_manifest.path, expected_manifest_sha256=spec.policy_manifest.sha256)
        expected = VerifiedPolicy(spec.checkpoint_path, spec.policy_sha256, spec.policy_manifest.sha256,
                                  manifest["file_count"], manifest["total_bytes"])
        if verified_policy != expected:
            raise ValueError("loaded policy verification receipt differs from policy spec")
        if not hasattr(policy, "_rng") or getattr(policy, "_is_pytorch_model", True):
            raise ValueError("campaign RNG protocol requires the JAX pi0.5 policy")
        if rng_key is None or rng_data is None:
            import jax

            rng_key, rng_data = jax.random.key, jax.random.key_data
        self.policy = policy
        self.spec = spec
        self._key, self._data = rng_key, rng_data
        self._episode = None
        self._seen = []
        self._plan = 0
        self._poisoned = False
        self._metadata = expected_metadata(spec)
        if not spec.run_dir.is_dir():
            raise FileNotFoundError("run_dir must already be reserved by the dispatcher")
        self._log = (spec.run_dir / "inference-rng.jsonl").open("x", buffering=1)
        try:
            self._canary()
        except BaseException:
            self._log.close()
            raise

    def _canary(self) -> None:
        obs = {"observation/image": np.full((224, 224, 3), 127, dtype=np.uint8),
               "observation/wrist_image": np.full((224, 224, 3), 127, dtype=np.uint8),
               "observation/state": np.zeros(8, dtype=np.float64), "prompt": self.spec.prompt}
        actions = []
        for _ in range(3):
            self.policy._rng = self._key(0)
            actions.append(np.asarray(self.policy.infer(obs)["actions"]).copy())
        if any(a.shape != (10, 7) or not np.isfinite(a).all() for a in actions):
            raise ValueError("invalid pi0.5 canary action shape or finite values")
        same = np.array_equal(actions[1], actions[2])
        receipt = dict(passed=same, within_process_exact=same,
                       within_process_max_delta=float(np.max(np.abs(actions[1] - actions[2]))),
                       physical_steps=0, input_hash=request_hash(obs), actions_hash=array_hash(actions[2]),
                       warmup_calls=1, repeat_calls=2)
        if self.spec.inference_canary_reference is not None:
            ref = np.load(self.spec.inference_canary_reference.path, allow_pickle=False)
            if ref.shape != actions[2].shape:
                raise ValueError("canary reference shape mismatch")
            matched = bool(ref.dtype == actions[2].dtype and np.array_equal(ref, actions[2]))
            receipt.update(reference=str(self.spec.inference_canary_reference.path),
                           cross_process_max_delta=float(np.max(np.abs(ref - actions[2]))),
                           cross_process_tolerance=0.0, cross_process_exact=matched,
                           cross_process_passed=matched)
            receipt["passed"] = bool(same and matched)
        with (self.spec.run_dir / "inference-canary-actions.npy").open("xb") as stream:
            np.save(stream, actions[2], allow_pickle=False)
        with (self.spec.run_dir / "inference-canary.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        if not receipt["passed"]:
            raise ValueError("inference repeatability canary failed")
        self._metadata["campaign_inference_canary_passed"] = True

    @property
    def metadata(self) -> dict:
        return {**self.policy.metadata, **self._metadata}

    def infer(self, obs: dict) -> dict:
        if self._poisoned:
            raise RuntimeError("policy request failed previously; refusing ambiguous RNG continuation")
        obs = dict(obs)
        tag = obs.pop(TAG, None)
        expected = dataclasses.asdict(self.spec.route.identity)
        expected["task"] = expected.pop("task_id")
        expected["seed"] = 0
        if not isinstance(tag, dict) or set(tag) != {*expected, "episode", "plan"}:
            raise ValueError("missing or invalid diagnostic episode tag")
        if any(type(tag[k]) is not type(value) or tag[k] != value for k, value in expected.items()):
            raise ValueError("diagnostic suite/task/BDDL/seed mismatch")
        ep, index = tag["episode"], tag["plan"]
        if type(ep) is not int or type(index) is not int or index < 0:
            raise ValueError("episode and plan must be integer indices")
        new_episode = ep != self._episode
        if new_episode:
            if (index != 0 or len(self._seen) >= len(self.spec.ordered_episode_indices)
                    or ep != self.spec.ordered_episode_indices[len(self._seen)]):
                raise ValueError("episode bank/order or initial plan mismatch")
        elif index != self._plan:
            raise ValueError("nonsequential policy plan")
        if obs.get("prompt") != self.spec.prompt:
            raise ValueError("policy prompt differs from registered benchmark instruction")
        if new_episode:
            self.policy._rng = self._key(0)
            self._episode, self._plan = ep, 0
            self._seen.append(ep)
        before = np.asarray(self._data(self.policy._rng)).tolist()
        input_hash = request_hash(obs)
        fields = {key: ({"string": value} if isinstance(value, str) else
                        {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
                         "sha256": array_hash(value)}) for key, value in obs.items()}
        try:
            out = dict(self.policy.infer(obs))
            actions = np.asarray(out["actions"])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("invalid pi0.5 raw actions")
            receipt = dict(**expected, episode=ep, plan=index, rng_before=before,
                           rng_after=np.asarray(self._data(self.policy._rng)).tolist(),
                           request_sha256=input_hash, actions_sha256=array_hash(actions),
                           actions_shape=list(actions.shape), actions_dtype=str(actions.dtype),
                           request_fields=fields, raw_actions=actions.tolist())
            self._log.write(json.dumps(receipt) + "\n")
        except BaseException:
            self._poisoned = True
            raise
        out[RECEIPT] = receipt
        self._plan += 1
        return out

    def close(self) -> None:
        self._log.close()


class EpisodePolicy:
    """Tag requests without changing the underlying policy's observation payload."""
    def __init__(self, policy, spec: PolicySpec, episode: int):
        self.policy, self.spec, self.episode = policy, spec, episode
        self.index = 0
        self.receipt = None

    def infer(self, obs: dict) -> dict:
        if TAG in obs:
            raise ValueError("diagnostic episode policy is already wrapped")
        tag = {"suite": self.spec.route.identity.suite, "task": self.spec.route.identity.task_id,
               "bddl_sha256": self.spec.route.identity.bddl_sha256,
               "episode": self.episode, "plan": self.index, "seed": 0}
        response = self.policy.infer({**obs, TAG: tag})
        receipt = response.get(RECEIPT, {})
        if any(receipt.get(key) != value for key, value in tag.items()):
            raise ValueError("server RNG receipt does not match episode request")
        if receipt.get("request_sha256") != request_hash(obs) or receipt.get("actions_sha256") != array_hash(response["actions"]):
            raise ValueError("server request/action receipt mismatch")
        self.receipt = receipt
        self.index += 1
        return response


def wrap_episode_policy(policy, spec: PolicySpec, episode: int) -> EpisodePolicy:
    """Session hook; call once per registered episode, before its first inference."""
    if type(episode) is not int or episode not in spec.ordered_episode_indices:
        raise ValueError("episode is outside the registered bank")
    if isinstance(policy, EpisodePolicy):
        raise ValueError("diagnostic episode policy is already wrapped")
    return EpisodePolicy(policy, spec, episode)


def install_evaluator_policy(main, spec: PolicySpec) -> None:
    """Install handshake-before-env and episode tags on the scoped evaluator module.

    Call before eval_speed_baseline. Do not install the older EpisodePolicy
    instrumentation as well. Other observation/action telemetry can remain.
    """
    if getattr(main, "_campaign_policy_installed", False):
        raise ValueError("campaign policy adapter already installed")
    original_client = main.websocket_client_policy.WebsocketClientPolicy
    original_env, original_episode = main._get_libero_env, main._run_episode
    original_eval = main.eval_speed_baseline
    connected = False
    seen_episodes = []

    def client(*args, **kwargs):
        nonlocal connected
        instance = original_client(*args, **kwargs)
        validate_metadata(instance.get_server_metadata(), spec)
        connected = True
        return instance

    def evaluate(args):
        if (args.task_suite_name != spec.route.identity.suite or args.task_start != spec.route.identity.task_id
                or args.task_count != 1 or args.num_trials_per_task != len(spec.ordered_episode_indices)
                or args.seed != 7 or args.method != spec.route.method
                or (args.method == "uniform" and args.uniform_stride != spec.route.uniform_stride)):
            raise ValueError("evaluator suite/task/method/bank/seed differs from campaign contract")
        return original_eval(args)

    def environment(*args, **kwargs):
        if not connected:
            raise RuntimeError("policy handshake must pass before environment creation")
        return original_env(*args, **kwargs)

    def episode(**kwargs):
        args = kwargs["args"]
        if (args.task_suite_name != spec.route.identity.suite or kwargs["task_id"] != spec.route.identity.task_id
                or args.method != spec.route.method
                or (args.method == "uniform" and args.uniform_stride != spec.route.uniform_stride)):
            raise ValueError("evaluator suite/task/method/stride differs from policy route")
        if (len(seen_episodes) >= len(spec.ordered_episode_indices)
                or kwargs["episode_index"] != spec.ordered_episode_indices[len(seen_episodes)]):
            raise ValueError("evaluator episode bank/order mismatch before reset")
        seen_episodes.append(kwargs["episode_index"])
        kwargs["policy"] = wrap_episode_policy(kwargs["policy"], spec, kwargs["episode_index"])
        return original_episode(**kwargs)

    main.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=client)
    main._get_libero_env, main._run_episode = environment, episode
    main.eval_speed_baseline = evaluate
    main._campaign_policy_installed = True


def serve(spec: PolicySpec, *, repo: pathlib.Path, port: int) -> None:
    preflight(spec, repo=repo)
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError("invalid server port")
    # These imports may initialize accelerators. Only the explicit serve path uses them.
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    base_policy, verified = create_verified_policy(
        spec.policy_manifest.path, expected_manifest_sha256=spec.policy_manifest.sha256,
        checkpoint_path=spec.checkpoint_path, expected_policy_sha256=spec.policy_sha256,
        loader=create_trained_policy, train_config=get_config(BASE_POLICY))
    policy = SeededPolicy(base_policy, spec, verified_policy=verified)
    try:
        WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=port, metadata=policy.metadata).serve_forever()
    finally:
        policy.close()
