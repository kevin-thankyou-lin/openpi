from __future__ import annotations

import collections
import dataclasses
import fcntl
import hashlib
import json
import logging
import math
import pathlib
import subprocess
from typing import Literal, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tqdm
import tyro

from .actions import SupActionComposer
from .actions import native_slices
from .actions import sail_precision_slices
from .actions import uniform_cadence_slices
from .actions import uniform_slices
from .controller import apply_sup_controller_patches
from .result_utils import atomic_write_json
from .result_utils import episode_indices
from .result_utils import summarize_episodes
from .result_utils import summarize_tasks
from .selector_client import SupSelectorClient
from .strider_client import STRIDER_AUTHORITY
from .strider_client import StriderPhaseSelector
from .strider_client import validate_strider_server_metadata
from .telemetry import record_from_obs
from .telemetry import write_episode

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SUP_SUPPLEMENT_SHA256 = "93f5b6a7615b94de6f3dfdec6dbe1fdc92332d26220dbf5a3ce6d7e3e7c3e2a1"


@dataclasses.dataclass(frozen=True)
class Args:
    method: Literal["native", "uniform", "sup", "sail", "strider"]
    run_dir: pathlib.Path
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite_name: str = "libero_10"
    task_start: int = 0
    task_count: Optional[int] = None
    episode_start: int = 0
    num_trials_per_task: int = 1
    seed: int = 7
    resize_size: int = 224
    chunk_size: int = 10
    num_steps_wait: int = 10
    uniform_stride: float = 2.0
    selector_host: str = "127.0.0.1"
    selector_port: int = 8888
    precision_threshold: float = 0.5
    precision_key: str = "precision_scores"
    sail_head_index: int = 2
    sail_expected_tau: float = 0.01
    fast_stride: int = 2
    strider_checkpoint: Optional[pathlib.Path] = None
    strider_phase_repo: Optional[pathlib.Path] = None
    strider_schedule: Optional[pathlib.Path] = None
    strider_device: str = "cuda"
    resume: bool = False
    save_strider_telemetry: bool = False


def eval_speed_baseline(args: Args) -> None:
    _validate_args(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.run_dir / "run.lock").open("w")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another evaluator owns {args.run_dir}") from error

    config = dataclasses.asdict(args)
    config.pop("resume")
    config["run_dir"] = str(args.run_dir)
    for name in ("strider_checkpoint", "strider_phase_repo", "strider_schedule"):
        if config[name] is not None:
            config[name] = str(config[name])
    config["openpi_commit"] = _git_head()
    config["libero_commit"] = _git_head(pathlib.Path("third_party/libero"))
    config["sup_supplement_sha256"] = SUP_SUPPLEMENT_SHA256
    config["implementation_sha256"] = _implementation_hashes()
    config["controller_contract"] = (
        "native" if args.method == "native" else "sup-supplement-unclip-panda-speed-0.02-mujoco-step"
    )
    config["action_composition"] = "controller-space sum-position left-compose-rotation last-gripper"
    if args.method == "strider":
        config["strider_checkpoint_sha256"] = _sha256(args.strider_checkpoint)
        config["strider_schedule_sha256"] = _sha256(args.strider_schedule)
        config["strider_phase_repo_commit"] = _git_head(args.strider_phase_repo)
        config["authority"] = STRIDER_AUTHORITY
        config["requires_human_confirmation"] = True
        config["speed_schedule_status"] = "candidate_not_promoted"
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_sha256"] = config_hash
    manifest_path = args.run_dir / "manifest.json"
    results_path = args.run_dir / "results.json"
    _initialize_or_validate_manifest(manifest_path, config, resume=args.resume)

    existing = _load_results(results_path) if args.resume else None
    episodes: list[dict] = list(existing.get("episodes", [])) if existing else []
    completed_keys = {(row["task_id"], row["episode_index"]) for row in episodes}

    np.random.seed(args.seed)
    if args.method != "native":
        apply_sup_controller_patches()
    composer = SupActionComposer.from_robosuite()
    # First inference can include accelerator compilation and exceed the websocket
    # library's 20-second keepalive timeout. Inference itself remains blocking, so
    # disable protocol pings for this local deterministic evaluator connection.
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, ping_interval=None)
    policy_metadata = policy.get_server_metadata()
    selector = SupSelectorClient(args.selector_host, args.selector_port) if args.method == "sup" else None
    strider = (
        StriderPhaseSelector(
            checkpoint=args.strider_checkpoint,
            phase_repo=args.strider_phase_repo,
            schedule=args.strider_schedule,
            fast_stride=args.fast_stride,
            device=args.strider_device,
        )
        if args.method == "strider"
        else None
    )
    if args.method == "sail":
        _validate_sail_metadata(policy_metadata, args)
    if args.method == "strider":
        validate_strider_server_metadata(
            policy_metadata,
            checkpoint_sha256=_sha256(args.strider_checkpoint),
            schedule_sha256=_sha256(args.strider_schedule),
            evaluator_commit=_git_head(),
            fast_stride=args.fast_stride,
        )

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    videos_dir = args.run_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    logging.info("Task suite: %s; method: %s; config: %s", args.task_suite_name, args.method, config_hash)

    try:
        task_stop = task_suite.n_tasks if args.task_count is None else args.task_start + args.task_count
        task_ids = range(args.task_start, min(task_stop, task_suite.n_tasks))
        for task_id in tqdm.tqdm(task_ids):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, args.seed)
            try:
                selected_episode_indices = episode_indices(
                    initial_state_count=len(initial_states),
                    episode_start=args.episode_start,
                    num_trials=args.num_trials_per_task,
                    task_id=task_id,
                )
                for episode_index in selected_episode_indices:
                    key = (task_id, episode_index)
                    if key in completed_keys:
                        continue
                    result = _run_episode(
                        env=env,
                        task_description=task_description,
                        initial_state=initial_states[episode_index],
                        task_id=task_id,
                        episode_index=episode_index,
                        args=args,
                        policy=policy,
                        composer=composer,
                        selector=selector,
                        strider=strider,
                        videos_dir=videos_dir,
                    )
                    episodes.append(result)
                    payload = {
                        "status": "running",
                        "config_sha256": config_hash,
                        "episodes": episodes,
                        "summary": summarize_episodes(episodes),
                        "task_summaries": summarize_tasks(episodes),
                    }
                    atomic_write_json(results_path, payload)
                    logging.info("TASK_RESULT %s", json.dumps(result, sort_keys=True))
            finally:
                env.close()
    finally:
        if selector is not None:
            selector.close()

    payload = {
        "status": "complete",
        "config_sha256": config_hash,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
        "task_summaries": summarize_tasks(episodes),
    }
    atomic_write_json(results_path, payload)
    logging.info("EVAL_COMPLETE %s", json.dumps(payload["summary"], sort_keys=True))


def _run_episode(
    *,
    env,
    task_description: str,
    initial_state: np.ndarray,
    task_id: int,
    episode_index: int,
    args: Args,
    policy,
    composer: SupActionComposer,
    selector: SupSelectorClient | None,
    strider: StriderPhaseSelector | None,
    videos_dir: pathlib.Path,
) -> dict:
    logging.info("Task %d: %s", task_id, task_description)
    env.reset()
    obs = env.set_init_state(initial_state)
    env_steps = 0
    source_actions_consumed = 0
    action_plan: collections.deque = collections.deque()
    decision_index = 0
    decisions: list[dict] = []
    phase_decisions: list[dict] = []
    frames: list[np.ndarray] = []
    telemetry_records: list[dict] = []
    done = False
    horizon = MAX_STEPS[args.task_suite_name]
    if strider is not None:
        strider.reset_episode()

    for _ in range(args.num_steps_wait):
        obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
        if done:
            break

    while env_steps < horizon and not done:
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, args.resize_size, args.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
        )
        if strider is not None:
            strider.observe(image, obs)
        if not action_plan:
            state = np.concatenate(
                [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
            )
            response = policy.infer(
                {
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
            )
            actions = np.asarray(response["actions"], dtype=np.float64)[: args.chunk_size]
            if strider is None:
                slices, decision = _schedule_actions(actions, response, state, args, composer, selector)
                action_plan.extend(slices)
            else:
                slices, decision = strider.plan(actions, task_id=task_id)
                action_plan.extend(slices)
            decision["decision_index"] = decision_index
            decision["env_step"] = env_steps
            decisions.append(decision)
            decision_index += 1
        action_slice = action_plan.popleft()
        if strider is not None:
            phase_decision = {}
            phase_decision.update(
                {
                    "env_step": env_steps,
                    **action_slice.metadata,
                    "source_stride": action_slice.stride,
                }
            )
            phase_decisions.append(phase_decision)
        if args.save_strider_telemetry:
            telemetry_records.append(
                record_from_obs(
                    obs,
                    action=action_slice.action,
                    env_step=env_steps,
                    source_stride=action_slice.stride,
                )
            )
        obs, _reward, done, _info = env.step(action_slice.action.tolist())
        env_steps += 1
        source_actions_consumed += action_slice.stride
        frames.append(image)

    success = bool(done)
    suffix = "success" if success else "failure"
    video_path = videos_dir / f"task_{task_id:02d}_episode_{episode_index:02d}_{suffix}.mp4"
    telemetry_path = None
    if frames:
        imageio.mimwrite(video_path, frames, fps=10)
        if args.save_strider_telemetry:
            telemetry_path = write_episode(
                args.run_dir / "strider_telemetry" / f"task_{task_id:02d}_episode_{episode_index:02d}",
                video_path=video_path,
                records=telemetry_records,
                metadata={
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_index": episode_index,
                    "success": success,
                },
            )
    result = {
        "task_id": task_id,
        "task_description": task_description,
        "episode_index": episode_index,
        "success": success,
        "env_steps": env_steps,
        "source_actions_consumed": source_actions_consumed,
        "decision_count": len(decisions),
        "decisions": decisions,
        "video": str(video_path),
    }
    if telemetry_path is not None:
        result["strider_telemetry"] = str(telemetry_path)
    if strider is not None:
        result["phase_decision_count"] = len(phase_decisions)
        result["phase_decisions"] = phase_decisions
    return result


def _schedule_actions(actions, response, state, args, composer, selector):
    if args.method == "native":
        return native_slices(actions), {"selected_k": 1, "slice_strides": [1] * len(actions)}
    if args.method == "uniform":
        if float(args.uniform_stride).is_integer():
            slices = uniform_slices(actions, stride=int(args.uniform_stride), composer=composer)
        else:
            slices = uniform_cadence_slices(actions, speed=args.uniform_stride, composer=composer)
        return slices, {
            "selected_k": args.uniform_stride,
            "requested_speed": args.uniform_stride,
            "slice_strides": [item.stride for item in slices],
        }
    if args.method == "sup":
        assert selector is not None
        selected_k = selector.predict_k(state, actions)
        slices = uniform_slices(actions, stride=selected_k, composer=composer)
        return slices, {"selected_k": selected_k, "slice_strides": [item.stride for item in slices]}
    scores = _extract_precision_scores(response, args.precision_key, args.sail_head_index, len(actions))
    slices = sail_precision_slices(
        actions,
        scores,
        threshold=args.precision_threshold,
        fast_stride=args.fast_stride,
        composer=composer,
    )
    return slices, {
        "selected_k": None,
        "precision_min": float(np.min(scores)),
        "precision_max": float(np.max(scores)),
        "precision_threshold": args.precision_threshold,
        "slice_strides": [item.stride for item in slices],
    }


def _extract_precision_scores(response, key: str, head_index: int, action_count: int) -> np.ndarray:
    if key not in response:
        raise KeyError(f"SAIL response missing {key!r}; keys={sorted(response)}")
    scores = np.asarray(response[key], dtype=np.float64)
    if scores.ndim == 2:
        if head_index < 0 or head_index >= scores.shape[1]:
            raise ValueError(f"SAIL head index {head_index} outside score shape {scores.shape}")
        scores = scores[:, head_index]
    scores = scores.reshape(-1)[:action_count]
    if len(scores) != action_count:
        raise ValueError(f"SAIL returned {len(scores)} scores for {action_count} actions")
    return scores


def _validate_sail_metadata(metadata: dict, args: Args) -> None:
    if metadata.get("method") != "sail_precision_head":
        raise ValueError(f"expected method=sail_precision_head, got metadata={metadata}")
    if int(metadata.get("fast_stride", -1)) != args.fast_stride:
        raise ValueError(f"SAIL fast_stride mismatch: metadata={metadata.get('fast_stride')} args={args.fast_stride}")
    precision_taus = metadata.get("precision_taus")
    if precision_taus is None:
        raise ValueError(f"SAIL metadata missing precision_taus: {metadata}")
    if args.sail_head_index < 0 or args.sail_head_index >= len(precision_taus):
        raise ValueError(f"SAIL head index {args.sail_head_index} outside precision_taus={precision_taus}")
    selected_tau = float(precision_taus[args.sail_head_index])
    if not math.isclose(selected_tau, args.sail_expected_tau, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"SAIL head-to-tau mismatch: head={args.sail_head_index} "
            f"metadata_tau={selected_tau} expected_tau={args.sail_expected_tau}"
        )


def _initialize_or_validate_manifest(path: pathlib.Path, config: dict, *, resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if not resume:
            raise FileExistsError(f"run manifest already exists: {path}")
        if existing != config:
            raise ValueError("resume configuration does not match immutable manifest")
    else:
        atomic_write_json(path, config)


def _load_results(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _get_libero_env(task, seed: int):
    task_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def _validate_args(args: Args) -> None:
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"unsupported task suite: {args.task_suite_name}")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if args.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if args.episode_start < 0:
        raise ValueError("episode_start must be nonnegative")
    if args.task_start < 0:
        raise ValueError("task_start must be nonnegative")
    if args.task_count is not None and args.task_count < 1:
        raise ValueError("task_count must be positive when provided")
    if args.method == "uniform" and float(args.uniform_stride) not in (1.0, 1.5, 2.0, 3.0):
        raise ValueError("registered LIBERO uniform candidate requires speed 1, 1.5, 2, or 3")
    if args.method in ("sup", "sail") and args.fast_stride != 2:
        raise ValueError("paper LIBERO accelerated methods require fast stride 2")
    if args.method == "strider":
        if args.fast_stride not in (2, 3):
            raise ValueError("registered Strider LIBERO candidate schedule requires fast stride 2 or 3")
        required = {
            "strider_checkpoint": args.strider_checkpoint,
            "strider_phase_repo": args.strider_phase_repo,
            "strider_schedule": args.strider_schedule,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Strider method requires {missing}")
        for name, path in required.items():
            if not pathlib.Path(path).exists():
                raise FileNotFoundError(f"{name} does not exist: {path}")


def _git_head(path: pathlib.Path = pathlib.Path(".")) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    package_dir = pathlib.Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package_dir.glob("*.py"))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_speed_baseline(tyro.cli(Args))
