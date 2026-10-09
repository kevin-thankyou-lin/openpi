"""Run the registered episode-5..9 STRIDER continuation for canary survivors."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from .result_utils import atomic_write_json
from .strider_search_runner import Args as SearchArgs
from .strider_search_runner import run as run_search_evaluator
from .strider_suite_canary_runner import AUTHORITY
from .strider_suite_canary_runner import CAMPAIGN_SCHEMA as CANARY_CAMPAIGN_SCHEMA
from .strider_suite_canary_runner import TaskRoute
from .strider_suite_canary_runner import _git_head
from .strider_suite_canary_runner import _sha256
from .strider_suite_canary_runner import _write_json_once
from .strider_suite_canary_runner import load_routes
from .strider_suite_canary_runner import select_routes

CAMPAIGN_SCHEMA = "strider-libero-suite-uniform2-stage2-campaign-v1"
EPISODE_START = 5
EPISODE_COUNT = 5


@dataclasses.dataclass(frozen=True)
class Stage1Task:
    task_id: int
    successes: int
    results_sha256: str
    video_sha256: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Args:
    campaign_dir: pathlib.Path
    canary_campaign: pathlib.Path
    registry: pathlib.Path
    phase_repo: pathlib.Path
    server_python: pathlib.Path
    task_ids: str = "0,1,4,5,6,7,8,9"
    num_trials: int = EPISODE_COUNT
    episode_start: int = EPISODE_START
    seed: int = 7
    initial_speed: float = 2.0
    server_host: str = "127.0.0.1"
    server_port: int = 8013


def run(args: Args) -> None:
    if args.num_trials != EPISODE_COUNT or args.episode_start != EPISODE_START:
        raise ValueError("stage 2 requires exactly episode indices 5..9")
    if args.initial_speed != 2.0:
        raise ValueError("stage 2 must preserve the uniform 2x canary schedule")
    if not pathlib.Path(args.phase_repo).is_dir() or not pathlib.Path(args.server_python).is_file():
        raise FileNotFoundError("phase_repo and server_python must exist")

    routes = select_routes(load_routes(args.registry), args.task_ids)
    registry_sha256 = _sha256(args.registry)
    stage1 = load_stage1_gate(
        args.canary_campaign,
        routes=routes,
        registry_sha256=registry_sha256,
        seed=args.seed,
    )
    eligible_routes = tuple(route for route in routes if route.task_id in stage1)
    excluded_task_ids = [route.task_id for route in routes if route.task_id not in stage1]
    if not eligible_routes:
        raise ValueError("no canary task reached the registered 3/5 continuation boundary")

    campaign_dir = pathlib.Path(args.campaign_dir)
    if campaign_dir.exists():
        raise FileExistsError(f"refusing to replace or resume stage-2 campaign: {campaign_dir}")
    campaign_dir.mkdir(parents=True)

    canary_receipt = pathlib.Path(args.canary_campaign) / "campaign_receipt.json"
    receipt: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "running",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "registry": str(pathlib.Path(args.registry).resolve()),
        "registry_sha256": registry_sha256,
        "evaluator_commit": _git_head(),
        "phase_repo_commit": _git_head(args.phase_repo),
        "source_canary_campaign": str(pathlib.Path(args.canary_campaign).resolve()),
        "source_canary_receipt_sha256": _sha256(canary_receipt),
        "source_canary_tasks": [dataclasses.asdict(stage1[route.task_id]) for route in eligible_routes],
        "requested_task_ids": [route.task_id for route in routes],
        "eligible_task_ids": [route.task_id for route in eligible_routes],
        "excluded_task_ids": excluded_task_ids,
        "stage1_episode_indices": list(range(0, EPISODE_START)),
        "stage2_episode_indices": list(range(EPISODE_START, EPISODE_START + EPISODE_COUNT)),
        "new_rollouts_per_task": EPISODE_COUNT,
        "seed": args.seed,
        "schedule": "uniform_2x_frozen_from_stage1",
        "execution": "strictly_sequential",
        "gate_after_10": {
            "0_to_8_successes": "reject_for_promotion",
            "9_to_10_successes": "eligible_for_fresh_added_10",
        },
        "prohibited_during_this_campaign": ["uniform_3x", "matched_final_evaluation", "final_test_seeds"],
        "tasks": [],
    }
    atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)

    for route in eligible_routes:
        run_dir = campaign_dir / f"task_{route.task_id:02d}_uniform2_stage2_ep05_09"
        run_dir.mkdir()
        schedule = dict.fromkeys(route.phase_names, args.initial_speed)
        _write_json_once(run_dir / "schedule.json", schedule)
        _write_json_once(
            run_dir / "config.json",
            {
                "schema": "strider-libero-stage2-run-config-v1",
                "task_id": route.task_id,
                "speed_schedule": schedule,
                "requested_rollouts": args.num_trials,
                "episode_start": args.episode_start,
                "episode_indices": list(range(args.episode_start, args.episode_start + args.num_trials)),
                "seed": args.seed,
                "phase_checkpoint": str(route.phase_checkpoint.resolve()),
                "phase_checkpoint_sha256": route.checkpoint_sha256,
                "phase_dataset": str(route.phase_dataset.resolve()),
                "ontology_sha256": route.ontology_sha256,
                "source_canary_results_sha256": stage1[route.task_id].results_sha256,
                "source_canary_successes": stage1[route.task_id].successes,
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
                "speed_schedule_status": "candidate_not_promoted",
            },
        )
        run_search_evaluator(
            SearchArgs(
                run_dir=run_dir,
                schedule=run_dir / "schedule.json",
                phase_checkpoint=route.phase_checkpoint,
                phase_repo=args.phase_repo,
                server_python=args.server_python,
                server_host=args.server_host,
                server_port=args.server_port,
                num_trials=args.num_trials,
                episode_start=args.episode_start,
                seed=args.seed,
                task_id=route.task_id,
                phase_names=",".join(route.phase_names),
            )
        )
        task_result = _validate_completed_run(
            run_dir,
            route.task_id,
            episode_start=args.episode_start,
            expected_trials=args.num_trials,
            stage1=stage1[route.task_id],
        )
        receipt["tasks"].append(task_result)
        atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)

    receipt["status"] = "complete"
    atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)


def load_stage1_gate(
    canary_campaign: pathlib.Path,
    *,
    routes: tuple[TaskRoute, ...],
    registry_sha256: str,
    seed: int,
) -> dict[int, Stage1Task]:
    campaign = pathlib.Path(canary_campaign)
    receipt_path = campaign / "campaign_receipt.json"
    payload = json.loads(receipt_path.read_text())
    requested_ids = [route.task_id for route in routes]
    if payload.get("schema") != CANARY_CAMPAIGN_SCHEMA or payload.get("status") != "complete":
        raise ValueError("source canary campaign must be terminal and use the registered schema")
    if payload.get("registry_sha256") != registry_sha256:
        raise ValueError("source canary registry hash does not match stage-2 registry")
    if payload.get("seed") != seed or payload.get("num_trials_per_task") != EPISODE_COUNT:
        raise ValueError("source canary seed or trial count does not match stage-2 contract")
    if payload.get("initial_schedule") != "uniform_2x" or payload.get("task_ids") != requested_ids:
        raise ValueError("source canary task order or schedule does not match stage-2 contract")

    rows = payload.get("tasks", [])
    by_id = {int(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(requested_ids):
        raise ValueError("source canary task receipts are missing or duplicated")

    eligible: dict[int, Stage1Task] = {}
    for route in routes:
        row = by_id[route.task_id]
        if row.get("status") != "complete" or row.get("trials") != EPISODE_COUNT:
            raise ValueError(f"task {route.task_id} source canary receipt is incomplete")
        run_dir = campaign / f"task_{route.task_id:02d}_uniform2_canary5"
        results_path = run_dir / "results.json"
        results = json.loads(results_path.read_text())
        expected_keys = {(route.task_id, index) for index in range(EPISODE_START)}
        actual_keys = {(item.get("task_id"), item.get("episode_index")) for item in results.get("episodes", [])}
        successes = sum(bool(item.get("success")) for item in results.get("episodes", []))
        if results.get("status") != "complete" or actual_keys != expected_keys:
            raise ValueError(f"task {route.task_id} source canary episode identity mismatch")
        if successes != row.get("successes") or _sha256(results_path) != row.get("results_sha256"):
            raise ValueError(f"task {route.task_id} source canary outcome hash mismatch")
        videos = sorted((run_dir / "videos").glob("*.mp4"))
        current_video_hashes = {path.name: _sha256(path) for path in videos}
        if len(videos) != EPISODE_COUNT or current_video_hashes != row.get("video_sha256"):
            raise ValueError(f"task {route.task_id} source canary video receipt mismatch")
        if 3 <= successes <= 5:
            eligible[route.task_id] = Stage1Task(
                task_id=route.task_id,
                successes=successes,
                results_sha256=row["results_sha256"],
                video_sha256=current_video_hashes,
            )
    return eligible


def _validate_completed_run(
    run_dir: pathlib.Path,
    task_id: int,
    *,
    episode_start: int,
    expected_trials: int,
    stage1: Stage1Task,
) -> dict[str, Any]:
    results = json.loads((run_dir / "results.json").read_text())
    episodes = results.get("episodes", [])
    expected_keys = {(task_id, index) for index in range(episode_start, episode_start + expected_trials)}
    if results.get("status") != "complete" or len(episodes) != expected_trials:
        raise ValueError(f"task {task_id} stage-2 continuation is not complete")
    if {(row.get("task_id"), row.get("episode_index")) for row in episodes} != expected_keys:
        raise ValueError(f"task {task_id} stage-2 episode identity mismatch")
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    if len(videos) != expected_trials or any(path.stat().st_size == 0 for path in videos):
        raise ValueError(f"task {task_id} stage-2 video gate failed")
    stage2_successes = sum(bool(row.get("success")) for row in episodes)
    cumulative_successes = stage1.successes + stage2_successes
    return {
        "task_id": task_id,
        "status": "complete",
        "episode_indices": list(range(episode_start, episode_start + expected_trials)),
        "results_sha256": _sha256(run_dir / "results.json"),
        "stage1_successes": stage1.successes,
        "stage2_successes": stage2_successes,
        "cumulative_10_successes": cumulative_successes,
        "gate_decision": (
            "eligible_for_fresh_added_10" if cumulative_successes >= 9 else "reject_for_promotion"
        ),
        "video_count": len(videos),
        "video_sha256": {path.name: _sha256(path) for path in videos},
        "summary": results.get("summary"),
    }


if __name__ == "__main__":
    import tyro

    run(tyro.cli(Args))
