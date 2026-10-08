"""Route task-specific STRIDER predictors through sequential LIBERO canaries."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
from typing import Any

from .result_utils import atomic_write_json
from .strider_search_runner import Args as SearchArgs
from .strider_search_runner import run as run_search_evaluator

REGISTRY_SCHEMA = "strider-libero-task-route-registry-v1"
CAMPAIGN_SCHEMA = "strider-libero-suite-canary-campaign-v1"
AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"


@dataclasses.dataclass(frozen=True)
class TaskRoute:
    task_id: int
    phase_names: tuple[str, ...]
    phase_checkpoint: pathlib.Path
    phase_dataset: pathlib.Path
    checkpoint_sha256: str
    ontology_sha256: str


@dataclasses.dataclass(frozen=True)
class Args:
    campaign_dir: pathlib.Path
    registry: pathlib.Path
    phase_repo: pathlib.Path
    server_python: pathlib.Path
    task_ids: str = "0,1,4,5,6,7,8,9"
    num_trials: int = 5
    seed: int = 7
    initial_speed: float = 2.0
    server_host: str = "127.0.0.1"
    server_port: int = 8013


def load_routes(path: pathlib.Path) -> dict[int, TaskRoute]:
    path = pathlib.Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"unsupported task route registry schema: {payload.get('schema')!r}")
    if payload.get("authority") != AUTHORITY or payload.get("requires_human_confirmation") is not True:
        raise ValueError("task route registry must retain the AI-candidate review gate")
    routes: dict[int, TaskRoute] = {}
    for raw in payload.get("tasks", []):
        task_id = int(raw["task_id"])
        if task_id in routes:
            raise ValueError(f"duplicate task route {task_id}")
        route = TaskRoute(
            task_id=task_id,
            phase_names=tuple(str(name) for name in raw["phase_names"]),
            phase_checkpoint=pathlib.Path(raw["phase_checkpoint"]),
            phase_dataset=pathlib.Path(raw["phase_dataset"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            ontology_sha256=str(raw["ontology_sha256"]),
        )
        _validate_route(route)
        routes[task_id] = route
    if not routes:
        raise ValueError("task route registry has no tasks")
    return routes


def select_routes(routes: dict[int, TaskRoute], task_ids: str) -> tuple[TaskRoute, ...]:
    selected_ids = tuple(int(value.strip()) for value in task_ids.split(",") if value.strip())
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("task_ids must contain unique comma-separated task IDs")
    missing = sorted(set(selected_ids) - set(routes))
    if missing:
        raise ValueError(f"task route registry is missing tasks {missing}")
    return tuple(routes[task_id] for task_id in selected_ids)


def run(args: Args) -> None:
    if args.num_trials != 5:
        raise ValueError("suite canary requires exactly five trials per task")
    if args.initial_speed != 2.0:
        raise ValueError("first accelerated challenger must be uniform 2x")
    if not pathlib.Path(args.phase_repo).is_dir() or not pathlib.Path(args.server_python).is_file():
        raise FileNotFoundError("phase_repo and server_python must exist")
    routes = select_routes(load_routes(args.registry), args.task_ids)
    campaign_dir = pathlib.Path(args.campaign_dir)
    if campaign_dir.exists():
        raise FileExistsError(f"refusing to replace or resume canary campaign: {campaign_dir}")
    campaign_dir.mkdir(parents=True)

    receipt = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "running",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "registry": str(pathlib.Path(args.registry).resolve()),
        "registry_sha256": _sha256(args.registry),
        "evaluator_commit": _git_head(),
        "phase_repo_commit": _git_head(args.phase_repo),
        "task_ids": [route.task_id for route in routes],
        "num_trials_per_task": args.num_trials,
        "seed": args.seed,
        "initial_schedule": "uniform_2x",
        "execution": "strictly_sequential",
        "tasks": [],
    }
    atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)

    for route in routes:
        run_dir = campaign_dir / f"task_{route.task_id:02d}_uniform2_canary5"
        run_dir.mkdir()
        schedule = dict.fromkeys(route.phase_names, args.initial_speed)
        _write_json_once(run_dir / "schedule.json", schedule)
        _write_json_once(
            run_dir / "config.json",
            {
                "schema": "strider-libero-canary-run-config-v1",
                "task_id": route.task_id,
                "speed_schedule": schedule,
                "requested_rollouts": args.num_trials,
                "seed": args.seed,
                "phase_checkpoint": str(route.phase_checkpoint.resolve()),
                "phase_checkpoint_sha256": route.checkpoint_sha256,
                "phase_dataset": str(route.phase_dataset.resolve()),
                "ontology_sha256": route.ontology_sha256,
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
                seed=args.seed,
                task_id=route.task_id,
                phase_names=",".join(route.phase_names),
            )
        )
        task_result = _validate_completed_run(run_dir, route.task_id, args.num_trials)
        receipt["tasks"].append(task_result)
        atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)

    receipt["status"] = "complete"
    atomic_write_json(campaign_dir / "campaign_receipt.json", receipt)


def _validate_route(route: TaskRoute) -> None:
    if route.task_id < 0 or not route.phase_names or len(set(route.phase_names)) != len(route.phase_names):
        raise ValueError(f"invalid task route {route.task_id}")
    checkpoint = route.phase_checkpoint
    ontology_path = route.phase_dataset / "ontology_v1.json"
    if not checkpoint.is_file() or not ontology_path.is_file():
        raise FileNotFoundError(f"task {route.task_id} route artifacts do not exist")
    ontology = json.loads(ontology_path.read_text())
    if ontology.get("task_id") != route.task_id or tuple(ontology.get("phases", [])) != route.phase_names:
        raise ValueError(f"task {route.task_id} ontology identity mismatch")
    if ontology.get("authority") != AUTHORITY or ontology.get("requires_human_confirmation") is not True:
        raise ValueError(f"task {route.task_id} ontology lost its review gate")
    if _sha256(checkpoint) != route.checkpoint_sha256 or _sha256(ontology_path) != route.ontology_sha256:
        raise ValueError(f"task {route.task_id} route hash mismatch")


def _validate_completed_run(run_dir: pathlib.Path, task_id: int, expected_trials: int) -> dict[str, Any]:
    results = json.loads((run_dir / "results.json").read_text())
    episodes = results.get("episodes", [])
    if results.get("status") != "complete" or len(episodes) != expected_trials:
        raise ValueError(f"task {task_id} canary is not complete")
    if {(row.get("task_id"), row.get("episode_index")) for row in episodes} != {
        (task_id, index) for index in range(expected_trials)
    }:
        raise ValueError(f"task {task_id} canary episode identity mismatch")
    videos = sorted((run_dir / "videos").glob("*.mp4"))
    if len(videos) != expected_trials or any(path.stat().st_size == 0 for path in videos):
        raise ValueError(f"task {task_id} canary video gate failed")
    return {
        "task_id": task_id,
        "status": "complete",
        "results_sha256": _sha256(run_dir / "results.json"),
        "successes": sum(bool(row.get("success")) for row in episodes),
        "trials": expected_trials,
        "video_count": len(videos),
        "video_sha256": {path.name: _sha256(path) for path in videos},
        "summary": results.get("summary"),
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: pathlib.Path | None = None) -> str:
    command = ["git"]
    if path is not None:
        command.extend(["-C", str(path)])
    command.extend(["rev-parse", "HEAD"])
    return subprocess.check_output(command, text=True).strip()


def _write_json_once(path: pathlib.Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(f"refusing to replace different file: {path}")
        return
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(serialized)
    os.replace(partial, path)


if __name__ == "__main__":
    import tyro

    run(tyro.cli(Args))
