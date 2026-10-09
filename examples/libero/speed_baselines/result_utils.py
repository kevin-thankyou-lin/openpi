from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any


def episode_indices(*, initial_state_count: int, episode_start: int, num_trials: int, task_id: int) -> range:
    episode_stop = episode_start + num_trials
    if episode_stop > initial_state_count:
        raise ValueError(
            f"task {task_id} has {initial_state_count} initial states, "
            f"cannot evaluate episode indices {episode_start}..{episode_stop - 1}"
        )
    return range(episode_start, episode_stop)


def atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [episode for episode in episodes if episode["success"]]
    return {
        "total_episodes": len(episodes),
        "total_successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "mean_success_env_steps": (
            sum(episode["env_steps"] for episode in successes) / len(successes) if successes else None
        ),
        "mean_success_source_actions": (
            sum(episode["source_actions_consumed"] for episode in successes) / len(successes)
            if successes
            else None
        ),
    }


def summarize_tasks(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return episode-weighted metrics for each task, ordered by task id."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(int(episode["task_id"]), []).append(episode)

    task_summaries = []
    for task_id, task_episodes in sorted(grouped.items()):
        descriptions = {str(episode["task_description"]) for episode in task_episodes}
        if len(descriptions) != 1:
            raise ValueError(f"task {task_id} has inconsistent descriptions: {sorted(descriptions)}")
        task_summaries.append(
            {
                "task_id": task_id,
                "task_description": descriptions.pop(),
                **summarize_episodes(task_episodes),
            }
        )
    return task_summaries
