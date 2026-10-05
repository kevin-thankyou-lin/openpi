from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any


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
