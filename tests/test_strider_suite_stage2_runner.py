from __future__ import annotations

import hashlib
import json

import pytest

from examples.libero.speed_baselines.strider_suite_canary_runner import AUTHORITY
from examples.libero.speed_baselines.strider_suite_canary_runner import CAMPAIGN_SCHEMA as CANARY_SCHEMA
from examples.libero.speed_baselines.strider_suite_canary_runner import TaskRoute
from examples.libero.speed_baselines.strider_suite_stage2_runner import load_stage1_gate


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(tmp_path, task_id):
    dataset = tmp_path / f"dataset-{task_id}"
    dataset.mkdir()
    checkpoint = tmp_path / f"checkpoint-{task_id}.pt"
    checkpoint.write_bytes(f"checkpoint-{task_id}".encode())
    return TaskRoute(
        task_id=task_id,
        phase_names=("first", "second"),
        phase_checkpoint=checkpoint,
        phase_dataset=dataset,
        checkpoint_sha256=_sha256(checkpoint),
        ontology_sha256="unused-by-stage2-gate",
    )


def _canary(tmp_path, routes, successes_by_task, registry_sha256="a" * 64):
    campaign = tmp_path / "canary"
    campaign.mkdir()
    task_rows = []
    for route in routes:
        run_dir = campaign / f"task_{route.task_id:02d}_uniform2_canary5"
        videos = run_dir / "videos"
        videos.mkdir(parents=True)
        episodes = [
            {
                "task_id": route.task_id,
                "episode_index": index,
                "success": index < successes_by_task[route.task_id],
            }
            for index in range(5)
        ]
        results = run_dir / "results.json"
        results.write_text(json.dumps({"status": "complete", "episodes": episodes}))
        video_hashes = {}
        for index in range(5):
            video = videos / f"task_{route.task_id:02d}_episode_{index:02d}.mp4"
            video.write_bytes(f"video-{route.task_id}-{index}".encode())
            video_hashes[video.name] = _sha256(video)
        task_rows.append(
            {
                "task_id": route.task_id,
                "status": "complete",
                "trials": 5,
                "successes": successes_by_task[route.task_id],
                "results_sha256": _sha256(results),
                "video_sha256": video_hashes,
            }
        )
    (campaign / "campaign_receipt.json").write_text(
        json.dumps(
            {
                "schema": CANARY_SCHEMA,
                "status": "complete",
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
                "registry_sha256": registry_sha256,
                "seed": 7,
                "num_trials_per_task": 5,
                "initial_schedule": "uniform_2x",
                "task_ids": [route.task_id for route in routes],
                "tasks": task_rows,
            }
        )
    )
    return campaign


def test_stage2_gate_includes_only_three_to_five_success_canaries(tmp_path):
    routes = (_route(tmp_path, 4), _route(tmp_path, 5))
    campaign = _canary(tmp_path, routes, {4: 3, 5: 2})

    eligible = load_stage1_gate(campaign, routes=routes, registry_sha256="a" * 64, seed=7)

    assert set(eligible) == {4}
    assert eligible[4].successes == 3


def test_stage2_gate_rejects_source_result_hash_drift(tmp_path):
    routes = (_route(tmp_path, 4),)
    campaign = _canary(tmp_path, routes, {4: 5})
    results = campaign / "task_04_uniform2_canary5" / "results.json"
    results.write_text(results.read_text() + "\n")

    with pytest.raises(ValueError, match="outcome hash mismatch"):
        load_stage1_gate(campaign, routes=routes, registry_sha256="a" * 64, seed=7)
