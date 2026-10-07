from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import socket
import subprocess

import tyro

from openpi.serving import websocket_policy_server

import serve_policy


AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"


@dataclasses.dataclass(frozen=True)
class Args:
    phase_checkpoint: pathlib.Path
    phase_schedule: pathlib.Path
    expected_evaluator_commit: str
    port: int = 8000
    fast_stride: int = 2


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(args: Args) -> dict:
    repo = pathlib.Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != args.expected_evaluator_commit:
        raise ValueError(
            f"evaluator commit mismatch: expected {args.expected_evaluator_commit}, got {commit}"
        )
    if args.fast_stride not in (2, 3):
        raise ValueError("registered Strider LIBERO schedule requires fast stride 2 or 3")
    schedule = json.loads(args.phase_schedule.read_text())
    schedule_schema = schedule.get("schema")
    supported_schemas = {
        "strider-libero-subtask-candidate-v1",
        "strider-libero-task2-fine-phase-schedule-v1",
    }
    if schedule_schema not in supported_schemas:
        raise ValueError(f"unsupported phase schedule schema: {schedule_schema!r}")
    if schedule_schema == "strider-libero-task2-fine-phase-schedule-v1":
        if schedule.get("authority") != AUTHORITY or schedule.get("requires_human_confirmation") is not True:
            raise ValueError("fine-phase schedule is missing its candidate-only authority gate")
        if schedule.get("status") != "AI_CANDIDATE_NOT_PROMOTED_OR_EVALUATED":
            raise ValueError("fine-phase schedule must remain unpromoted before evaluation")
    return {
        "method": "strider_phase_candidate",
        "base_policy": "pi05_libero",
        "base_checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "phase_checkpoint_sha256": sha256_file(args.phase_checkpoint),
        "phase_schedule_sha256": sha256_file(args.phase_schedule),
        "evaluator_commit": commit,
        "server_script_sha256": sha256_file(pathlib.Path(__file__)),
        "fast_stride": args.fast_stride,
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "speed_schedule_status": "candidate_not_promoted",
        "phase_schedule_schema": schedule_schema,
    }


def main(args: Args) -> None:
    metadata = build_metadata(args)
    policy = serve_policy.create_default_policy(serve_policy.EnvMode.LIBERO)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating Strider base-policy server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("STRIDER_SERVER_METADATA %s", json.dumps(metadata, sort_keys=True))
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
