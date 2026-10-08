from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
from typing import Any

from .strider_client import STRIDER_ALLOWED_SPEEDS

TASK_ID = 2
FAST_STRIDE = 3
AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"
SCHEDULE_SCHEMA = "strider-libero-task2-fine-phase-schedule-v1"
PHASES = (
    "stove_approach",
    "stove_toggle",
    "moka_approach",
    "moka_acquire",
    "moka_transport",
    "moka_place",
)
ALLOWED_SPEEDS = STRIDER_ALLOWED_SPEEDS


@dataclasses.dataclass(frozen=True)
class Args:
    run_dir: pathlib.Path
    schedule: pathlib.Path
    phase_checkpoint: pathlib.Path
    phase_repo: pathlib.Path
    server_python: pathlib.Path
    server_host: str = "127.0.0.1"
    server_port: int = 8010
    server_start_timeout_seconds: float = 180.0
    num_trials: int = 5
    seed: int = 7
    task_id: int = TASK_ID
    phase_names: str = ",".join(PHASES)


def render_libero_schedule(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    task_id: int = TASK_ID,
    phases: tuple[str, ...] = PHASES,
) -> dict[str, Any]:
    """Validate STRIDER's plain schedule and render one LIBERO task schema."""

    source = pathlib.Path(source)
    raw = json.loads(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("STRIDER schedule must be a JSON object")
    if set(raw) != set(phases):
        missing = sorted(set(phases) - set(raw))
        extra = sorted(set(raw) - set(phases))
        raise ValueError(f"task schedule phase mismatch: missing={missing} extra={extra}")

    speeds: dict[str, float] = {}
    for phase in phases:
        value = raw[phase]
        if isinstance(value, bool) or not isinstance(value, (int, float)):  # noqa: UP038
            raise ValueError(f"speed for {phase!r} must be numeric")
        speed = float(value)
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed for {phase!r} must be one of {sorted(ALLOWED_SPEEDS)}, got {value!r}")
        speeds[phase] = speed

    payload = {
        "schema": (
            SCHEDULE_SCHEMA if task_id == TASK_ID and phases == PHASES else "strider-libero-task-fine-phase-schedule-v1"
        ),
        "status": "AI_CANDIDATE_NOT_PROMOTED_OR_EVALUATED",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "task_id": task_id,
        "candidate_speed_set": sorted(ALLOWED_SPEEDS),
        "maximum_stride": FAST_STRIDE,
        "source_schedule": str(source.resolve()),
        "source_schedule_sha256": _sha256(source),
        "phase_schedule": [[f"task_{task_id:02d}:{phase}", speeds[phase]] for phase in phases],
    }
    _write_json_once(destination, payload)
    return payload


def validate_search_run_config(config_path: pathlib.Path, *, schedule_path: pathlib.Path, num_trials: int) -> None:
    """Require the evaluator request to match STRIDER's reserved run contract."""

    config = json.loads(pathlib.Path(config_path).read_text())
    schedule = json.loads(pathlib.Path(schedule_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("search config must be a JSON object")
    if config.get("speed_schedule") != schedule:
        raise ValueError("config speed_schedule does not match schedule.json")
    if config.get("requested_rollouts") != num_trials:
        raise ValueError(
            "config requested_rollouts does not match --num-trials: "
            f"{config.get('requested_rollouts')!r} != {num_trials}"
        )


def build_server_command(args: Args, *, rendered_schedule: pathlib.Path, commit: str) -> tuple[str, ...]:
    repo = pathlib.Path(__file__).resolve().parents[3]
    return (
        str(args.server_python),
        str(repo / "scripts" / "serve_strider_libero.py"),
        "--phase-checkpoint",
        str(args.phase_checkpoint),
        "--phase-schedule",
        str(rendered_schedule),
        "--expected-evaluator-commit",
        commit,
        "--port",
        str(args.server_port),
        "--fast-stride",
        str(FAST_STRIDE),
    )


def run(args: Args) -> None:
    _validate_args(args)
    repo = pathlib.Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.schedule.resolve().parent != args.run_dir.resolve():
        raise ValueError("schedule must be inside the reserved run directory")
    validate_search_run_config(
        args.run_dir / "config.json",
        schedule_path=args.schedule,
        num_trials=args.num_trials,
    )
    rendered_schedule = args.run_dir / "libero_schedule.json"
    phases = tuple(item.strip() for item in args.phase_names.split(",") if item.strip())
    render_libero_schedule(
        args.schedule,
        rendered_schedule,
        task_id=args.task_id,
        phases=phases,
    )
    server_log_path = args.run_dir / "server.log"
    server_command = build_server_command(args, rendered_schedule=rendered_schedule, commit=commit)

    if _port_is_open(args.server_host, args.server_port):
        raise RuntimeError(f"server port {args.server_host}:{args.server_port} is already in use")

    with server_log_path.open("x") as server_log:
        server = subprocess.Popen(
            server_command,
            cwd=repo,
            env=os.environ.copy(),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            _wait_for_server(
                server,
                host=args.server_host,
                port=args.server_port,
                timeout_seconds=args.server_start_timeout_seconds,
                log_path=server_log_path,
            )
            from .main import Args as EvalArgs
            from .main import eval_speed_baseline

            eval_speed_baseline(
                EvalArgs(
                    method="strider",
                    run_dir=args.run_dir,
                    host=args.server_host,
                    port=args.server_port,
                    task_suite_name="libero_10",
                    task_start=args.task_id,
                    task_count=1,
                    num_trials_per_task=args.num_trials,
                    seed=args.seed,
                    fast_stride=FAST_STRIDE,
                    strider_checkpoint=args.phase_checkpoint,
                    strider_phase_repo=args.phase_repo,
                    strider_schedule=rendered_schedule,
                    strider_device="cpu",
                    save_strider_telemetry=True,
                )
            )
        finally:
            _stop_process_group(server)


def _validate_args(args: Args) -> None:
    for name, path in (
        ("schedule", args.schedule),
        ("phase_checkpoint", args.phase_checkpoint),
        ("phase_repo", args.phase_repo),
        ("server_python", args.server_python),
    ):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.num_trials < 1:
        raise ValueError("num_trials must be positive")
    if not 0 < args.server_port < 65536:
        raise ValueError("server_port must be within 1..65535")
    if args.server_start_timeout_seconds <= 0:
        raise ValueError("server_start_timeout_seconds must be positive")
    if args.task_id < 0:
        raise ValueError("task_id must be nonnegative")
    phases = tuple(item.strip() for item in args.phase_names.split(",") if item.strip())
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("phase_names must contain unique comma-separated names")


def _wait_for_server(
    process: subprocess.Popen,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    log_path: pathlib.Path,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"policy server exited {returncode} before readiness:\n{_tail(log_path)}")
        if _port_is_open(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"policy server did not open {host}:{port}:\n{_tail(log_path)}")


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _tail(path: pathlib.Path, *, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<server log unavailable>"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: pathlib.Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(f"refusing to replace different rendered schedule: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(serialized)
    os.replace(temporary, path)


if __name__ == "__main__":
    import tyro

    run(tyro.cli(Args))
