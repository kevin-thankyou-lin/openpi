"""Run one matched LIBERO uniform-speed reference with a private policy server."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
import socket
import subprocess
import time

import tyro

from .main import Args as EvalArgs
from .main import eval_speed_baseline


ALLOWED_SPEEDS = frozenset({1.0, 1.5, 2.0, 3.0})


@dataclasses.dataclass(frozen=True)
class Args:
    run_dir: pathlib.Path
    server_python: pathlib.Path
    task_id: int
    speed: float
    server_host: str = "127.0.0.1"
    server_port: int = 8011
    server_start_timeout_seconds: float = 180.0
    num_trials: int = 5
    seed: int = 7


def run(args: Args) -> None:
    _validate(args)
    repo = pathlib.Path(__file__).resolve().parents[3]
    args.run_dir.mkdir(parents=True, exist_ok=False)
    server_log_path = args.run_dir / "server.log"
    command = (
        str(args.server_python),
        str(repo / "scripts" / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(args.server_port),
    )
    with server_log_path.open("x") as server_log:
        server = subprocess.Popen(
            command,
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
            eval_speed_baseline(
                EvalArgs(
                    method="uniform",
                    run_dir=args.run_dir,
                    host=args.server_host,
                    port=args.server_port,
                    task_suite_name="libero_10",
                    task_start=args.task_id,
                    task_count=1,
                    num_trials_per_task=args.num_trials,
                    seed=args.seed,
                    uniform_stride=args.speed,
                    save_strider_telemetry=True,
                )
            )
        finally:
            _stop_process_group(server)


def _validate(args: Args) -> None:
    if not args.server_python.is_file():
        raise FileNotFoundError(f"server_python does not exist: {args.server_python}")
    if args.task_id < 0 or args.num_trials < 1:
        raise ValueError("task_id must be nonnegative and num_trials must be positive")
    if float(args.speed) not in ALLOWED_SPEEDS:
        raise ValueError(f"speed must be one of {sorted(ALLOWED_SPEEDS)}")
    if not 0 < args.server_port < 65536:
        raise ValueError("server_port must be within 1..65535")
    if _port_is_open(args.server_host, args.server_port):
        raise RuntimeError(f"server port {args.server_host}:{args.server_port} is already in use")


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
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(f"policy server exited {returncode} before readiness:\n{tail}")
        if _port_is_open(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"policy server did not open {host}:{port}")


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


if __name__ == "__main__":
    run(tyro.cli(Args))
