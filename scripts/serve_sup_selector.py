"""Serve the frozen July SuP IQL selector through OpenPI's predict_k protocol.

Only selector inference runs here. No pi05 policy, environment or physics.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

WEIGHTS = {
    "0.0175": "93fedf6e3deb7e5a64cbbe613d58034c90bb01e23b60b57c66056468cd5a8214",
    "0.02": "c1b7f7fc8d33f0adb98dd97e3add3d3e94538ea7dafb65e08b2647fa7a36cc64",
}
EXPECTED_BYTES = 5586716
SOURCE_HASHES = {
    "speedtuning_rl/sup/iql.py": "a63a60b5da6c5501317079760745c8ef67f0c3ae08eb3247cd4b65dcf8b44657",
    "speedtuning_rl/sup/models.py": "c116069e4d28c6704962ad2968dac169934737f69efe12be717d8ecd8ead949c",
    "speedtuning_rl/sup/actions.py": "fafb1187a2dc1cf901d6aacd18ed9167dd607d571310ac9f4273537ffb8c2ae2",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path, epsilon):
    path = Path(path)
    if path.stat().st_size != EXPECTED_BYTES or sha256(path) != WEIGHTS[epsilon]:
        raise ValueError("checkpoint size/SHA256 differs from frozen July identity")


def validate_config(config):
    expected = {"state_dim": 8, "action_dim": 7, "hidden_dim": 256,
                "gru_layers": 3, "action_protocol": "protocol_v2"}
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"scheduler config mismatch: {key}")
    if tuple(config.k_values) != (1, 2):
        raise ValueError("scheduler k_values must be (1,2)")


def validate_source(root):
    for name, expected in SOURCE_HASHES.items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"July scheduler source identity mismatch: {name}")


def load_scheduler(root, checkpoint, device="cpu"):
    # The historical package __init__ imports unrelated dataset/robosuite tools.
    # Load the exact hash-verified inference modules in a private namespace.
    validate_source(root)
    package = types.ModuleType("_frozen_july_sup")
    package.__path__ = [str(Path(root).resolve() / "speedtuning_rl/sup")]
    sys.modules[package.__name__] = package
    scheduler_class = importlib.import_module("_frozen_july_sup.iql").IQLScheduler
    scheduler = scheduler_class.load(checkpoint, device=device)
    validate_config(scheduler.config)
    return scheduler


def metadata(checkpoint, epsilon, source_root):
    return {"method": "sup_iql_scheduler", "protocol": "predict_k_v1",
            "scheduler_sha256": WEIGHTS[epsilon], "scheduler_bytes": EXPECTED_BYTES,
            "epsilon": float(epsilon), "k_values": [1, 2],
            "state_dim": 8, "action_dim": 7, "chunk_size": 10, "control_hz": 20,
            "action_protocol": "protocol_v2",
            "action_composition": "controller-space sum-position left-compose-rotation last-gripper",
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "source_root": str(Path(source_root).resolve()),
            "source_sha256": {str(p): sha256(Path(source_root) / p) for p in (
                "speedtuning_rl/sup/iql.py", "speedtuning_rl/sup/models.py",
                "speedtuning_rl/sup/actions.py")}}


def predict(scheduler, request):
    if not isinstance(request, dict) or request.get("type") != "predict_k":
        raise ValueError("request must declare type=predict_k")
    state = np.asarray(request["s_env"], dtype=np.float32)
    actions = np.asarray(request["A"], dtype=np.float32)
    if state.shape != (8,) or actions.shape != (10, 7):
        raise ValueError("expected state(8,) and actions(10,7)")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("state/actions must be finite")
    selected, scores = scheduler.select_k(state, actions)
    if selected not in (1, 2) or set(scores) != {1, 2} or not np.isfinite(list(scores.values())).all():
        raise ValueError("invalid learned selector response")
    return {"status": "success", "predicted_k": int(selected),
            "q": {str(k): float(v) for k, v in scores.items()}}


async def serve(scheduler, identity, host, port, codec):
    import websockets

    async def handler(connection, *unused):
        packer = codec.Packer()
        await connection.send(packer.pack(identity))
        async for payload in connection:
            try:
                response = predict(scheduler, codec.unpackb(payload))
            except Exception as error:
                response = {"status": "error", "error": str(error)}
            await connection.send(packer.pack(response))

    async with websockets.serve(handler, host, port, compression=None, max_size=None):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epsilon", required=True, choices=tuple(WEIGHTS))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8888, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    validate_checkpoint(args.checkpoint, args.epsilon)
    validate_source(args.source_root)
    sys.path[:0] = [str(args.source_root.resolve()),
                   str(args.openpi_root.resolve() / "packages/openpi-client/src")]
    from openpi_client import msgpack_numpy

    scheduler = load_scheduler(args.source_root, args.checkpoint, args.device)
    identity = metadata(args.checkpoint, args.epsilon, args.source_root)
    print(json.dumps(identity, sort_keys=True), flush=True)
    asyncio.run(serve(scheduler, identity, args.host, args.port, msgpack_numpy))


if __name__ == "__main__":
    main()
