from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client


class SupSelectorClient:
    """Persistent client for the supplied SuP predict-k websocket protocol."""

    def __init__(self, host: str, port: int, timeout_seconds: float = 10.0) -> None:
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.timeout_seconds = timeout_seconds
        self._packer = msgpack_numpy.Packer()
        self._connection, self.metadata = self._wait_for_server()

    def _wait_for_server(self):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                connection = websockets.sync.client.connect(
                    self.uri,
                    open_timeout=self.timeout_seconds,
                    compression=None,
                    max_size=None,
                )
                return connection, msgpack_numpy.unpackb(connection.recv())
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"SuP selector did not become ready at {self.uri}")
                logging.info("Waiting for SuP selector at %s", self.uri)
                time.sleep(1)

    def predict_k(self, state: np.ndarray, actions: np.ndarray) -> int:
        request = {"type": "predict_k", "s_env": state, "A": actions}
        self._connection.send(self._packer.pack(request))
        payload = self._connection.recv()
        if isinstance(payload, str):
            raise RuntimeError(f"SuP selector error: {payload}")
        response: dict[str, Any] = msgpack_numpy.unpackb(payload)
        if response.get("status") not in (None, "success"):
            raise RuntimeError(f"SuP selector rejected request: {response}")
        selected = int(response["predicted_k"])
        if selected not in (1, 2):
            raise ValueError(f"SuP selector returned unsupported k={selected}")
        return selected

    def close(self) -> None:
        self._connection.close()
