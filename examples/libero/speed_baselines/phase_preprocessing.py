"""Opt-in, checkpoint-bound phase preprocessing; never changes policy images.

The caller loads the trusted, SHA-pinned checkpoint once, supplies its payload
here, and constructs the predictor from that same payload. Historical selectors
without this contract retain their existing behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FEATURES = tuple([f"robot0.joint.{i}" for i in range(7)] + [f"robot0.gripper.{i}" for i in range(2)])
PREPROCESSING = {
    "schema": "strider-phase-preprocessing-v1",
    "image": {
        "dtype": "uint8",
        "layout": "HWC",
        "color": "RGB",
        "height": 84,
        "width": 84,
        "resize": "pillow-bilinear",
    },
    "proprio_feature_names": list(FEATURES),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PhaseInputContract:
    dataset_manifest_sha256: str
    checkpoint_sha256: str
    class_names: tuple[str, ...]
    history_length: int
    horizon: int

    def image(self, value: np.ndarray) -> np.ndarray:
        """Match dataset preparation on RGB bytes; do not normalize twice."""
        value = np.asarray(value)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
            raise ValueError("phase input must be nonempty uint8 HWC RGB")
        return np.asarray(Image.fromarray(value).resize((84, 84), Image.Resampling.BILINEAR)).copy()


def bind_phase_inputs(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    checkpoint_payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    phase_names: tuple[str, ...],
    history_length: int = 4,
    horizon: int = 10,
    checkpoint_bytes: bytes | None = None,
) -> PhaseInputContract:
    """Verify provenance/shape before starting an episode, without model inference.

    ``checkpoint_payload`` must be the payload loaded from the verified file,
    and must also be used to construct the predictor. This helper deliberately
    does not deserialize pickle or import torch.
    """
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("phase dataset manifest SHA mismatch")
    checkpoint_digest = (
        _sha(Path(checkpoint_path)) if checkpoint_bytes is None else hashlib.sha256(checkpoint_bytes).hexdigest()
    )
    if checkpoint_digest != expected_checkpoint_sha256:
        raise ValueError("phase checkpoint SHA mismatch")
    manifest = json.loads(raw)
    if manifest.get("identity") != dict(identity):
        raise ValueError("phase dataset scene identity mismatch")
    if manifest.get("preprocessing") != PREPROCESSING:
        raise ValueError("unsupported or absent explicit phase preprocessing")
    metadata = checkpoint_payload.get("metadata", {})
    if metadata.get("schema") != "strider-observation-history-subtask-checkpoint-v1":
        raise ValueError("unsupported phase checkpoint schema")
    if metadata.get("dataset_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("checkpoint was trained with another dataset manifest")
    names = tuple(row["name"] for row in manifest["classes"])
    if not names or len(set(names)) != len(names) or names != tuple(phase_names):
        raise ValueError("phase ontology order differs from dataset")
    if tuple(checkpoint_payload["class_names"]) != names:
        raise ValueError("checkpoint class order differs from dataset/ontology")
    config = checkpoint_payload["model_config"]
    expected = {
        "history_length": history_length,
        "horizon": horizon,
        "class_count": len(names),
        "proprio_dim": 9,
        "image_channels": 3,
    }
    if any(type(config.get(k)) is not int or config[k] != v for k, v in expected.items()):
        raise ValueError("checkpoint model shape differs from phase input contract")
    for key, width in (("image_mean", 3), ("image_std", 3), ("proprio_mean", 9), ("proprio_std", 9)):
        value = np.asarray(manifest["normalization"][key], dtype=np.float64)
        other = np.asarray(checkpoint_payload["normalization"][key], dtype=np.float64)
        if value.shape != (width,) or not np.isfinite(value).all() or not np.array_equal(value, other):
            raise ValueError(f"checkpoint normalization mismatch: {key}")
        if key.endswith("std") and np.any(value <= 0):
            raise ValueError(f"invalid normalization scale: {key}")
    return PhaseInputContract(expected_manifest_sha256, expected_checkpoint_sha256, names, history_length, horizon)
