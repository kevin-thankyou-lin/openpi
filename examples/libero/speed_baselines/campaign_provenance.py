"""Verify actual model and controller bytes before scientific campaign execution.

Policy content identity uses the existing 16-file campaign manifest convention.
Controller and phase identities are separate, explicit contracts; they do not
claim equivalence to older hashes that included different runtime files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any

POLICY_SCHEMA = "strider-base-policy-content-manifest-v1"
COMPOSITION = "controller-space sum-position left-compose-rotation last-gripper"


class ProvenanceError(ValueError):
    """Pinned scientific inputs differ from the files that would execute."""


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProvenanceError("expected lowercase SHA256")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProvenanceError("invalid relative file path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ProvenanceError("invalid relative file path")
    return value


def _regular_files(root: Path, *, python_only=False) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProvenanceError(f"symlink is not a sealed content file: {path}")
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        if python_only and path.suffix != ".py":
            continue
        files[path.relative_to(root).as_posix()] = path
    return files


@dataclasses.dataclass(frozen=True)
class VerifiedPolicy:
    checkpoint_path: Path
    policy_sha256: str
    manifest_sha256: str
    file_count: int
    total_bytes: int


def read_policy_manifest(manifest_path: Path, *, expected_manifest_sha256: str) -> dict:
    """Validate the pinned manifest and its content identity without reading weights."""
    data = Path(manifest_path).read_bytes()
    manifest_sha = hashlib.sha256(data).hexdigest()
    if manifest_sha != _sha(expected_manifest_sha256):
        raise ProvenanceError("policy manifest file SHA256 mismatch")
    manifest = json.loads(data)
    if manifest.get("schema") != POLICY_SCHEMA or manifest.get("policy") != "pi05_libero":
        raise ProvenanceError("unsupported base policy manifest")
    rows = manifest["files"]
    if not isinstance(rows, list) or not rows:
        raise ProvenanceError("empty or malformed policy files")
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "size_bytes", "sha256"}:
            raise ProvenanceError("malformed policy file record")
        names.append(_relative(row["relative_path"]))
        _sha(row["sha256"])
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 0 or names[-1].endswith(".lock"):
            raise ProvenanceError("invalid policy file size or excluded lock file")
    if names != sorted(set(names)):
        raise ProvenanceError("policy file records must be unique and sorted")
    content_sha = _canonical(rows)
    if content_sha != _sha(manifest["policy_sha256"]):
        raise ProvenanceError("policy content identity differs from file records")
    total_bytes = sum(row["size_bytes"] for row in rows)
    if manifest.get("file_count") != len(rows) or manifest.get("total_bytes") != total_bytes:
        raise ProvenanceError("policy manifest totals mismatch")
    return manifest


def verify_policy_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    checkpoint_path: Path,
    expected_policy_sha256: str | None = None,
) -> VerifiedPolicy:
    """Hash every model file, using the unchanged legacy content-hash definition.

    ``checkpoint_path`` is an explicit local directory, not the manifest's old
    provenance URI. A copied checkpoint is permitted only with identical content.
    Every regular file except ``*.lock`` must appear exactly once in the manifest.
    Returned ``checkpoint_path`` is the only path that the policy loader may use.
    """
    manifest = read_policy_manifest(manifest_path, expected_manifest_sha256=expected_manifest_sha256)
    rows = manifest["files"]
    names = [row["relative_path"] for row in rows]
    content_sha = manifest["policy_sha256"]
    if expected_policy_sha256 is not None and content_sha != _sha(expected_policy_sha256):
        raise ProvenanceError("policy content identity differs from required policy")
    total_bytes = manifest["total_bytes"]
    manifest_sha = expected_manifest_sha256
    root = Path(checkpoint_path).resolve(strict=True)
    if not root.is_dir():
        raise ProvenanceError("checkpoint must be an explicit local directory")
    actual = _regular_files(root)
    if set(actual) != set(names):
        raise ProvenanceError("policy file set differs: missing or extra checkpoint content")
    for row in rows:
        path = actual[row["relative_path"]]
        if path.stat().st_size != row["size_bytes"] or file_sha256(path) != row["sha256"]:
            raise ProvenanceError(f"policy file bytes differ: {row['relative_path']}")
    return VerifiedPolicy(root, content_sha, manifest_sha, len(rows), total_bytes)


def create_verified_policy(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    checkpoint_path: Path,
    loader: Callable,
    train_config: Any,
    expected_policy_sha256: str | None = None,
) -> tuple[Any, VerifiedPolicy]:
    """Verify then load that exact local path; no URI/cache/default-path fallback."""
    if getattr(train_config, "name", None) != "pi05_libero":
        raise ProvenanceError("verified campaign policy requires the pi05_libero training config")
    verified = verify_policy_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        checkpoint_path=checkpoint_path,
        expected_policy_sha256=expected_policy_sha256,
    )
    return loader(train_config, verified.checkpoint_path), verified


def verify_controller_identity(
    *,
    files: Mapping[str, Mapping],
    actual_files: Mapping[str, Path],
    control_hz: int,
    chunk_size: int,
    composition: str = COMPOSITION,
    expected_sha256: str | None = None,
) -> dict:
    """Bind composition code and its fixed timing contract independently of speed.

    Pass paths of the actual imported modules through ``actual_files``. Absolute
    deployment paths are recorded but excluded from the portable content identity.
    This hash is intentionally distinct from old omnibus runtime-source hashes.
    """
    if set(files) != {"actions", "controller"} or set(actual_files) != set(files):
        raise ProvenanceError("controller requires exact actions/controller source pins")
    if type(control_hz) is not int or control_hz != 20 or type(chunk_size) is not int or chunk_size != 10:
        raise ProvenanceError("unvalidated controller rate or policy chunk size")
    if composition != COMPOSITION:
        raise ProvenanceError("unvalidated action composition")
    sources, paths = {}, {}
    for name in sorted(files):
        pin = files[name]
        pinned = Path(pin["path"]).resolve(strict=True)
        actual = Path(actual_files[name]).resolve(strict=True)
        if pinned != actual:
            raise ProvenanceError(f"pinned {name} is not the actual imported module")
        expected = _sha(pin["sha256"])
        if file_sha256(actual) != expected:
            raise ProvenanceError(f"controller source bytes differ: {name}")
        sources[name] = expected
        paths[name] = str(actual)
    identity = {
        "schema": "strider-sup-controller-content-v1",
        "sources": sources,
        "control_hz": control_hz,
        "chunk_size": chunk_size,
        "composition": composition,
    }
    value = _canonical(identity)
    if expected_sha256 is not None and value != _sha(expected_sha256):
        raise ProvenanceError("controller identity mismatch")
    return {"identity": identity, "sha256": value, "actual_paths": paths}


def verify_phase_runtime(
    *,
    root: Path,
    files: Mapping[str, str],
    actual_root: Path,
    expected_sha256: str | None = None,
) -> dict:
    """Pin every Python source under the actual predictor import source root.

    ``root`` and ``actual_root`` must resolve identically (typically phase_repo/src).
    Include every .py file beneath that source root, including package initializers;
    an unlisted added helper is rejected. Uniform baselines need no phase runtime.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or root != Path(actual_root).resolve(strict=True):
        raise ProvenanceError("phase root differs from actual predictor import root")
    if not isinstance(files, Mapping) or not files:
        raise ProvenanceError("explicit phase runtime files required")
    expected = {_relative(k): _sha(v) for k, v in files.items()}
    if any(not name.endswith(".py") for name in expected):
        raise ProvenanceError("phase runtime entries must be Python sources")
    actual = _regular_files(root, python_only=True)
    if set(actual) != set(expected):
        raise ProvenanceError("phase runtime Python source set differs")
    for name, path in actual.items():
        if file_sha256(path) != expected[name]:
            raise ProvenanceError(f"phase runtime source bytes differ: {name}")
    identity = {"schema": "strider-phase-runtime-content-v1", "files": dict(sorted(expected.items()))}
    value = _canonical(identity)
    if expected_sha256 is not None and value != _sha(expected_sha256):
        raise ProvenanceError("phase runtime identity mismatch")
    return {"identity": identity, "sha256": value, "actual_root": str(root)}
