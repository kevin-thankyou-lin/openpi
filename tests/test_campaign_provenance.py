"""Only temporary model/source files; no GPU, checkpoint load, or simulator."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from examples.libero.speed_baselines import campaign_provenance as p


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_manifest(path, value):
    path.write_text(json.dumps(value))
    return p.file_sha256(path)


@pytest.fixture
def policy_files(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "params").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "params/weights").write_bytes(b"exact weights")
    (root / "assets/norm.json").write_text('{"mean":0}')
    rows = [
        {"relative_path": name, "size_bytes": (root / name).stat().st_size, "sha256": p.file_sha256(root / name)}
        for name in ["assets/norm.json", "params/weights"]
    ]
    manifest = {
        "schema": p.POLICY_SCHEMA,
        "policy": "pi05_libero",
        "checkpoint_path": "/original/machine/provenance/path",
        "uri": "gs://openpi-assets/checkpoints/pi05_libero",
        "files": rows,
        "policy_sha256": canonical(rows),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
    }
    path = tmp_path / "manifest.json"
    digest = save_manifest(path, manifest)
    return root, path, digest, manifest


def test_legacy_identity_and_explicit_loader_path(policy_files):
    root, path, digest, manifest = policy_files
    (root / "runtime.lock").write_text("excluded")
    calls = []

    def loader(config, checkpoint):
        calls.append((config.name, checkpoint))
        return "policy"

    policy, verified = p.create_verified_policy(
        path,
        expected_manifest_sha256=digest,
        checkpoint_path=root,
        loader=loader,
        train_config=SimpleNamespace(name="pi05_libero"),
    )
    assert policy == "policy"
    assert calls == [("pi05_libero", root.resolve())]
    assert verified.policy_sha256 == manifest["policy_sha256"]
    assert verified.file_count == 2
    assert verified.total_bytes == manifest["total_bytes"]


def test_lightweight_read_does_not_require_checkpoint(policy_files):
    root, path, digest, manifest = policy_files
    (root / "params/weights").unlink()
    assert p.read_policy_manifest(path, expected_manifest_sha256=digest) == manifest
    with pytest.raises(p.ProvenanceError, match="file set"):
        p.verify_policy_manifest(path, expected_manifest_sha256=digest, checkpoint_path=root)


@pytest.mark.parametrize("mutation", ["extra", "missing", "corrupt", "same_size_corrupt", "symlink"])
def test_bad_checkpoint_refuses_before_loader(policy_files, mutation):
    root, path, digest, _ = policy_files
    if mutation == "extra":
        (root / "extra").write_text("not pinned")
    elif mutation == "missing":
        (root / "params/weights").unlink()
    elif mutation == "symlink":
        (root / "link").symlink_to(root / "params/weights")
    else:
        (root / "params/weights").write_bytes(b"WRONG weights" if mutation == "same_size_corrupt" else b"wrong")
    calls = []
    with pytest.raises(p.ProvenanceError, match=r"file set|bytes differ|symlink"):
        p.create_verified_policy(
            path,
            expected_manifest_sha256=digest,
            checkpoint_path=root,
            loader=lambda *args: calls.append(args),
            train_config=SimpleNamespace(name="pi05_libero"),
        )
    assert not calls


@pytest.mark.parametrize("mutation", ["claimed_hash", "order", "duplicate", "traversal", "totals", "row_fields"])
def test_lightweight_reader_rejects_wrong_identity_and_shape(policy_files, mutation):
    _, path, _, manifest = policy_files
    if mutation == "claimed_hash":
        manifest["policy_sha256"] = "0" * 64
    elif mutation == "order":
        manifest["files"].reverse()
    elif mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif mutation == "traversal":
        manifest["files"][0]["relative_path"] = "../outside"
    elif mutation == "totals":
        manifest["total_bytes"] += 1
    else:
        manifest["files"][0]["extra"] = "bad"
    digest = save_manifest(path, manifest)
    with pytest.raises(p.ProvenanceError):
        p.read_policy_manifest(path, expected_manifest_sha256=digest)


def test_file_manifest_and_required_content_hash_both_enforced(policy_files):
    root, path, digest, _ = policy_files
    with pytest.raises(p.ProvenanceError, match="manifest file SHA"):
        p.read_policy_manifest(path, expected_manifest_sha256="0" * 64)
    with pytest.raises(p.ProvenanceError, match="required policy"):
        p.verify_policy_manifest(
            path, expected_manifest_sha256=digest, checkpoint_path=root, expected_policy_sha256="0" * 64
        )


@pytest.fixture
def controller_files(tmp_path):
    actual = {key: tmp_path / f"{key}.py" for key in ("actions", "controller")}
    for key, path in actual.items():
        path.write_text(f"# {key}\n")
    pins = {key: {"path": str(path), "sha256": p.file_sha256(path)} for key, path in actual.items()}
    return {"files": pins, "actual_files": actual, "control_hz": 20, "chunk_size": 10}


def test_controller_identity_stable_across_deployment_paths(controller_files, tmp_path):
    first = p.verify_controller_identity(**controller_files)
    other = tmp_path / "copy"
    other.mkdir()
    copied = {}
    for key, value in controller_files["actual_files"].items():
        dest = other / value.name
        dest.write_bytes(value.read_bytes())
        copied[key] = dest
    pins = {key: {"path": str(path), "sha256": p.file_sha256(path)} for key, path in copied.items()}
    second = p.verify_controller_identity(files=pins, actual_files=copied, control_hz=20, chunk_size=10)
    assert first["sha256"] == second["sha256"]
    assert first["actual_paths"] != second["actual_paths"]


@pytest.mark.parametrize("mutation", ["source", "actual_path", "rate", "chunk", "composition", "identity"])
def test_controller_wrong_source_or_contract_rejected(controller_files, mutation, tmp_path):
    kwargs = controller_files
    if mutation == "source":
        kwargs["actual_files"]["actions"].write_text("changed")
    elif mutation == "actual_path":
        copied = tmp_path / "copy.py"
        copied.write_bytes(kwargs["actual_files"]["actions"].read_bytes())
        kwargs["actual_files"]["actions"] = copied
    elif mutation == "rate":
        kwargs["control_hz"] = 25
    elif mutation == "chunk":
        kwargs["chunk_size"] = 20
    elif mutation == "composition":
        kwargs["composition"] = "first gripper"
    else:
        kwargs["expected_sha256"] = "0" * 64
    with pytest.raises(p.ProvenanceError):
        p.verify_controller_identity(**kwargs)


@pytest.fixture
def phase_files(tmp_path):
    root = tmp_path / "phase/src"
    (root / "strider").mkdir(parents=True)
    (root / "strider/__init__.py").write_text("")
    (root / "strider/predict.py").write_text("# predictor\n")
    files = {path.relative_to(root).as_posix(): p.file_sha256(path) for path in root.rglob("*.py")}
    return {"root": root, "actual_root": root, "files": files}


def test_phase_all_python_sources_verified(phase_files):
    receipt = p.verify_phase_runtime(**phase_files)
    assert receipt["identity"]["files"] == phase_files["files"]
    assert p.verify_phase_runtime(**phase_files, expected_sha256=receipt["sha256"]) == receipt


@pytest.mark.parametrize("mutation", ["added_helper", "modified", "missing", "wrong_root", "wrong_identity"])
def test_phase_runtime_mismatch_rejected(phase_files, mutation, tmp_path):
    if mutation == "added_helper":
        (phase_files["root"] / "helper.py").write_text("new")
    elif mutation == "modified":
        (phase_files["root"] / "strider/predict.py").write_text("changed")
    elif mutation == "missing":
        (phase_files["root"] / "strider/__init__.py").unlink()
    elif mutation == "wrong_root":
        phase_files["actual_root"] = tmp_path
    else:
        phase_files["expected_sha256"] = "0" * 64
    with pytest.raises(p.ProvenanceError):
        p.verify_phase_runtime(**phase_files)


def test_loader_requires_exact_pi05_libero_config(policy_files):
    root, path, digest, _ = policy_files
    calls = []
    with pytest.raises(p.ProvenanceError, match="pi05_libero training config"):
        p.create_verified_policy(
            path,
            expected_manifest_sha256=digest,
            checkpoint_path=root,
            loader=lambda *a: calls.append(a),
            train_config=SimpleNamespace(name="other"),
        )
    assert not calls
