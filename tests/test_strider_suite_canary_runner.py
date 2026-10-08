from __future__ import annotations

import hashlib
import json

import pytest

from examples.libero.speed_baselines.strider_suite_canary_runner import AUTHORITY
from examples.libero.speed_baselines.strider_suite_canary_runner import REGISTRY_SCHEMA
from examples.libero.speed_baselines.strider_suite_canary_runner import load_routes
from examples.libero.speed_baselines.strider_suite_canary_runner import select_routes


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    ontology = dataset / "ontology_v1.json"
    ontology.write_text(
        json.dumps(
            {
                "task_id": 4,
                "phases": ["first", "second"],
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
            }
        )
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA,
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
                "tasks": [
                    {
                        "task_id": 4,
                        "phase_names": ["first", "second"],
                        "phase_checkpoint": str(checkpoint),
                        "phase_dataset": str(dataset),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "ontology_sha256": _sha256(ontology),
                    }
                ],
            }
        )
    )
    return registry


def test_task_route_registry_is_hash_and_identity_pinned(tmp_path):
    routes = load_routes(_registry(tmp_path))
    selected = select_routes(routes, "4")
    assert selected[0].task_id == 4
    assert selected[0].phase_names == ("first", "second")


def test_task_route_registry_rejects_hash_drift(tmp_path):
    registry = _registry(tmp_path)
    payload = json.loads(registry.read_text())
    payload["tasks"][0]["checkpoint_sha256"] = "0" * 64
    registry.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_routes(registry)


def test_task_route_selection_rejects_missing_task(tmp_path):
    routes = load_routes(_registry(tmp_path))
    with pytest.raises(ValueError, match="missing tasks"):
        select_routes(routes, "4,9")
