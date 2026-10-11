import hashlib
import json
from pathlib import Path

import pytest

from examples.libero.speed_baselines import phase_capture_admission as a
from examples.libero.speed_baselines import prepare_suite_phases as builder


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": builder.sha(path)}


@pytest.fixture
def package(tmp_path, monkeypatch):
    pins = {"policy_sha256": "b" * 64, "controller_sha256": "c" * 64, "controller_contract": "fixture"}
    recipe = {"identity": {"policy_sha256": pins["policy_sha256"], "files": []}}
    recipe["recipe_sha256"] = hashlib.sha256(
        json.dumps(recipe["identity"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(a, "RECIPE", recipe["recipe_sha256"])
    units, audits, sources = [], [], []
    contracts = []
    for i, role in enumerate(("old", "new")):
        bank = [0] if i == 0 else list(range(1, 40))
        fit = [0] if i == 0 else list(range(1, 20))
        root = tmp_path / role
        contract = dict(pins, evaluator_commit=a.OLD_COMMIT if i == 0 else a.NEW_COMMIT)
        if i:
            contract["inference_cache_sha256"] = a.RECIPE
        contracts.append(contract)
        impl = {}
        for key in (
            "evaluator",
            "session",
            "policy_adapter",
            "actions",
            "controller",
            "scene_replay",
            "telemetry",
            "provenance",
        ):
            ref = write(root / (key + ".py"), {"synthetic_source": key, "role": role})
            impl[key] = ref
            sources.append(dict(ref, role=role, source_key=key))
        route = write(
            root / "routes.json",
            {
                "routes": [
                    {
                        "suite": "libero_spatial",
                        "task_id": 0,
                        "bddl": {"sha256": a.IDENTITY["bddl_sha256"]},
                        "method": "uniform",
                        "uniform_stride": 1,
                    }
                ]
            },
        )
        policy = write(
            root / "policy.json",
            dict(
                suite="libero_spatial",
                task_id=0,
                arm="uniform1",
                ordered_episode_indices=bank,
                prompt="fixture",
                route_registry=route,
                **pins,
            ),
        )
        receipt = {
            "sha256": a.OLD_SESSION_SHA256 if i == 0 else str(i) * 64,
            "spec": {
                "purpose": "baseline",
                "scene_mode": "capture",
                "source_pins": contract,
                "implementation_sources": impl,
                "policy_spec": policy,
                "initial_states": {str(ep): {"sha256": "e" * 64} for ep in bank},
            },
        }
        session = write(root / "session.json", receipt)
        results = write(
            root / "results.json",
            {
                "status": "complete",
                "episodes": [
                    {
                        "episode_index": ep,
                        "suite": "libero_spatial",
                        "task_id": 0,
                        "bddl_sha256": a.IDENTITY["bddl_sha256"],
                        "campaign_session_sha256": receipt["sha256"],
                        "initial_observations": {"image": "hash"},
                        "env_steps": 2,
                        "success": True,
                    }
                    for ep in bank
                ],
            },
        )
        metadata = [
            {
                "episode_index": ep,
                "metadata": write(
                    root / f"task_00_episode_{ep:02d}/metadata.json",
                    {"task_id": 0, "task_description": "fixture", "success": True, "frame_count": 2, "sha256": {}},
                ),
            }
            for ep in fit
        ]
        units.append({"session": session, "policy": policy, "routes": route, "results": results, "episodes": metadata})
        audits.append(
            write(
                root / "audit.json",
                {
                    "passed": True,
                    "session_sha256": receipt["sha256"],
                    "episodes": [{"episode": ep, "passed": True} for ep in bank],
                },
            )
        )
    semantics = {
        "passed": True,
        "old_commit": a.OLD_COMMIT,
        "new_commit": a.NEW_COMMIT,
        "reviewer": "fixture",
        "reasoning": "synthetic only",
        "unchanged": dict.fromkeys(
            (
                "policy_inputs",
                "robot_proprio_features",
                "pre_action_alignment",
                "rgb_orientation",
                "control_composition",
                "reset_and_warmup",
            ),
            True,
        ),
        "sources": sources,
    }
    probe = {
        "passed": True,
        "cross_process_exact": True,
        "within_process_exact": True,
        "original_cache_unchanged": True,
        "physics_steps": 0,
        "inference_calls": 3,
        "max_abs": 0,
        "rms": 0,
        "policy_sha256": pins["policy_sha256"],
        "actions_sha256": "f" * 64,
        "reference_actions_sha256": "f" * 64,
        "input_sha256": "a" * 64,
    }
    canary = {
        "identity": a.IDENTITY,
        "actions_sha256": "f" * 64,
        "input_sha256": "a" * 64,
        "actions": write(tmp_path / "actions.json", [1]),
        "evidence": [units[0]["policy"]],
    }
    migration = {
        "passed": True,
        "source_unchanged": True,
        "physical_starts_added": 0,
        "source": {"sha256": a.SOURCE_LEDGER},
        "snapshot": {
            "total_starts": 2,
            "limits": {
                "baselines": {"spent": 2, "reserved": 0, "cap": {"limit": 71}},
                "primary": {"spent": 0, "reserved": 0, "cap": {"limit": 20}},
            },
        },
    }
    proof = {
        "schema": "strider-spatial0-fit-capture-admission-v1",
        "identity": a.IDENTITY,
        "review": {"status": "reviewed_candidate", "reviewer": "fixture", "evidence": "synthetic only"},
        "units": units,
        "old_source_pins": contracts[0],
        "new_source_pins": contracts[1],
        "completed_audits": audits,
        "capture_canaries": [
            write(
                tmp_path / f"capture-canary-{i}.json",
                {"passed": True, "cross_process_exact": True, "actions_hash": "f" * 64, "input_hash": "a" * 64},
            )
            for i in range(2)
        ],
        "migration_receipt": write(tmp_path / "migration.json", migration),
        "cache_recipe": write(tmp_path / "recipe.json", recipe),
        "cache_probe": write(tmp_path / "probe.json", probe),
        "canary_reference": write(tmp_path / "canary.json", canary),
        "input_semantics_review": write(tmp_path / "semantics.json", semantics),
    }
    inventory = {
        "identity": a.IDENTITY,
        "units": units,
        "legacy_capture_admission": write(tmp_path / "admission.json", proof),
    }
    return tmp_path, inventory, proof


def refresh(root, inventory, proof):
    inventory["legacy_capture_admission"] = write(root / "admission.json", proof)


def test_explicit_admission_retains_twenty_frames_source_pins_and_input_bytes(package):
    root, inventory, _ = package
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rows = builder.validate_units(inventory, root)
    assert [r["episode_index"] for r in rows] == list(range(20))
    assert rows[0]["source_pins"]["evaluator_commit"] == a.OLD_COMMIT
    assert rows[1]["source_pins"]["evaluator_commit"] == a.NEW_COMMIT
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_default_still_rejects_mixed_contracts(package):
    root, inventory, _ = package
    del inventory["legacy_capture_admission"]
    with pytest.raises(ValueError, match="mixed source contracts"):
        builder.validate_units(inventory, root)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("arm", "uniform2", "only original1x"), ("ordered_episode_indices", [1], "only original1x")],
)
def test_invalid_ds2_or_wrong_original_bank_never_admitted(package, field, value, match):
    root, inventory, proof = package
    unit = inventory["units"][0]
    p = builder.read(Path(unit["policy"]["path"]))
    p[field] = value
    unit["policy"] = write(Path(unit["policy"]["path"]), p)
    refresh(root, inventory, proof)
    with pytest.raises(ValueError, match=match):
        builder.validate_units(inventory, root)


@pytest.mark.parametrize(
    ("evidence", "field", "value", "match"),
    [
        ("cache_probe", "cross_process_exact", False, "exact canary"),
        ("input_semantics_review", "passed", False, "semantic review"),
        ("migration_receipt", "source_unchanged", False, "conserve source"),
    ],
)
def test_required_evidence_fails_closed(package, evidence, field, value, match):
    root, inventory, proof = package
    path = Path(proof[evidence]["path"])
    data = builder.read(path)
    data[field] = value
    proof[evidence] = write(path, data)
    refresh(root, inventory, proof)
    with pytest.raises(ValueError, match=match):
        builder.validate_units(inventory, root)


def test_missing_episode0_cannot_be_silently_dropped(package):
    root, inventory, proof = package
    inventory["units"][0]["episodes"] = []
    refresh(root, inventory, proof)
    with pytest.raises(ValueError, match="full fixed fit bank"):
        builder.validate_units(inventory, root)


def test_changed_evidence_bytes_rejected(package):
    root, inventory, proof = package
    Path(proof["cache_probe"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence hash"):
        builder.validate_units(inventory, root)


def test_unreviewed_source_file_rejected(package):
    root, inventory, proof = package
    path = Path(proof["input_semantics_review"]["path"])
    s = builder.read(path)
    s["sources"].pop()
    proof["input_semantics_review"] = write(path, s)
    refresh(root, inventory, proof)
    with pytest.raises(ValueError, match="incomplete source"):
        builder.validate_units(inventory, root)
