"""Narrow, explicit offline admission of Spatial0's original successful fit capture.

This does not rewrite any campaign receipt or establish trajectory equivalence.
The normal dataset builder continues to reject mixed source contracts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

IDENTITY = {
    "suite": "libero_spatial",
    "task_id": 0,
    "bddl_sha256": "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b",
}
OLD_SESSION_SHA256 = "d211a54ffaa6cc410702a9a7627fa77096da72ad8a91d20d94eba8916ac0c83b"
OLD_COMMIT = "2ed01b90bbeab01ae55b1100f9841ce884fe1149"
NEW_COMMIT = "02e77ffb40a1183592b83f6752c101818c2103a3"
SOURCE_LEDGER = "17507503adbeebff29c20f1a989dc359307e17af6b9d2df02b9754850a696cf8"
RECIPE = "915fc441cc0fd52c3c77f4dcfede8bb98b52c34ba3c4884ab3b91a1aeb727145"


def _require(condition, message):
    if not condition:
        raise ValueError("legacy capture admission: " + message)


def _read(base, ref):
    path = (base / ref["path"]).resolve()
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "evidence hash mismatch")
    return path, json.loads(raw)


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_admission(inventory: dict, base: Path) -> dict | None:
    reference = inventory.get("legacy_capture_admission")
    if reference is None:
        return None
    path, proof = _read(base, reference)
    base = path.parent
    _require(proof.get("schema") == "strider-spatial0-fit-capture-admission-v1", "unknown schema")
    _require(inventory["identity"] == proof["identity"] == IDENTITY, "only exact Spatial0 identity allowed")
    review = proof["review"]
    _require(
        review.get("status") == "reviewed_candidate" and review.get("reviewer") and review.get("evidence"),
        "review required",
    )
    _require(len(inventory["units"]) == 2 and len(proof["units"]) == 2, "exact two capture units required")
    contracts = []
    specs = []
    for index, (unit, pinned) in enumerate(zip(inventory["units"], proof["units"], strict=True)):
        _require(unit == pinned, "source inventory differs from reviewed unit")
        _, receipt = _read(base, pinned["session"])
        _, policy = _read(base, pinned["policy"])
        _, results = _read(base, pinned["results"])
        _read(base, pinned["routes"])
        expected_bank = [0] if index == 0 else list(range(1, 40))
        fit_bank = [0] if index == 0 else list(range(1, 20))
        _require(
            policy["arm"] == "uniform1" and policy["ordered_episode_indices"] == expected_bank,
            "only original1x and fixed1x continuation allowed",
        )
        _require([r["episode_index"] for r in unit["episodes"]] == fit_bank, "full fixed fit bank required")
        _require(index != 0 or receipt["sha256"] == OLD_SESSION_SHA256, "not the original successful capture0 session")
        spec = receipt["spec"]
        _require(spec["purpose"] == "baseline" and spec["scene_mode"] == "capture", "not a baseline capture")
        _require(
            spec["source_pins"]["evaluator_commit"] == (OLD_COMMIT if index == 0 else NEW_COMMIT), "wrong source commit"
        )
        _require(
            results["status"] == "complete" and [r["episode_index"] for r in results["episodes"]] == expected_bank,
            "incomplete original source results",
        )
        _require(index != 0 or results["episodes"][0]["success"] is True, "old capture0 must be successful")
        for item in unit["episodes"]:
            metadata_path, metadata = _read(base, item["metadata"])
            for name, expected in metadata["sha256"].items():
                _require(Path(name).name == name, "unsafe telemetry name")
                _require(
                    hashlib.sha256((metadata_path.parent / name).read_bytes()).hexdigest() == expected,
                    "source telemetry changed",
                )
        _, audit = _read(base, proof["completed_audits"][index])
        _require(
            audit.get("passed") is True and audit["session_sha256"] == receipt["sha256"],
            "capture terminal audit mismatch",
        )
        _require(
            [r["episode"] for r in audit["episodes"]] == expected_bank
            and all(r["passed"] is True for r in audit["episodes"]),
            "incomplete capture audit",
        )
        contracts.append(spec["source_pins"])
        specs.append(spec)
    capture_canaries = [_read(base, ref)[1] for ref in proof["capture_canaries"]]
    _require(
        len(capture_canaries) == 2 and all(c.get("passed") is True for c in capture_canaries),
        "both capture canaries required",
    )
    _require(capture_canaries[1].get("cross_process_exact") is True, "fixed capture must pass exact canary")
    old, new = contracts
    _require(proof["old_source_pins"] == old and proof["new_source_pins"] == new, "source pin binding mismatch")
    _require(
        {k: v for k, v in old.items() if k != "evaluator_commit"}
        == {k: v for k, v in new.items() if k not in ("evaluator_commit", "inference_cache_sha256")},
        "policy/controller or other source contract changed",
    )
    _require(new.get("inference_cache_sha256") == RECIPE, "wrong adopted cache")
    _, migration = _read(base, proof["migration_receipt"])
    _require(
        migration.get("passed") is True
        and migration.get("source_unchanged") is True
        and migration.get("physical_starts_added") == 0,
        "migration did not conserve source",
    )
    _require(migration["source"]["sha256"] == SOURCE_LEDGER, "wrong original source ledger")
    snap = migration["snapshot"]
    _require(
        snap["total_starts"] == 2
        and snap["limits"]["baselines"]["spent"] == 2
        and snap["limits"]["primary"]["spent"] == 0,
        "both original starts must remain counted",
    )
    _require(
        snap["limits"]["baselines"]["cap"]["limit"] == 71
        and snap["limits"]["primary"]["cap"]["limit"] == 20
        and all(v["reserved"] == 0 for v in snap["limits"].values()),
        "migration caps/reservations differ",
    )
    _, recipe = _read(base, proof["cache_recipe"])
    _require(
        _canonical(recipe["identity"]) == recipe["recipe_sha256"] == RECIPE
        and recipe["identity"]["policy_sha256"] == old["policy_sha256"],
        "cache recipe not bound",
    )
    _, probe = _read(base, proof["cache_probe"])
    _require(
        all(
            probe.get(k) is True
            for k in ("passed", "cross_process_exact", "within_process_exact", "original_cache_unchanged")
        ),
        "exact canary probe required",
    )
    _require(
        probe["physics_steps"] == 0
        and probe["inference_calls"] == 3
        and probe["max_abs"] == probe["rms"] == 0
        and probe["policy_sha256"] == old["policy_sha256"],
        "invalid zero-physics probe",
    )
    canary_path, canary = _read(base, proof["canary_reference"])
    _require({k: canary["identity"][k] for k in IDENTITY} == IDENTITY, "canary task mismatch")
    _require(
        canary["actions_sha256"] == probe["actions_sha256"] == probe["reference_actions_sha256"]
        and canary["input_sha256"] == probe["input_sha256"],
        "canary reference mismatch",
    )
    _require(
        all(
            c["actions_hash"] == probe["actions_sha256"] and c["input_hash"] == probe["input_sha256"]
            for c in capture_canaries
        ),
        "capture canary differs from adopted reference",
    )
    for source in [canary["actions"], *canary["evidence"]]:
        raw = (canary_path.parent / source["path"]).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == source["sha256"], "canary evidence changed")
    semantics_path, semantics = _read(base, proof["input_semantics_review"])
    _require(
        semantics.get("passed") is True
        and semantics.get("old_commit") == OLD_COMMIT
        and semantics.get("new_commit") == NEW_COMMIT
        and semantics.get("reviewer")
        and semantics.get("reasoning"),
        "explicit source semantic review required",
    )
    required = {
        "policy_inputs",
        "robot_proprio_features",
        "pre_action_alignment",
        "rgb_orientation",
        "control_composition",
        "reset_and_warmup",
    }
    _require(
        set(semantics["unchanged"]) == required and all(v is True for v in semantics["unchanged"].values()),
        "input semantics not reviewed",
    )
    required_sources = {
        "evaluator",
        "session",
        "policy_adapter",
        "actions",
        "controller",
        "scene_replay",
        "telemetry",
        "provenance",
    }
    observed = set()
    for source in semantics["sources"]:
        role, key = source["role"], source["source_key"]
        _require(role in ("old", "new") and key in required_sources, "unexpected reviewed source")
        _require((role, key) not in observed, "duplicate reviewed source")
        observed.add((role, key))
        spec = specs[0 if role == "old" else 1]
        _require(
            source["sha256"] == spec["implementation_sources"][key]["sha256"],
            "review source differs from capture implementation",
        )
        data = (semantics_path.parent / source["path"]).read_bytes()
        _require(hashlib.sha256(data).hexdigest() == source["sha256"], "reviewed source changed")
    _require(
        observed == {(role, key) for role in ("old", "new") for key in required_sources},
        "incomplete source semantics review",
    )
    return {
        "sha256": reference["sha256"],
        "contracts": contracts,
        "scope": "offline successful1x fit footage only; invalid DS2 remains excluded",
    }
