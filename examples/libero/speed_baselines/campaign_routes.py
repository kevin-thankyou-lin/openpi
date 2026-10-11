"""Hash-pinned suite/task routes; importing this module never imports a simulator.

Counts mirror LIBERO's benchmark/libero_suite_task_map.py. Horizons mirror this
package's main.MAX_STEPS, checked by a source-level test. Support is distinct
from campaign authorization: callers must supply their allowed suite list.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
from types import MappingProxyType

from .phase_preprocessing import PREPROCESSING

SCHEMA = "strider-libero-campaign-routes-v1"
SUITE_TASK_COUNTS = MappingProxyType(
    {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
)
SUITE_HORIZONS = MappingProxyType(
    {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
)
DEFAULT_CAMPAIGN_SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def validate_task(suite: str, task_id: int) -> None:
    if suite not in SUITE_TASK_COUNTS:
        raise ValueError(f"unsupported LIBERO suite: {suite!r}")
    if type(task_id) is not int or not 0 <= task_id < SUITE_TASK_COUNTS[suite]:
        raise ValueError(f"task_id outside {suite} range: {task_id!r}")


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class TaskIdentity:
    suite: str
    task_id: int
    bddl_sha256: str

    def __post_init__(self) -> None:
        validate_task(self.suite, self.task_id)
        _digest(self.bddl_sha256)

    @property
    def key(self) -> str:
        return f"{self.suite}:task_{self.task_id:02d}:{self.bddl_sha256}"


@dataclasses.dataclass(frozen=True)
class FileAsset:
    path: pathlib.Path
    sha256: str

    def verify(self) -> None:
        _digest(self.sha256)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if file_sha256(self.path) != self.sha256:
            raise ValueError(f"asset hash mismatch: {self.path}")


@dataclasses.dataclass(frozen=True)
class CampaignRoute:
    identity: TaskIdentity
    bddl: FileAsset
    method: str
    uniform_stride: int | None = None
    phase_checkpoint: FileAsset | None = None
    ontology: FileAsset | None = None
    dataset_manifest: FileAsset | None = None
    phase_names: tuple[str, ...] = ()
    asset_binding: TaskIdentity | None = None

    @property
    def arm(self) -> str:
        return f"uniform{self.uniform_stride}" if self.method == "uniform" else self.method

    @property
    def key(self) -> tuple[str, int, str]:
        return self.identity.suite, self.identity.task_id, self.arm

    def verify(self) -> None:
        self.bddl.verify()
        if self.identity.bddl_sha256 != self.bddl.sha256:
            raise ValueError("task identity BDDL hash mismatch")
        if self.method == "uniform":
            if type(self.uniform_stride) is not int or self.uniform_stride not in (1, 2, 3):
                raise ValueError("uniform stride must be 1, 2 or 3")
            if (
                self.phase_checkpoint
                or self.ontology
                or self.phase_names
                or self.asset_binding
                or self.dataset_manifest
            ):
                raise ValueError("uniform route must not bind phase assets")
            return
        if self.method != "strider" or self.uniform_stride is not None:
            raise ValueError("route must be uniform or strider without a uniform stride")
        if self.phase_checkpoint is None or self.ontology is None or self.dataset_manifest is None:
            raise ValueError("challenger requires explicit checkpoint, ontology and dataset manifest")
        if self.asset_binding != self.identity:
            raise ValueError("phase asset binding must match suite, task and BDDL hash")
        if not self.phase_names or len(set(self.phase_names)) != len(self.phase_names):
            raise ValueError("phase names must be nonempty and unique")
        if any(not isinstance(name, str) or not name.strip() or ":" in name for name in self.phase_names):
            raise ValueError("phase names must be nonempty unqualified strings")
        self.phase_checkpoint.verify()
        self.ontology.verify()
        ontology = json.loads(self.ontology.path.read_text())
        if (
            not isinstance(ontology, dict)
            or type(ontology.get("task_id")) is not int
            or ontology["task_id"] != self.identity.task_id
        ):
            raise ValueError("ontology task_id mismatch")
        if tuple(ontology.get("phases", ())) != self.phase_names:
            raise ValueError("ontology phase names mismatch")
        # Legacy ontologies are suite-less. Their outer asset_binding is mandatory.
        if "suite" in ontology and ontology["suite"] != self.identity.suite:
            raise ValueError("ontology suite mismatch")
        if "bddl_sha256" in ontology and ontology["bddl_sha256"] != self.identity.bddl_sha256:
            raise ValueError("ontology BDDL hash mismatch")
        if "identity" in ontology and ontology["identity"] != dataclasses.asdict(self.identity):
            raise ValueError("ontology nested identity mismatch")

        self.dataset_manifest.verify()
        manifest = json.loads(self.dataset_manifest.path.read_text())
        if manifest.get("schema") != "strider-libero-prepared-phase-dataset-v1":
            raise ValueError("unsupported phase dataset manifest schema")
        if manifest.get("identity") != dataclasses.asdict(self.identity):
            raise ValueError("phase dataset task identity mismatch")
        if manifest.get("preprocessing") != PREPROCESSING:
            raise ValueError("phase dataset requires explicit supported preprocessing")
        if tuple(c["name"] for c in manifest.get("classes", ())) != self.phase_names:
            raise ValueError("phase dataset/ontology order mismatch")

    def verify_phase_input_proof(self, proof: dict) -> None:
        self.verify()
        if self.method != "strider":
            raise ValueError("uniform route has no phase input proof")
        expected = {
            "schema": "strider-verified-phase-inputs-v1",
            "dataset_manifest_sha256": self.dataset_manifest.sha256,
            "checkpoint_sha256": self.phase_checkpoint.sha256,
            "identity": dataclasses.asdict(self.identity),
            "class_names": list(self.phase_names),
            "history_length": 4,
            "horizon": 10,
            "preprocessing": PREPROCESSING,
            "checkpoint_load": "same SHA-verified in-memory bytes; constructed from verified payload",
        }
        if proof != expected:
            raise ValueError("phase input proof differs from registered assets/contract")

    def phase_input_binding(self) -> dict:
        """Only return verified campaign bindings; checkpoint contents checked by selector."""
        self.verify()
        if self.method != "strider":
            raise ValueError("uniform route has no phase input binding")
        return {
            "manifest_path": self.dataset_manifest.path,
            "expected_manifest_sha256": self.dataset_manifest.sha256,
            "expected_checkpoint_sha256": self.phase_checkpoint.sha256,
            "identity": dataclasses.asdict(self.identity),
            "phase_names": self.phase_names,
        }


def load_routes(
    path: pathlib.Path, *, allowed_suites: tuple[str, ...] = DEFAULT_CAMPAIGN_SUITES
) -> dict[tuple[str, int, str], CampaignRoute]:
    """Load immutable asset bindings; relative paths resolve against the registry."""
    path = pathlib.Path(path).resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unsupported campaign route schema")
    entries = payload.get("routes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("routes must be a nonempty list")

    def asset(value: dict) -> FileAsset:
        source = pathlib.Path(value["path"])
        return FileAsset((path.parent / source).resolve(), value["sha256"])

    routes = {}
    task_hashes = {}
    for entry in entries:
        identity = TaskIdentity(entry["suite"], entry["task_id"], entry["bddl"]["sha256"])
        if identity.suite not in allowed_suites:
            raise ValueError(f"suite not authorized by route allowlist: {identity.suite}")
        task_key = identity.suite, identity.task_id
        if task_hashes.setdefault(task_key, identity.bddl_sha256) != identity.bddl_sha256:
            raise ValueError(f"conflicting BDDL identity across arms: {task_key}")
        binding = entry.get("asset_binding")
        route = CampaignRoute(
            identity=identity,
            bddl=asset(entry["bddl"]),
            method=entry["method"],
            uniform_stride=entry.get("uniform_stride"),
            phase_checkpoint=asset(entry["phase_checkpoint"]) if entry.get("phase_checkpoint") else None,
            ontology=asset(entry["ontology"]) if entry.get("ontology") else None,
            dataset_manifest=asset(entry["dataset_manifest"]) if entry.get("dataset_manifest") else None,
            phase_names=tuple(entry.get("phase_names", ())),
            asset_binding=TaskIdentity(**binding) if binding is not None else None,
        )
        route.verify()
        if route.key in routes:
            raise ValueError(f"duplicate route: {route.key}")
        routes[route.key] = route
    return routes


def select_route(
    routes: dict[tuple[str, int, str], CampaignRoute], *, suite: str, task_id: int, arm: str
) -> CampaignRoute:
    validate_task(suite, task_id)
    try:
        route = routes[suite, task_id, arm]
    except KeyError as error:
        raise ValueError(f"missing route: {(suite, task_id, arm)}") from error
    route.verify()
    return route
