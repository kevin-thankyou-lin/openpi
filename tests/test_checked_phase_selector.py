"""Real CPU phase checkpoint integration; no policy server or simulator."""

# Tests inspect buffered frames and import optional Torch only within CPU fixtures.
# ruff: noqa: PLC0415, SLF001

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from examples.libero.speed_baselines.actions import SupActionComposer
from examples.libero.speed_baselines.campaign_routes import CampaignRoute
from examples.libero.speed_baselines.campaign_routes import FileAsset
from examples.libero.speed_baselines.campaign_routes import TaskIdentity
from examples.libero.speed_baselines.phase_preprocessing import PREPROCESSING
from examples.libero.speed_baselines.strider_client import StriderPhaseSelector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checked(tmp_path):
    torch = pytest.importorskip("torch")
    import strider.subtask_model as models

    torch.set_num_threads(1)
    bddl = tmp_path / "task.bddl"
    bddl.write_text("test BDDL")
    identity = TaskIdentity("libero_spatial", 0, sha(bddl))
    normalization = {
        "image_mean": [0.5] * 3,
        "image_std": [0.2] * 3,
        "proprio_mean": [0.0] * 9,
        "proprio_std": [1.0] * 9,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "strider-libero-prepared-phase-dataset-v1",
                "identity": dataclasses.asdict(identity),
                "preprocessing": PREPROCESSING,
                "classes": [{"name": "approach"}, {"name": "place"}],
                "normalization": normalization,
            }
        )
    )
    config = models.SubtaskModelConfig(history_length=4, horizon=10, class_count=2, proprio_dim=9)
    model = models.ObservationHistorySubtaskModel(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = {
        "model_config": dataclasses.asdict(config),
        "model_state_dict": model.state_dict(),
        "class_names": ["approach", "place"],
        "normalization": normalization,
        "metadata": {
            "schema": "strider-observation-history-subtask-checkpoint-v1",
            "dataset_manifest_sha256": sha(manifest),
        },
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"task_id": 0, "phases": ["approach", "place"]}))
    route = CampaignRoute(
        identity=identity,
        bddl=FileAsset(bddl, sha(bddl)),
        method="strider",
        phase_checkpoint=FileAsset(checkpoint, sha(checkpoint)),
        ontology=FileAsset(ontology, sha(ontology)),
        dataset_manifest=FileAsset(manifest, sha(manifest)),
        phase_names=("approach", "place"),
        asset_binding=identity,
    )
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema": "strider-libero-task-fine-phase-schedule-v1",
                "status": "AI_CANDIDATE_NOT_PROMOTED_OR_EVALUATED",
                "authority": "AI_CANDIDATE_NOT_HUMAN_ANNOTATION",
                "requires_human_confirmation": True,
                "phase_schedule": [["task_00:approach", 2], ["task_00:place", 1]],
            }
        )
    )
    composer = SupActionComposer(
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.array([0.05] * 3 + [0.5] * 3),
        output_max=np.array([0.05] * 3 + [0.5] * 3),
    )
    kwargs = {
        "checkpoint": checkpoint,
        "phase_repo": Path(models.__file__).parents[2],
        "schedule": schedule,
        "fast_stride": 3,
        "device": "cpu",
        "phase_input_binding": route.phase_input_binding(),
        "composer": composer,
    }
    return kwargs, route, payload, models


def test_real_checkpoint_preprocess_plan_and_proof(checked):
    kwargs, route, _payload, _models = checked
    selector = StriderPhaseSelector(**kwargs)
    source = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    before = source.copy()
    obs = {"robot0_joint_pos": np.arange(7), "robot0_gripper_qpos": np.zeros(2)}
    selector.observe(source, obs)
    assert np.array_equal(source, before)
    expected = np.asarray(Image.fromarray(source).resize((84, 84), Image.Resampling.BILINEAR))
    assert np.array_equal(selector._observations[0]["rgb"], expected)
    slices, decision = selector.plan(np.zeros((10, 7)), task_id=0)
    assert [s.source_indices for s in slices] == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    assert decision["predicted_phases"] == ["task_00:approach"] * 10
    route.verify_phase_input_proof(selector.phase_input_proof)
    with pytest.raises(ValueError, match="phase input proof"):
        route.verify_phase_input_proof({**selector.phase_input_proof, "horizon": 20})


def test_constructor_uses_same_verified_payload_without_reopening(checked, monkeypatch):
    kwargs, _route, _payload, models = checked
    import torch

    original = torch.load
    calls = []

    def load(stream, **kw):
        assert hasattr(stream, "read")
        result = original(stream, **kw)
        calls.append(result)
        kwargs["checkpoint"].write_bytes(b"file changed after loaded bytes")
        return result

    def forbidden(*a, **kw):
        raise AssertionError("checkpoint reopened by predictor")

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(models.TorchObservationHistorySubtaskPredictor, "from_checkpoint", forbidden)
    selector = StriderPhaseSelector(**kwargs)
    assert len(calls) == 1
    assert selector.predictor.horizon == 10


@pytest.mark.parametrize("change", ["sha", "dataset", "classes", "shape"])
def test_bad_checkpoint_refused_before_model_construction(checked, monkeypatch, change):
    kwargs, _route, payload, models = checked
    import torch

    if change == "dataset":
        payload["metadata"]["dataset_manifest_sha256"] = "a" * 64
    if change == "classes":
        payload["class_names"].reverse()
    if change == "shape":
        payload["model_config"]["horizon"] = 20
    torch.save(payload, kwargs["checkpoint"])
    if change != "sha":
        kwargs["phase_input_binding"]["expected_checkpoint_sha256"] = sha(kwargs["checkpoint"])
    else:
        kwargs["phase_input_binding"]["expected_checkpoint_sha256"] = "0" * 64

    def forbidden(*a, **kw):
        raise AssertionError("model constructed before input binding validation")

    monkeypatch.setattr(models, "ObservationHistorySubtaskModel", forbidden)
    with pytest.raises(ValueError, match=r"SHA mismatch|another dataset|class order|model shape"):
        StriderPhaseSelector(**kwargs)


def test_wrong_import_root_and_implicit_legacy_refused(checked):
    kwargs, _route, _payload, _models = checked
    with pytest.raises(ValueError, match="pinned phase source root"):
        StriderPhaseSelector(**{**kwargs, "phase_repo": kwargs["checkpoint"].parent})
    with pytest.raises(ValueError, match="explicit historical"):
        StriderPhaseSelector(**{**kwargs, "phase_input_binding": None})
    with pytest.raises(ValueError, match="explicit historical"):
        StriderPhaseSelector(**{**kwargs, "legacy_input_mode": True})


def test_checked_campaign_rejects_fractional_schedule(checked):
    kwargs, _route, _payload, _models = checked
    path = kwargs["schedule"]
    schedule = json.loads(path.read_text())
    schedule["phase_schedule"][0][1] = 1.5
    path.write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match="must be integers"):
        StriderPhaseSelector(**kwargs)
