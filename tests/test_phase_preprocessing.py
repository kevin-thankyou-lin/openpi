import copy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from examples.libero.speed_baselines.phase_preprocessing import PREPROCESSING
from examples.libero.speed_baselines.phase_preprocessing import bind_phase_inputs


@pytest.fixture
def bundle(tmp_path):
    identity = {"suite": "libero_spatial", "task_id": 0, "bddl_sha256": "a" * 64}
    normalization = {
        "image_mean": [0.5] * 3,
        "image_std": [0.2] * 3,
        "proprio_mean": [0.0] * 9,
        "proprio_std": [1.0] * 9,
    }
    manifest = {
        "identity": identity,
        "preprocessing": copy.deepcopy(PREPROCESSING),
        "classes": [{"name": "approach"}, {"name": "place"}],
        "normalization": normalization,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint loaded by test caller")
    payload = {
        "class_names": ["approach", "place"],
        "normalization": copy.deepcopy(normalization),
        "model_config": {"history_length": 4, "horizon": 10, "class_count": 2, "proprio_dim": 9, "image_channels": 3},
        "metadata": {"schema": "strider-observation-history-subtask-checkpoint-v1", "dataset_manifest_sha256": digest},
    }
    return {
        "manifest_path": path,
        "expected_manifest_sha256": digest,
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_payload": payload,
        "identity": identity,
        "phase_names": ("approach", "place"),
    }


def test_pixel_parity_and_policy_image_unchanged(bundle):
    contract = bind_phase_inputs(**bundle)
    source = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    original = source.copy()
    expected = np.asarray(Image.fromarray(source).resize((84, 84), Image.Resampling.BILINEAR))
    actual = contract.image(source)
    assert np.array_equal(actual, expected)
    assert np.array_equal(source, original)
    assert not np.shares_memory(actual, source)
    assert actual.shape == (84, 84, 3)


@pytest.mark.parametrize("field", ["expected_manifest_sha256", "expected_checkpoint_sha256"])
def test_reject_hash_mismatch(bundle, field):
    bundle[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        bind_phase_inputs(**bundle)


@pytest.mark.parametrize(("field", "value"), [("suite", "libero_goal"), ("task_id", 1), ("bddl_sha256", "b" * 64)])
def test_reject_cross_scene_even_same_phase_names(bundle, field, value):
    bundle["identity"] = {**bundle["identity"], field: value}
    with pytest.raises(ValueError, match="scene identity"):
        bind_phase_inputs(**bundle)


@pytest.mark.parametrize("change", ["dataset", "classes", "ontology", "horizon", "proprio", "norm", "zero_std"])
def test_reject_checkpoint_contract_mismatch(bundle, change):
    p = bundle["checkpoint_payload"]
    if change == "dataset":
        p["metadata"]["dataset_manifest_sha256"] = "b" * 64
    if change == "classes":
        p["class_names"].reverse()
    if change == "ontology":
        bundle["phase_names"] = ("place", "approach")
    if change == "horizon":
        p["model_config"]["horizon"] = 20
    if change == "proprio":
        p["model_config"]["proprio_dim"] = 8
    if change == "norm":
        p["normalization"]["image_mean"][0] = 0.1
    if change == "zero_std":
        p["normalization"]["image_std"][0] = 0.0
    with pytest.raises(ValueError, match=r"another dataset|order|model shape|normalization"):
        bind_phase_inputs(**bundle)


def test_no_implicit_legacy_preprocessing(bundle):
    path = bundle["manifest_path"]
    manifest = json.loads(path.read_text())
    del manifest["preprocessing"]
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle["expected_manifest_sha256"] = digest
    bundle["checkpoint_payload"]["metadata"]["dataset_manifest_sha256"] = digest
    with pytest.raises(ValueError, match="explicit phase preprocessing"):
        bind_phase_inputs(**bundle)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 3), dtype=float),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_reject_invalid_images(bundle, image):
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        bind_phase_inputs(**bundle).image(image)
