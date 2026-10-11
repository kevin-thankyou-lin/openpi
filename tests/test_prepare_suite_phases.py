import copy
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from examples.libero.speed_baselines import prepare_suite_phases as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": m.sha(path)}


@pytest.fixture
def package(tmp_path):
    identity = {"suite": "libero_spatial", "task_id": 0, "bddl_sha256": "a" * 64}
    splits = m.split_bank(identity, 1)
    bank = [next(ep for ep, role in splits.items() if role == split) for split in ("train", "validation", "test")]
    fail = next(ep for ep in splits if ep not in bank)
    review = {"status": "reviewed_candidate", "reviewer": "fixture-reviewer", "evidence": "synthetic test frames"}
    ontology = write(
        tmp_path / "ontology.json",
        {"identity": identity, "task_description": "fixture task", "phases": ["approach", "acquire"], "review": review},
    )
    route = write(
        tmp_path / "route.json",
        {
            "routes": [
                {
                    "suite": identity["suite"],
                    "task_id": 0,
                    "bddl": {"sha256": "a" * 64},
                    "method": "uniform",
                    "uniform_stride": 1,
                }
            ]
        },
    )
    policy = write(
        tmp_path / "policy.json",
        {
            "suite": identity["suite"],
            "task_id": 0,
            "arm": "uniform1",
            "policy_sha256": "b" * 64,
            "controller_sha256": "d" * 64,
            "ordered_episode_indices": [*bank, fail],
            "prompt": "fixture task",
            "route_registry": route,
        },
    )
    session = write(
        tmp_path / "session.json",
        {
            "sha256": "c" * 64,
            "spec": {
                "purpose": "baseline",
                "policy_spec": policy,
                "source_pins": {"policy_sha256": "b" * 64, "controller_sha256": "d" * 64},
                "initial_states": {str(ep): {"sha256": str(ep).zfill(64)} for ep in [*bank, fail]},
            },
        },
    )
    episodes, results, annotations = [], [], []
    for ep in [*bank, fail]:
        folder = tmp_path / f"task_00_episode_{ep:02d}"
        folder.mkdir()
        n = 6
        frames = np.arange(n * 16 * 16 * 3, dtype=np.uint8).reshape(n, 16, 16, 3)
        imageio.mimwrite(folder / "top_camera-images-rgb.mp4", frames, fps=10)
        values = {
            "proprio": np.arange(n * 9).reshape(n, 9).astype(np.float32) + ep * 100,
            "action": np.zeros((n, 7), np.float32),
            "env_step": np.arange(n),
            "source_stride": np.ones(n, np.int64),
        }
        for name, value in values.items():
            np.save(folder / f"{name}.npy", value)
        hashes = {p.name: m.sha(p) for p in folder.iterdir()}
        meta = write(
            folder / "metadata.json",
            {
                "task_id": 0,
                "task_description": "fixture task",
                "success": ep != fail,
                "frame_count": n,
                "fps": 10,
                "alignment": "pre-action observation, rendered frame, and applied controller action at each env step",
                "proprio_feature_names": [f"robot0.joint.{i}" for i in range(7)]
                + [f"robot0.gripper.{i}" for i in range(2)],
                "sha256": hashes,
            },
        )
        episodes.append({"episode_index": ep, "metadata": meta})
        results.append(
            {
                "episode_index": ep,
                "task_id": 0,
                "success": ep != fail,
                "env_steps": n,
                "suite": identity["suite"],
                "bddl_sha256": "a" * 64,
                "campaign_session_sha256": "c" * 64,
                "initial_observations": {"image": "hash"},
            }
        )
        if ep != fail:
            annotations.append(
                {
                    "episode_index": ep,
                    "metadata_sha256": meta["sha256"],
                    "review": review,
                    "segments": [
                        {"start": 0, "stop": 2, "phase": "approach"},
                        {"start": 2, "stop": 4, "phase": "acquire"},
                        {"start": 4, "stop": 6, "phase": "approach"},
                    ],
                }
            )
    result = write(tmp_path / "results.json", {"status": "complete", "episodes": results})
    inventory = {
        "identity": identity,
        "units": [{"session": session, "policy": policy, "routes": route, "results": result, "episodes": episodes}],
    }
    annotation = {"identity": identity, "ontology": ontology, "episodes": annotations}
    write(tmp_path / "inventory.json", inventory)
    write(tmp_path / "annotations.json", annotation)
    return tmp_path, inventory, annotation, bank


def build(package):
    root, _, _, _ = package
    return m.prepare(
        root / "inventory.json", root / "annotations.json", root / "out", image_size=8, minimum_successes=3
    )


def test_trainer_artifacts_revisits_train_only_normalization_and_exclusions(package):
    root, _, _, bank = package
    assert build(package)["episodes"] == 3
    manifest = m.read(root / "out/manifest.json")
    assert len(manifest["excluded_unsuccessful_source_episodes"]) == 1
    assert {e["split"] for e in manifest["episodes"]} == {"train", "validation", "test"}
    for episode in manifest["episodes"]:
        np.testing.assert_array_equal(np.load(root / "out" / episode["labels"]), [0, 0, 1, 1, 0, 0])
        assert np.load(root / "out" / episode["images"]).shape == (6, 8, 8, 3)
    expected = np.load(root / f"task_00_episode_{bank[0]:02d}/proprio.npy").mean(0)
    np.testing.assert_allclose(manifest["normalization"]["proprio_mean"], expected)
    assert manifest["identity"]["suite"] == "libero_spatial"
    with pytest.raises(FileExistsError):
        build(package)


@pytest.mark.parametrize("bad_episode", [20, 30, True])
def test_reject_search_confirmation_or_noninteger_episode(package, bad_episode):
    root, inventory, _, _ = package
    inventory["units"][0]["episodes"][0]["episode_index"] = bad_episode
    write(root / "inventory.json", inventory)
    with pytest.raises(ValueError, match="restricted"):
        build(package)


def test_duplicate_across_split_capture_units_rejected(package):
    root, inventory, _, _ = package
    inventory["units"].append(copy.deepcopy(inventory["units"][0]))
    write(root / "inventory.json", inventory)
    with pytest.raises(ValueError, match="duplicate suite"):
        build(package)


def test_union_of_nonoverlapping_capture_units(package):
    root, inventory, _, _ = package
    unit = inventory["units"][0]
    other = copy.deepcopy(unit)
    other["episodes"] = unit["episodes"][1:]
    unit["episodes"] = unit["episodes"][:1]
    inventory["units"].append(other)
    write(root / "inventory.json", inventory)
    assert build(package)["episodes"] == 3


def test_source_hash_corruption_rejected(package):
    root, inventory, _, _ = package
    policy = Path(inventory["units"][0]["policy"]["path"])
    policy.write_text(policy.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(package)


@pytest.mark.parametrize("kind", ["search", "ds2"])
def test_nonbaseline1x_rejected_even_with_consistent_hashes(package, kind):
    root, inventory, _, _ = package
    unit = inventory["units"][0]
    session = m.read(Path(unit["session"]["path"]))
    if kind == "search":
        session["spec"]["purpose"] = "search_candidate"
    else:
        policy = m.read(Path(unit["policy"]["path"]))
        policy["arm"] = "uniform2"
        unit["policy"] = write(root / "policy.json", policy)
        session["spec"]["policy_spec"] = unit["policy"]
    unit["session"] = write(root / "session.json", session)
    write(root / "inventory.json", inventory)
    with pytest.raises(ValueError, match="baseline uniform1"):
        build(package)


def test_pending_annotations_cannot_generate_labels(package):
    root, _, ann, _ = package
    ann["episodes"][0]["review"]["status"] = "proposed"
    write(root / "annotations.json", ann)
    with pytest.raises(ValueError, match="reviewed candidate"):
        build(package)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [{"start": 1, "stop": 6, "phase": "approach"}],
        [{"start": 0, "stop": 3, "phase": "approach"}, {"start": 2, "stop": 6, "phase": "acquire"}],
    ],
)
def test_no_unreviewed_gaps_or_overlap(segments):
    with pytest.raises(ValueError, match="cover every frame|incomplete reviewed"):
        m.segment_labels(segments, ["approach", "acquire"], 6)


def test_reject_stride2_with_updated_array_and_metadata_hash(package):
    root, inventory, ann, bank = package
    folder = root / f"task_00_episode_{bank[0]:02d}"
    np.save(folder / "source_stride.npy", np.full(6, 2))
    meta = m.read(folder / "metadata.json")
    meta["sha256"]["source_stride.npy"] = m.sha(folder / "source_stride.npy")
    ref = write(folder / "metadata.json", meta)
    inventory["units"][0]["episodes"][0]["metadata"] = ref
    ann["episodes"][0]["metadata_sha256"] = ref["sha256"]
    write(root / "inventory.json", inventory)
    write(root / "annotations.json", ann)
    with pytest.raises(ValueError, match="contiguous"):
        build(package)


def test_insufficient_data_does_not_expand_bank(package):
    root, _, _, _ = package
    with pytest.raises(ValueError, match="no extra rollouts"):
        m.prepare(root / "inventory.json", root / "annotations.json", root / "out")
    assert not (root / "out").exists()


@pytest.mark.parametrize("field", ["alignment", "proprio_feature_names"])
def test_reject_post_action_or_privileged_features(package, field):
    root, inventory, ann, bank = package
    folder = root / f"task_00_episode_{bank[0]:02d}"
    meta = m.read(folder / "metadata.json")
    if field == "alignment":
        meta[field] = "post-action"
    else:
        meta[field][0] = "bowl.goal_distance"
    ref = write(folder / "metadata.json", meta)
    inventory["units"][0]["episodes"][0]["metadata"] = ref
    ann["episodes"][0]["metadata_sha256"] = ref["sha256"]
    write(root / "inventory.json", inventory)
    write(root / "annotations.json", ann)
    with pytest.raises(ValueError, match="pre-action frame|robot joint/gripper"):
        build(package)


def test_result_cannot_substitute_other_suite(package):
    root, inventory, _, _ = package
    unit = inventory["units"][0]
    results = m.read(Path(unit["results"]["path"]))
    results["episodes"][0]["suite"] = "libero_goal"
    unit["results"] = write(root / "results.json", results)
    write(root / "inventory.json", inventory)
    with pytest.raises(ValueError, match="result campaign identity"):
        build(package)


def test_missing_training_phase_blocks_artifact(package):
    root, _, ann, bank = package
    next(e for e in ann["episodes"] if e["episode_index"] == bank[0])["segments"] = [
        {"start": 0, "stop": 6, "phase": "approach"}
    ]
    write(root / "annotations.json", ann)
    with pytest.raises(ValueError, match="training split lacks"):
        build(package)
