"""Build trainer-compatible phase artifacts from reviewed, baseline-only footage.

No simulator, inference, automatic annotations, or training. The inventory binds
local copies of campaign receipts; annotations use half-open pre-action frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def asset(base: Path, ref: dict) -> tuple[Path, dict]:
    path = (base / ref["path"]).resolve()
    if sha(path) != ref["sha256"]:
        raise ValueError(f"asset hash mismatch: {path}")
    return path, read(path)


def reviewed(value: dict) -> None:
    if value.get("status") != "reviewed_candidate" or not value.get("reviewer") or not value.get("evidence"):
        raise ValueError("explicit reviewed candidate with reviewer and evidence required")


def split_bank(identity: dict, seed: int) -> dict[int, str]:
    # Split the fixed bank BEFORE outcomes/annotation availability are observed.
    key = json.dumps(identity, sort_keys=True)
    order = sorted(range(20), key=lambda ep: hashlib.sha256(f"{seed}:{key}:{ep}".encode()).digest())
    return {ep: "train" if n < 12 else "validation" if n < 16 else "test" for n, ep in enumerate(order)}


def segment_labels(segments: list[dict], phases: list[str], length: int) -> np.ndarray:
    labels = np.empty(length, dtype=np.int64)
    cursor = 0
    for segment in segments:
        start, stop = segment["start"], segment["stop"]
        if type(start) is not int or type(stop) is not int or start != cursor or not start < stop <= length:
            raise ValueError("reviewed segments must cover every frame once, without gaps/overlap")
        if segment["phase"] not in phases:
            raise ValueError("unknown reviewed phase")
        labels[start:stop] = phases.index(segment["phase"])
        cursor = stop
    if cursor != length:
        raise ValueError("incomplete reviewed labels")
    return labels


def validate_units(inventory: dict, base: Path) -> list[dict]:
    identity = inventory["identity"]
    suite, task = identity["suite"], identity["task_id"]
    if suite not in SUITES or type(task) is not int or task not in range(10):
        raise ValueError("unknown suite/task")
    if len(identity["bddl_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in identity["bddl_sha256"]):
        raise ValueError("BDDL content identity required")
    seen = set()
    output = []
    common_pins = None
    for unit in inventory["units"]:
        _, receipt = asset(base, unit["session"])
        _, policy = asset(base, unit["policy"])
        _, routes = asset(base, unit["routes"])
        _, results = asset(base, unit["results"])
        session = receipt["spec"]
        if results.get("status") != "complete":
            raise ValueError("completed baseline source results required")
        if session["purpose"] != "baseline" or policy["arm"] != "uniform1":
            raise ValueError("only baseline uniform1 can supply phase-fit data")
        if session["policy_spec"]["sha256"] != unit["policy"]["sha256"]:
            raise ValueError("session-policy binding mismatch")
        if policy["route_registry"]["sha256"] != unit["routes"]["sha256"]:
            raise ValueError("policy-route binding mismatch")
        if (policy["suite"], policy["task_id"]) != (suite, task):
            raise ValueError("policy suite/task mismatch")
        matches = [r for r in routes["routes"] if (r["suite"], r["task_id"]) == (suite, task)]
        if len(matches) != 1:
            raise ValueError("ambiguous source route")
        route = matches[0]
        if (
            route["bddl"]["sha256"] != identity["bddl_sha256"]
            or route["method"] != "uniform"
            or route["uniform_stride"] != 1
        ):
            raise ValueError("route does not match task or uniform1")
        pins = session["source_pins"]
        if not {"policy_sha256", "controller_sha256"} <= pins.keys() or not session["initial_states"]:
            raise ValueError("missing source/initial-state provenance")
        if any(policy[key] != pins[key] for key in ("policy_sha256", "controller_sha256")):
            raise ValueError("policy/controller content identity mismatch")
        if common_pins is not None and pins != common_pins:
            raise ValueError("mixed source contracts")
        common_pins = pins
        episodes = {e["episode_index"]: e for e in results["episodes"]}
        if len(episodes) != len(results["episodes"]):
            raise ValueError("duplicate source result episode")
        for item in unit["episodes"]:
            ep = item["episode_index"]
            if type(ep) is not int or ep not in range(20):
                raise ValueError("phase fitting is restricted to baseline bank0..19")
            if ep in seen:
                raise ValueError("duplicate suite/task/episode across source units")
            seen.add(ep)
            if ep not in policy["ordered_episode_indices"] or str(ep) not in session["initial_states"]:
                raise ValueError("episode absent from registered source bank")
            path, meta = asset(base, item["metadata"])
            result = episodes[ep]
            if (
                result["suite"] != suite
                or result["bddl_sha256"] != identity["bddl_sha256"]
                or result["campaign_session_sha256"] != receipt["sha256"]
                or not result["initial_observations"]
            ):
                raise ValueError("result campaign identity mismatch")
            if path.parent.name != f"task_{task:02d}_episode_{ep:02d}":
                raise ValueError("telemetry directory does not identify registered episode")
            if meta["task_id"] != task or result["task_id"] != task or meta["task_description"] != policy["prompt"]:
                raise ValueError("telemetry task identity mismatch")
            if (
                type(meta["success"]) is not bool
                or meta["success"] != result["success"]
                or meta["frame_count"] != result["env_steps"]
            ):
                raise ValueError("result-telemetry mismatch")
            output.append(
                {
                    "episode_index": ep,
                    "metadata_path": path,
                    "metadata": meta,
                    "metadata_sha256": item["metadata"]["sha256"],
                    "unit": unit,
                    "initial_state": session["initial_states"][str(ep)],
                    "source_pins": pins,
                }
            )
    return sorted(output, key=lambda row: row["episode_index"])


def prepare(
    inventory_path: Path,
    annotations_path: Path,
    output: Path,
    *,
    image_size: int = 84,
    split_seed: int = 1,
    minimum_successes: int = 15,
) -> dict:
    if image_size < 1 or minimum_successes < 1:
        raise ValueError("positive image size and minimum successes required")
    inventory, annotation = read(inventory_path), read(annotations_path)
    if annotation["identity"] != inventory["identity"]:
        raise ValueError("annotation task identity mismatch")
    ontology_path, ontology = asset(annotations_path.parent, annotation["ontology"])
    if ontology["identity"] != inventory["identity"]:
        raise ValueError("ontology task identity mismatch")
    reviewed(ontology["review"])
    phases = ontology["phases"]
    if not phases or len(set(phases)) != len(phases) or not all(isinstance(x, str) and x for x in phases):
        raise ValueError("unique named phases required")
    labels_by_episode = {}
    for row in annotation["episodes"]:
        ep = row["episode_index"]
        if type(ep) is not int or ep not in range(20) or ep in labels_by_episode:
            raise ValueError("duplicate or out-of-fit annotation episode")
        reviewed(row["review"])
        labels_by_episode[ep] = row
    sources = validate_units(inventory, inventory_path.parent)
    if any(r["metadata"]["task_description"] != ontology["task_description"] for r in sources):
        raise ValueError("ontology language differs from baseline task")
    if set(labels_by_episode) - {r["episode_index"] for r in sources}:
        raise ValueError("annotation has no source footage")
    successes = [r for r in sources if r["metadata"]["success"]]
    if len(successes) < minimum_successes:
        raise ValueError("insufficient successful baseline footage; no extra rollouts are authorized")
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite dataset or failed partial build")
    partial.mkdir(parents=True)
    splits = split_bank(inventory["identity"], split_seed)
    records = []
    train_classes = set()
    sums = {
        "image": np.zeros(3),
        "image2": np.zeros(3),
        "image_n": 0,
        "proprio": None,
        "proprio2": None,
        "proprio_n": 0,
    }
    excluded = []
    for row in sources:
        ep, meta = row["episode_index"], row["metadata"]
        if not meta["success"]:
            excluded.append(
                {"episode_index": ep, "reason": "unsuccessful baseline", "metadata_sha256": row["metadata_sha256"]}
            )
            continue
        if ep not in labels_by_episode:
            raise ValueError("successful baseline missing reviewed annotations")
        ann = labels_by_episode[ep]
        if ann["metadata_sha256"] != row["metadata_sha256"]:
            raise ValueError("annotations do not bind exact footage metadata")
        folder = row["metadata_path"].parent
        names = ["proprio.npy", "action.npy", "env_step.npy", "source_stride.npy", "top_camera-images-rgb.mp4"]
        for name in names:
            if sha(folder / name) != meta["sha256"][name]:
                raise ValueError(f"telemetry hash mismatch: {name}")
        if (
            meta.get("alignment")
            != "pre-action observation, rendered frame, and applied controller action at each env step"
        ):
            raise ValueError("pre-action frame contract required")
        n = meta["frame_count"]
        if type(n) is not int or n < 1:
            raise ValueError("positive integer frame count required")
        proprio = np.load(folder / "proprio.npy", allow_pickle=False)
        steps = np.load(folder / "env_step.npy", allow_pickle=False)
        stride = np.load(folder / "source_stride.npy", allow_pickle=False)
        actions = np.load(folder / "action.npy", allow_pickle=False)
        names = meta["proprio_feature_names"]
        if proprio.ndim != 2 or proprio.shape != (n, len(names)) or not np.isfinite(proprio).all():
            raise ValueError("invalid robot proprioception")
        expected = [f"robot0.joint.{i}" for i in range(7)] + [f"robot0.gripper.{i}" for i in range(2)]
        if names != expected:
            raise ValueError("only fixed robot joint/gripper features are allowed")
        if not np.array_equal(steps, np.arange(n)) or not np.array_equal(stride, np.ones(n)):
            raise ValueError("expected contiguous pre-action1x source frames")
        if actions.shape != (n, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid aligned source actions")
        with imageio.get_reader(folder / "top_camera-images-rgb.mp4") as reader:
            images = np.asarray(
                [
                    np.asarray(Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR))
                    for frame in reader
                ],
                dtype=np.uint8,
            )
        if images.shape != (n, image_size, image_size, 3):
            raise ValueError("video/pre-action telemetry frame mismatch")
        labels = segment_labels(ann["segments"], phases, n)
        task = f"task_{inventory['identity']['task_id']:02d}"
        episode_id = f"{inventory['identity']['suite']}:{task}:episode_{ep:02d}"
        target = partial / "episodes" / f"episode_{ep:02d}"
        target.mkdir(parents=True)
        paths = {}
        for name, value in {"images": images, "proprio": proprio.astype(np.float32), "labels": labels}.items():
            path = target / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            paths[name] = str(path.relative_to(partial))
        split = splits[ep]
        records.append(
            dict(
                episode_id=episode_id,
                task=task,
                split=split,
                num_frames=n,
                fps=meta["fps"],
                control_hz=20,
                **paths,
                sha256={k: sha(partial / v) for k, v in paths.items()},
                source_metadata_sha256=row["metadata_sha256"],
                initial_state=row["initial_state"],
                source_pins=row["source_pins"],
                candidate_review_status=AUTHORITY,
            )
        )
        if split == "train":
            train_classes.update(labels.tolist())
            pixels = images.reshape(-1, 3).astype(np.float64) / 255
            values = proprio.astype(np.float64)
            sums["image"] += pixels.sum(0)
            sums["image2"] += np.square(pixels).sum(0)
            sums["image_n"] += len(pixels)
            if sums["proprio"] is None:
                sums["proprio"] = np.zeros(values.shape[1])
                sums["proprio2"] = np.zeros(values.shape[1])
            sums["proprio"] += values.sum(0)
            sums["proprio2"] += np.square(values).sum(0)
            sums["proprio_n"] += len(values)
    if {r["split"] for r in records} != {"train", "validation", "test"}:
        raise ValueError("each fixed episode split needs reviewed successful data")
    if train_classes != set(range(len(phases))):
        raise ValueError("fixed training split lacks reviewed examples for a phase")
    normalization = {"fit_split": "train"}
    for key in ("image", "proprio"):
        mean = sums[key] / sums[f"{key}_n"]
        std = np.sqrt(np.maximum(0, sums[f"{key}2"] / sums[f"{key}_n"] - mean * mean))
        if np.any(std <= 0):
            raise ValueError("training normalization has zero variance")
        normalization[f"{key}_mean"] = mean.tolist()
        normalization[f"{key}_std"] = std.tolist()
    manifest = {
        "schema": "strider-libero-prepared-phase-dataset-v1",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "identity": inventory["identity"],
        "preprocessing": {
            "schema": "strider-phase-preprocessing-v1",
            "image": {
                "dtype": "uint8",
                "layout": "HWC",
                "color": "RGB",
                "height": 84,
                "width": 84,
                "resize": "pillow-bilinear",
            },
            "proprio_feature_names": [f"robot0.joint.{i}" for i in range(7)]
            + [f"robot0.gripper.{i}" for i in range(2)],
        },
        "classes": [
            {
                "class_index": i,
                "key": f"task_{inventory['identity']['task_id']:02d}:{phase}",
                "name": phase,
                "source_phase_id": i,
                "task": f"task_{inventory['identity']['task_id']:02d}",
            }
            for i, phase in enumerate(phases)
        ],
        "episodes": records,
        "excluded_unsuccessful_source_episodes": excluded,
        "normalization": normalization,
        "split_strategy": {"name": "fixed_fit_bank_before_outcomes", "seed": split_seed, "episode_assignments": splits},
        "provenance": {
            "inventory_sha256": sha(inventory_path),
            "annotations_sha256": sha(annotations_path),
            "ontology_sha256": sha(ontology_path),
            "generator_sha256": sha(Path(__file__)),
        },
        "runtime_inputs": "Pre-action RGB plus robot joint/gripper positions only; no actions, contacts, object poses or goal signals",
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    published_ontology = dict(
        ontology,
        schema="strider-libero-task-phase-ontology-v1",
        authority=AUTHORITY,
        requires_human_confirmation=True,
        task_id=inventory["identity"]["task_id"],
    )
    (partial / "ontology_v1.json").write_text(json.dumps(published_ontology, indent=2, sort_keys=True) + "\n")
    (partial / "reviewed_annotations.json").write_bytes(annotations_path.read_bytes())
    (partial / "source_inventory.json").write_bytes(inventory_path.read_bytes())
    receipt = {
        "passed": True,
        "manifest_sha256": sha(partial / "manifest.json"),
        "episodes": len(records),
        "physical_steps": 0,
        "training_steps": 0,
    }
    (partial / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    partial.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--minimum-successes", type=int, default=15)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.inventory,
                args.annotations,
                args.output,
                image_size=args.image_size,
                split_seed=args.split_seed,
                minimum_successes=args.minimum_successes,
            )
        )
    )


if __name__ == "__main__":
    main()
