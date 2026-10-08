"""Prepare task-specific LIBERO-Long phase-predictor datasets.

The phase boundaries are derived from causal gripper commands in successful
native rollouts.  They are reviewable AI-candidate annotations, not human
ground truth.  Failed native episodes are preserved in the source telemetry
but deliberately excluded from predictor fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw

AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"


@dataclass(frozen=True)
class TaskSpec:
    description: str
    phases: tuple[str, ...]
    pattern: str


TASK_SPECS = {
    0: TaskSpec(
        "put both the alphabet soup and the tomato sauce in the basket",
        ("alphabet_soup_approach", "alphabet_soup_manipulate", "tomato_sauce_approach", "tomato_sauce_manipulate"),
        "two_object",
    ),
    1: TaskSpec(
        "put both the cream cheese box and the butter in the basket",
        ("cream_cheese_approach", "cream_cheese_manipulate", "butter_approach", "butter_manipulate"),
        "two_object",
    ),
    4: TaskSpec(
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        ("white_mug_approach", "white_mug_manipulate", "yellow_white_mug_approach", "yellow_white_mug_manipulate"),
        "two_object",
    ),
    5: TaskSpec(
        "pick up the book and place it in the back compartment of the caddy",
        ("book_approach", "book_manipulate"),
        "one_object",
    ),
    6: TaskSpec(
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        ("white_mug_approach", "white_mug_manipulate", "pudding_approach", "pudding_manipulate"),
        "two_object",
    ),
    7: TaskSpec(
        "put both the alphabet soup and the cream cheese box in the basket",
        ("alphabet_soup_approach", "alphabet_soup_manipulate", "cream_cheese_approach", "cream_cheese_manipulate"),
        "two_object",
    ),
    8: TaskSpec(
        "put both moka pots on the stove",
        ("moka_1_approach", "moka_1_manipulate", "moka_2_approach", "moka_2_manipulate"),
        "two_object",
    ),
    9: TaskSpec(
        "put the yellow and white mug in the microwave and close it",
        ("mug_approach", "mug_manipulate", "microwave_close"),
        "one_object_then_close",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(task_id: int, episode_ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(
        episode_ids,
        key=lambda episode_id: (
            hashlib.sha256(f"{seed}:task_{task_id:02d}:{episode_id}".encode()).hexdigest(),
            episode_id,
        ),
    )
    train_count = round(len(ordered) * 0.6)
    validation_count = round(len(ordered) * 0.2)
    return {
        episode_id: (
            "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        )
        for index, episode_id in enumerate(ordered)
    }


def _runs(gripper_commands: np.ndarray) -> list[tuple[bool, int, int]]:
    commands = np.asarray(gripper_commands)
    if commands.ndim != 1 or not len(commands):
        raise ValueError("gripper commands must be a non-empty vector")
    closed = commands > 0
    starts = np.r_[0, np.flatnonzero(closed[1:] != closed[:-1]) + 1]
    stops = np.r_[starts[1:], len(closed)]
    return [(bool(closed[start]), int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _debounced_runs(gripper_commands: np.ndarray, minimum_hold: int) -> list[tuple[bool, int, int]]:
    closed = np.asarray(gripper_commands) > 0
    while True:
        runs = _runs(closed.astype(np.int8))
        replacement = next(
            (
                (start, stop, runs[index - 1][0])
                for index, (_, start, stop) in enumerate(runs[1:-1], start=1)
                if stop - start < minimum_hold and runs[index - 1][0] == runs[index + 1][0]
            ),
            None,
        )
        if replacement is None:
            return runs
        start, stop, value = replacement
        closed[start:stop] = value


def _boundaries(gripper_commands: np.ndarray, pattern: str, *, minimum_hold: int = 10) -> tuple[int, ...]:
    runs = _debounced_runs(gripper_commands, minimum_hold)
    if pattern == "one_object":
        candidates = [start for closed, start, stop in runs if closed and stop - start >= minimum_hold]
        if not candidates:
            raise ValueError("no sustained object grasp")
        return (candidates[-1],)

    if pattern == "one_object_then_close":
        for index, (closed, start, stop) in enumerate(runs[:-1]):
            if not closed or stop - start < minimum_hold:
                continue
            for next_closed, release, release_stop in runs[index + 1 :]:
                if not next_closed and release_stop - release >= minimum_hold:
                    return start, release
        raise ValueError("no sustained grasp-release pair")

    if pattern != "two_object":
        raise ValueError(f"unknown task pattern {pattern!r}")

    first_pair = None
    for index, (closed, start, stop) in enumerate(runs[:-1]):
        next_closed, release, release_stop = runs[index + 1]
        if closed and not next_closed and stop - start >= minimum_hold and release_stop - release >= minimum_hold:
            first_pair = (start, release, index + 1)
            break
    if first_pair is None:
        raise ValueError("no sustained first-object grasp-release pair")
    first_close, first_release, release_run_index = first_pair

    second_candidates = [
        start for closed, start, stop in runs[release_run_index + 1 :] if closed and stop - start >= minimum_hold
    ]
    if not second_candidates:
        raise ValueError("no sustained second-object grasp")
    return first_close, first_release, second_candidates[-1]


def _decode(path: Path, image_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    reader = imageio.get_reader(path)
    full_frames = []
    resized = []
    try:
        for raw_frame in reader:
            frame = np.asarray(raw_frame, dtype=np.uint8)
            full_frames.append(frame)
            resized.append(
                np.asarray(Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR))
            )
    finally:
        reader.close()
    return np.asarray(resized, dtype=np.uint8), full_frames


def _labels(length: int, boundaries: tuple[int, ...]) -> np.ndarray:
    if not boundaries or boundaries[0] <= 0 or boundaries[-1] >= length:
        raise ValueError(f"invalid boundaries {boundaries} for {length} frames")
    if any(left >= right for left, right in pairwise(boundaries)):
        raise ValueError(f"boundaries are not strictly ordered: {boundaries}")
    labels = np.empty(length, dtype=np.int64)
    starts = (0, *boundaries)
    stops = (*boundaries, length)
    for phase_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        labels[start:stop] = phase_id
    return labels


def _review_sheet(
    frames: list[np.ndarray], boundaries: tuple[int, ...], phases: tuple[str, ...], destination: Path
) -> None:
    selected = {0, len(frames) - 1}
    for boundary in boundaries:
        selected.update(max(0, min(len(frames) - 1, boundary + offset)) for offset in (-8, -3, 0, 3, 8))
    indices = sorted(selected)
    columns = 5
    cell_width = 224
    cell_height = 244
    rows = (len(indices) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    draw = ImageDraw.Draw(sheet)
    boundary_names = dict(zip(boundaries, phases[1:], strict=True))
    for item, frame_index in enumerate(indices):
        frame = Image.fromarray(frames[frame_index]).resize((cell_width, cell_width), Image.Resampling.BILINEAR)
        left = (item % columns) * cell_width
        top = (item // columns) * cell_height
        sheet.paste(frame, (left, top + 20))
        suffix = f" -> {boundary_names[frame_index]}" if frame_index in boundary_names else ""
        draw.text((left + 4, top + 3), f"frame {frame_index}{suffix}", fill="yellow")
    sheet.save(destination)


def prepare(args: argparse.Namespace) -> dict:
    spec = TASK_SPECS[args.task_id]
    output = args.output.resolve()
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to replace existing output or partial output for {output}")
    partial.mkdir(parents=True)
    (partial / "review_sheets").mkdir()

    source_episodes = []
    excluded = []
    for source in sorted(args.telemetry_root.glob(f"task_{args.task_id:02d}_episode_*")):
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata.get("task_id") != args.task_id:
            raise ValueError(f"{source.name}: task ID mismatch")
        if metadata.get("task_description") != spec.description:
            raise ValueError(f"{source.name}: task description mismatch")
        if metadata.get("success") is True:
            source_episodes.append(source)
        else:
            excluded.append(source.name)
    if len(source_episodes) < args.minimum_successes:
        raise ValueError(
            f"Task {args.task_id} has {len(source_episodes)} successful episodes; "
            f"need at least {args.minimum_successes}"
        )
    splits = _split(args.task_id, [source.name for source in source_episodes], args.split_seed)

    manifest_episodes = []
    annotation_rows = []
    train_images = []
    train_proprio = []
    for source in source_episodes:
        episode_id = source.name
        actions = np.load(source / "action.npy", allow_pickle=False)
        proprio = np.asarray(np.load(source / "proprio.npy", allow_pickle=False), dtype=np.float32)
        boundaries = _boundaries(actions[:, -1], spec.pattern, minimum_hold=args.minimum_hold)
        images, full_frames = _decode(source / args.camera_file, args.image_size)
        if len(images) != len(actions) or len(proprio) != len(actions):
            raise ValueError(f"{episode_id}: video/action/proprio frame mismatch")
        labels = _labels(len(images), boundaries)
        if int(labels.max()) + 1 != len(spec.phases):
            raise ValueError(f"{episode_id}: phase-count mismatch")

        episode_dir = partial / "episodes" / f"task_{args.task_id:02d}" / episode_id
        episode_dir.mkdir(parents=True)
        paths = {
            "images": episode_dir / "images.npy",
            "proprio": episode_dir / "proprio.npy",
            "labels": episode_dir / "labels.npy",
        }
        np.save(paths["images"], images, allow_pickle=False)
        np.save(paths["proprio"], proprio, allow_pickle=False)
        np.save(paths["labels"], labels, allow_pickle=False)
        review = partial / "review_sheets" / f"{episode_id}_phase_review.png"
        _review_sheet(full_frames, boundaries, spec.phases, review)
        split = splits[episode_id]
        if split == "train":
            train_images.append(images)
            train_proprio.append(proprio)

        annotation_rows.append(
            {
                "schema": "strider-libero-task-phase-candidate-v1",
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
                "episode_id": episode_id,
                "task_id": args.task_id,
                "task_description": spec.description,
                "phase_order": list(spec.phases),
                "boundary_frames": list(boundaries),
                "boundary_evidence": "debounced causal gripper-command transitions",
                "review_status": AUTHORITY,
                "review_sheet": str(review.relative_to(partial)),
            }
        )
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "task": f"task_{args.task_id:02d}",
                "split": split,
                "fps": 10.0,
                "num_frames": len(images),
                **{name: str(path.relative_to(partial)) for name, path in paths.items()},
                "source_episode_path": str(source.resolve()),
                "candidate_review_status": AUTHORITY,
                "requires_human_confirmation": True,
                "sha256": {name: sha256_file(path) for name, path in paths.items()},
            }
        )

    image_values = np.concatenate(train_images).astype(np.float64) / 255.0
    proprio_values = np.concatenate(train_proprio).astype(np.float64)
    image_mean = image_values.mean(axis=(0, 1, 2))
    image_std = image_values.std(axis=(0, 1, 2))
    proprio_mean = proprio_values.mean(axis=0)
    proprio_std = proprio_values.std(axis=0)
    if np.any(image_std <= 0) or np.any(proprio_std <= 0):
        raise ValueError("training normalization has a zero-variance feature")

    annotations = partial / "annotations_v1.jsonl"
    annotations.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in annotation_rows))
    ontology = {
        "schema": "strider-libero-task-phase-ontology-v1",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "task_id": args.task_id,
        "task_description": spec.description,
        "phases": list(spec.phases),
        "runtime_inputs": "RGB plus causal robot proprioception; no object state or future success",
    }
    ontology_path = partial / "ontology_v1.json"
    ontology_path.write_text(json.dumps(ontology, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "strider-libero-prepared-phase-dataset-v1",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "classes": [
            {
                "class_index": index,
                "key": f"task_{args.task_id:02d}:{phase}",
                "name": phase,
                "source_phase_id": index,
                "task": f"task_{args.task_id:02d}",
            }
            for index, phase in enumerate(spec.phases)
        ],
        "episodes": manifest_episodes,
        "excluded_unsuccessful_source_episodes": excluded,
        "normalization": {
            "fit_split": "train",
            "image_mean": image_mean.tolist(),
            "image_std": image_std.tolist(),
            "proprio_mean": proprio_mean.tolist(),
            "proprio_std": proprio_std.tolist(),
        },
        "split_frame_counts": {
            split: sum(row["num_frames"] for row in manifest_episodes if row["split"] == split)
            for split in ("train", "validation", "test")
        },
        "split_strategy": {
            "name": "sha256_episode_split",
            "seed": args.split_seed,
            "ratios": [0.6, 0.2, 0.2],
        },
    }
    manifest_path = partial / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "strider-libero-phase-preparation-receipt-v1",
        "authority": AUTHORITY,
        "task_id": args.task_id,
        "episode_count": len(manifest_episodes),
        "excluded_unsuccessful_episode_count": len(excluded),
        "phase_count": len(spec.phases),
        "manifest_sha256": sha256_file(manifest_path),
        "annotations_sha256": sha256_file(annotations),
        "ontology_sha256": sha256_file(ontology_path),
        "generator_sha256": sha256_file(Path(__file__)),
    }
    (partial / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-file", default="top_camera-images-rgb.mp4")
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--minimum-hold", type=int, default=10)
    parser.add_argument("--minimum-successes", type=int, default=15)
    args = parser.parse_args()
    if args.image_size < 1 or args.minimum_hold < 1 or args.minimum_successes < 1:
        parser.error("image-size, minimum-hold, and minimum-successes must be positive")
    if not args.telemetry_root.is_dir():
        parser.error("telemetry-root must exist")
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
