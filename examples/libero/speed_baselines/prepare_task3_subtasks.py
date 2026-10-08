"""Prepare five-phase LIBERO-Long Task-3 predictor data from native telemetry.

The offline labels use causal robot signals available in the preserved native
rollouts. They are AI-candidate annotations, not human ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw

TASK_ID = 3
TASK_DESCRIPTION = "put the black bowl in the bottom drawer of the cabinet and close it"
AUTHORITY = "AI_CANDIDATE_NOT_HUMAN_ANNOTATION"
PHASES = (
    "bowl_approach",
    "bowl_acquire",
    "bowl_transport",
    "bowl_place",
    "drawer_close",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split(episode_ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(
        episode_ids,
        key=lambda episode_id: (
            hashlib.sha256(f"{seed}:task_03:{episode_id}".encode()).hexdigest(),
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


def _sustained_lift(eef: np.ndarray, close: int, release: int) -> int:
    threshold = float(eef[close, 2]) + 0.04
    for index in range(close + 1, release - 1):
        if np.all(eef[index : index + 3, 2] >= threshold):
            return index
    raise ValueError("no three-frame sustained 4 cm lift before release")


def _drawer_side_descent(eef: np.ndarray, lift: int, release: int) -> int:
    peak = float(np.max(eef[lift : release + 1, 2]))
    for index in range(lift + 1, release):
        if eef[index, 1] >= 0.09 and eef[index, 2] <= peak - 0.012 and eef[index, 2] <= eef[index - 1, 2]:
            return index
    raise ValueError("no drawer-side descending placement entry before release")


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


def _labels(length: int, boundaries: tuple[int, int, int, int]) -> np.ndarray:
    close, lift, place, release = boundaries
    if not 0 < close < lift < place < release < length:
        raise ValueError(f"invalid ordered boundaries {boundaries} for {length} frames")
    labels = np.empty(length, dtype=np.int64)
    for phase_id, (start, stop) in enumerate(
        zip(
            (0, close, lift, place, release),
            (close, lift, place, release, length),
            strict=True,
        )
    ):
        labels[start:stop] = phase_id
    return labels


def _review_sheet(frames: list[np.ndarray], boundaries: tuple[int, int, int, int], destination: Path) -> None:
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
    boundary_names = dict(zip(boundaries, PHASES[1:], strict=True))
    for item, frame_index in enumerate(indices):
        frame = Image.fromarray(frames[frame_index]).resize((cell_width, cell_width), Image.Resampling.BILINEAR)
        left = (item % columns) * cell_width
        top = (item // columns) * cell_height
        sheet.paste(frame, (left, top + 20))
        suffix = f" -> {boundary_names[frame_index]}" if frame_index in boundary_names else ""
        draw.text((left + 4, top + 3), f"frame {frame_index}{suffix}", fill="yellow")
    sheet.save(destination)


def prepare(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to replace existing output or partial output for {output}")
    partial.mkdir(parents=True)
    (partial / "review_sheets").mkdir()

    coarse_rows = {
        row["episode_id"]: row
        for row in _jsonl(args.coarse_annotations)
        if row.get("task_index") == TASK_ID and row.get("episode_outcome") == "success"
    }
    if len(coarse_rows) != 20:
        raise ValueError(f"expected 20 successful Task-3 annotation rows, got {len(coarse_rows)}")
    splits = _split(sorted(coarse_rows), args.split_seed)
    manifest_episodes = []
    annotation_rows = []
    train_images = []
    train_proprio = []

    for episode_id in sorted(coarse_rows):
        source = args.telemetry_root / episode_id
        coarse = coarse_rows[episode_id]
        close, release = (int(value) for value in coarse["candidate_boundary_frames"][:2])
        eef = np.load(source / "robot0-eef_pos.npy", allow_pickle=False)
        proprio = np.asarray(np.load(source / "proprio.npy", allow_pickle=False), dtype=np.float32)
        lift = _sustained_lift(eef, close, release)
        place = _drawer_side_descent(eef, lift, release)
        boundaries = (close, lift, place, release)
        images, full_frames = _decode(source / args.camera_file, args.image_size)
        if len(images) != len(eef) or len(proprio) != len(eef):
            raise ValueError(f"{episode_id}: video/proprio/eef frame mismatch")
        labels = _labels(len(images), boundaries)

        episode_dir = partial / "episodes" / "task_03" / episode_id
        episode_dir.mkdir(parents=True)
        paths = {
            "images": episode_dir / "images.npy",
            "proprio": episode_dir / "proprio.npy",
            "labels": episode_dir / "labels.npy",
        }
        np.save(paths["images"], images, allow_pickle=False)
        np.save(paths["proprio"], proprio, allow_pickle=False)
        np.save(paths["labels"], labels, allow_pickle=False)
        review = partial / "review_sheets" / f"{episode_id}_fine_phase_review.png"
        _review_sheet(full_frames, boundaries, review)
        split = splits[episode_id]
        if split == "train":
            train_images.append(images)
            train_proprio.append(proprio)

        annotation_rows.append(
            {
                "schema": "strider-libero-task3-fine-phase-candidate-v1",
                "authority": AUTHORITY,
                "requires_human_confirmation": True,
                "episode_id": episode_id,
                "task_id": TASK_ID,
                "task_description": TASK_DESCRIPTION,
                "phase_order": list(PHASES),
                "boundary_frames": list(boundaries),
                "boundary_evidence": {
                    "bowl_acquire": "first gripper-command transition",
                    "bowl_transport": "first three-frame sustained lift 4 cm above close height",
                    "bowl_place": "drawer-side descent after lift",
                    "drawer_close": "bowl-release gripper-command transition",
                },
                "review_status": "AI_CANDIDATE_NOT_HUMAN_ANNOTATION",
                "review_sheet": str(review.relative_to(partial)),
            }
        )
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "task": "task_03",
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
        "schema": "strider-libero-task3-fine-phase-ontology-v1",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "task_id": TASK_ID,
        "task_description": TASK_DESCRIPTION,
        "phases": list(PHASES),
        "runtime_inputs": "RGB plus causal robot proprioception; no object state or future success",
    }
    (partial / "ontology_v1.json").write_text(json.dumps(ontology, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "strider-libero-task3-prepared-phase-dataset-v1",
        "authority": AUTHORITY,
        "requires_human_confirmation": True,
        "classes": [
            {
                "class_index": index,
                "key": f"task_03:{phase}",
                "name": phase,
                "source_phase_id": index,
                "task": "task_03",
            }
            for index, phase in enumerate(PHASES)
        ],
        "episodes": manifest_episodes,
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
        "schema": "strider-libero-task3-phase-preparation-receipt-v1",
        "authority": AUTHORITY,
        "episode_count": len(manifest_episodes),
        "phase_count": len(PHASES),
        "manifest_sha256": sha256_file(manifest_path),
        "annotations_sha256": sha256_file(annotations),
        "source_coarse_annotations_sha256": sha256_file(args.coarse_annotations),
        "generator_sha256": sha256_file(Path(__file__)),
    }
    (partial / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--coarse-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-file", default="top_camera-images-rgb.mp4")
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--split-seed", type=int, default=1)
    args = parser.parse_args()
    if args.image_size < 1:
        parser.error("image-size must be positive")
    if not args.telemetry_root.is_dir() or not args.coarse_annotations.is_file():
        parser.error("telemetry-root and coarse-annotations must exist")
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
