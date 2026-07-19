#!/usr/bin/env python3
"""Convert authorized CarFusion COCO keypoints to deterministic YOLO pose data."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image


KEYPOINT_COUNT = 14


class InvalidBBoxError(ValueError):
    """Raised when an annotation has no trainable bounding box."""


def derive_group(file_name):
    """Derive a leakage boundary from directory hierarchy or a non-numeric frame prefix."""
    path = PurePosixPath(str(file_name).replace("\\", "/"))
    for part in path.parts[:-1]:
        match = re.match(r"^(car_.+?)[12]$", part)
        if match:
            return match.group(1)
    if len(path.parts) > 1:
        return "/".join(path.parts[:-1])
    match = re.match(r"^(.+?)[_-]\d+$", path.stem)
    return match.group(1) if match else None


def normalize_annotation(annotation, width, height):
    """Return one validated YOLO pose label and non-fatal quality issues."""
    annotation_id = annotation.get("id", "unknown")
    bbox = annotation.get("bbox", [])
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        raise InvalidBBoxError(f"annotation {annotation_id}: invalid bbox")
    x, y, box_width, box_height = map(float, bbox)
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise InvalidBBoxError(f"annotation {annotation_id}: invalid bbox outside image")

    keypoints = annotation.get("keypoints", [])
    if len(keypoints) != KEYPOINT_COUNT * 3:
        raise ValueError(f"annotation {annotation_id}: expected 14 keypoints")
    normalized = []
    issues = []
    for index in range(KEYPOINT_COUNT):
        keypoint_x, keypoint_y, visibility = keypoints[index * 3 : index * 3 + 3]
        if visibility not in (0, 1, 2):
            raise ValueError(f"annotation {annotation_id}: invalid visibility {visibility}")
        if visibility == 0:
            normalized.extend((0.0, 0.0, 0))
        elif not (0 <= keypoint_x <= width and 0 <= keypoint_y <= height):
            normalized.extend((0.0, 0.0, 0))
            issues.append(f"annotation {annotation_id}: keypoint {index} outside image; marked invalid")
        else:
            normalized.extend((float(keypoint_x) / width, float(keypoint_y) / height, int(visibility)))

    fields = [0, (x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height]
    output = [str(fields[0]), *(f"{value:.6f}" for value in fields[1:])]
    for index in range(KEYPOINT_COUNT):
        keypoint_x, keypoint_y, visibility = normalized[index * 3 : index * 3 + 3]
        output.extend((f"{keypoint_x:.6f}", f"{keypoint_y:.6f}", str(visibility)))
    return " ".join(output), issues


def split_groups(groups, seed=0, ratios=(0.7, 0.15, 0.15), fixed_test_groups=None):
    """Assign whole capture groups to deterministic train/val/test sets."""
    names = sorted(groups)
    if len(names) < 3:
        raise ValueError("at least three independent groups are required for train/val/test")
    fixed_test_groups = set(fixed_test_groups or ())
    unknown = fixed_test_groups - set(names)
    if unknown:
        raise ValueError(f"fixed test groups are missing: {sorted(unknown)}")
    if fixed_test_groups:
        remaining = sorted(set(names) - fixed_test_groups)
        if len(remaining) < 2:
            raise ValueError("fixed test policy leaves fewer than two train/val groups")
        random.Random(seed).shuffle(remaining)
        train_val_ratio = ratios[0] + ratios[1]
        val_count = max(1, round(len(remaining) * ratios[1] / train_val_ratio))
        val_count = min(val_count, len(remaining) - 1)
        return {
            "train": sorted(remaining[val_count:]),
            "val": sorted(remaining[:val_count]),
            "test": sorted(fixed_test_groups),
        }

    random.Random(seed).shuffle(names)
    train_count = min(max(1, round(len(names) * ratios[0])), len(names) - 2)
    remaining = len(names) - train_count
    val_count = min(max(1, round(len(names) * ratios[1])), remaining - 1)
    return {
        "train": names[:train_count],
        "val": names[train_count : train_count + val_count],
        "test": names[train_count + val_count :],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_coco_dataset(
    coco_json,
    images_root,
    output,
    seed=0,
    invalid_bbox_policy="error",
    fixed_test_groups=None,
    verify_images=True,
):
    """Convert a COCO-keypoint file while preserving group boundaries and traceability."""
    if invalid_bbox_policy not in {"error", "drop"}:
        raise ValueError("invalid_bbox_policy must be 'error' or 'drop'")
    coco_json, images_root, output = map(Path, (coco_json, images_root, output))
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = [image.get("id") for image in images]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image id")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate annotation id")

    groups = defaultdict(list)
    group_by_image = {}
    for image in images:
        group = derive_group(image["file_name"])
        if group is None:
            raise ValueError(
                f"cannot derive group for {image['file_name']}; create ACTION_REQUIRED_SPLIT_POLICY.md before conversion"
            )
        groups[group].append(image["id"])
        group_by_image[image["id"]] = group
    split_group_names = split_groups(groups, seed=seed, fixed_test_groups=fixed_test_groups)
    split_by_group = {
        group: split_name for split_name, group_names in split_group_names.items() for group in group_names
    }

    annotations_by_image = defaultdict(list)
    for annotation in annotations:
        if annotation.get("image_id") not in group_by_image:
            raise ValueError(f"annotation {annotation.get('id')} references missing image")
        annotations_by_image[annotation["image_id"]].append(annotation)

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    issues = []
    id_map = []
    split_images = Counter()
    split_annotations = Counter()
    visibility = Counter()
    list_entries = defaultdict(list)
    converted_annotations = 0
    dropped_invalid_bboxes = 0
    for image in sorted(images, key=lambda item: str(item["file_name"])):
        source = images_root / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"missing image: {source}")
        if verify_images:
            with Image.open(source) as opened:
                opened.verify()
        width, height = int(image["width"]), int(image["height"])
        split = split_by_group[group_by_image[image["id"]]]
        safe_name = image["file_name"].replace("/", "__").replace("\\", "__")
        target_image = output / "images" / split / safe_name
        if not target_image.exists():
            target_image.symlink_to(source.resolve())
        label_path = output / "labels" / split / f"{Path(safe_name).stem}.txt"
        labels = []
        for annotation in sorted(annotations_by_image[image["id"]], key=lambda item: item["id"]):
            try:
                label, annotation_issues = normalize_annotation(annotation, width, height)
            except InvalidBBoxError as error:
                if invalid_bbox_policy == "error":
                    raise
                dropped_invalid_bboxes += 1
                issues.append(f"{error}; dropped by explicit policy")
                id_map.append(
                    {
                        "image_id": image["id"],
                        "annotation_id": annotation["id"],
                        "split": split,
                        "status": "dropped_invalid_bbox",
                    }
                )
                continue
            labels.append(label)
            issues.extend(annotation_issues)
            split_annotations[split] += 1
            converted_annotations += 1
            visibility.update(annotation["keypoints"][2::3])
            id_map.append(
                {
                    "image_id": image["id"],
                    "annotation_id": annotation["id"],
                    "split": split,
                    "label": str(label_path.relative_to(output)),
                    "status": "converted",
                }
            )
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        split_images[split] += 1
        # This YOLOv10 loader resolves only './' entries relative to the list file.
        list_entries[split].append(f"./{target_image.relative_to(output)}")

    for split, entries in list_entries.items():
        (output / f"{split}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    split_summary = {
        "seed": seed,
        "groups": split_group_names,
        "group_counts": {name: len(value) for name, value in split_group_names.items()},
        "image_counts": dict(split_images),
        "leakage": False,
    }
    summary = {
        "status": "READY",
        "image_verification": "complete" if verify_images else "deferred_to_quality_gate",
        "source_json": str(coco_json.resolve()),
        "source_sha256": _sha256(coco_json),
        "images": len(images),
        "annotations": len(annotations),
        "converted_annotations": converted_annotations,
        "dropped_invalid_bboxes": dropped_invalid_bboxes,
        "invalid_bbox_policy": invalid_bbox_policy,
        "split_images": dict(split_images),
        "split_annotations": dict(split_annotations),
        "visibility": {str(key): value for key, value in sorted(visibility.items())},
        "issues": issues,
    }
    (output / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "id_map.json").write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    return summary


def parse_args():  # pragma: no cover - exercised through the documented CLI smoke command
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("datasets/CarFusion/yolo_pose"), type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--invalid-bbox-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--fixed-test-group", action="append", default=[])
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Defer image integrity verification to the immediately following quality gate.",
    )
    return parser.parse_args()


def main():  # pragma: no cover - thin CLI wrapper
    args = parse_args()
    summary = convert_coco_dataset(
        args.coco_json,
        args.images_root,
        args.output,
        seed=args.seed,
        invalid_bbox_policy=args.invalid_bbox_policy,
        fixed_test_groups=set(args.fixed_test_group),
        verify_images=not args.skip_image_verify,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
