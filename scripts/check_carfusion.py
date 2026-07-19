#!/usr/bin/env python3
"""Validate converted CarFusion YOLO pose labels and group split integrity."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image


BBOX_SERIALIZATION_TOLERANCE = 1e-6


def _describe(values):
    """Return stable compact distribution statistics for a numeric sequence."""
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(ratio):
        return ordered[round((len(ordered) - 1) * ratio)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def check_dataset(root):
    root = Path(root)
    errors, warnings = [], []
    summary_path = root / "conversion_summary.json"
    split_path = root / "split_summary.json"
    if not summary_path.is_file() or not split_path.is_file():
        return {"status": "ERROR", "errors": ["conversion or split summary is missing"], "warnings": []}
    conversion = json.loads(summary_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    group_sets = [set(split["groups"].get(name, [])) for name in ("train", "val", "test")]
    group_leakage = bool(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    if group_leakage:
        errors.append("train/val/test group leakage")

    visibility = Counter()
    keypoint_stats = {
        str(index): {"visibility": Counter({0: 0, 1: 0, 2: 0}), "x": [], "y": [], "outside_bbox": 0, "valid": 0}
        for index in range(14)
    }
    cars_per_image = []
    bbox_widths, bbox_heights, bbox_areas, bbox_pixel_areas = [], [], [], []
    vehicle_scales = Counter({"small": 0, "medium": 0, "large": 0})
    tiny_vehicles = duplicate_instances = 0
    image_count = label_count = instance_count = 0
    for split_name in ("train", "val", "test"):
        image_dir, label_dir = root / "images" / split_name, root / "labels" / split_name
        images = sorted(image_dir.glob("*"))
        labels = sorted(label_dir.glob("*.txt"))
        image_count += len(images)
        label_count += len(labels)
        image_sizes = {}
        if len(images) != len(labels):
            errors.append(f"{split_name}: image/label count mismatch {len(images)} != {len(labels)}")
        for image in images:
            try:
                with Image.open(image) as opened:
                    image_sizes[image.stem] = opened.size
                    opened.verify()
            except Exception as exc:
                errors.append(f"corrupt image {image}: {exc}")
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            cars_per_image.append(len(lines))
            duplicate_instances += len(lines) - len(set(lines))
            if not lines:
                warnings.append(f"empty label: {label}")
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 47:
                    errors.append(f"{label}:{line_number}: expected 47 fields, got {len(fields)}")
                    continue
                try:
                    values = [float(value) for value in fields]
                except ValueError:
                    errors.append(f"{label}:{line_number}: non-numeric field")
                    continue
                if not all(math.isfinite(value) for value in values):
                    errors.append(f"{label}:{line_number}: non-finite field")
                    continue
                if values[0] != 0:
                    errors.append(f"{label}:{line_number}: class must be 0")
                center_x, center_y, width, height = values[1:5]
                if (
                    any(not 0 <= value <= 1 for value in values[1:5])
                    or width <= 0
                    or height <= 0
                    or center_x - width / 2 < -BBOX_SERIALIZATION_TOLERANCE
                    or center_y - height / 2 < -BBOX_SERIALIZATION_TOLERANCE
                    or center_x + width / 2 > 1 + BBOX_SERIALIZATION_TOLERANCE
                    or center_y + height / 2 > 1 + BBOX_SERIALIZATION_TOLERANCE
                ):
                    errors.append(f"{label}:{line_number}: bbox outside normalized range")
                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(width * height)
                image_width, image_height = image_sizes.get(label.stem, (0, 0))
                pixel_area = width * image_width * height * image_height
                bbox_pixel_areas.append(pixel_area)
                if pixel_area < 32**2:
                    vehicle_scales["small"] += 1
                elif pixel_area < 96**2:
                    vehicle_scales["medium"] += 1
                else:
                    vehicle_scales["large"] += 1
                tiny_vehicles += int(pixel_area < 16**2)
                for index in range(14):
                    x, y, visible = values[5 + index * 3 : 8 + index * 3]
                    if visible not in (0, 1, 2):
                        errors.append(f"{label}:{line_number}: invalid visibility {visible}")
                        continue
                    visible = int(visible)
                    visibility[visible] += 1
                    point = keypoint_stats[str(index)]
                    point["visibility"][visible] += 1
                    if visible:
                        if not (0 <= x <= 1 and 0 <= y <= 1):
                            errors.append(f"{label}:{line_number}: keypoint {index} outside normalized range")
                        point["x"].append(x)
                        point["y"].append(y)
                        point["valid"] += 1
                        point["outside_bbox"] += int(
                            x < center_x - width / 2
                            or x > center_x + width / 2
                            or y < center_y - height / 2
                            or y > center_y + height / 2
                        )
                instance_count += 1
    if image_count != conversion.get("images"):
        errors.append(f"image total differs from conversion summary: {image_count} != {conversion.get('images')}")
    expected_instances = conversion.get("converted_annotations", conversion.get("annotations"))
    if instance_count != expected_instances:
        errors.append(
            f"annotation total differs from conversion summary: {instance_count} != {expected_instances}"
        )
    keypoints = {}
    for index, point in keypoint_stats.items():
        keypoints[index] = {
            "visibility": {str(value): point["visibility"][value] for value in (0, 1, 2)},
            "x": _describe(point["x"]),
            "y": _describe(point["y"]),
            "valid": point["valid"],
            "outside_bbox": point["outside_bbox"],
            "outside_bbox_ratio": point["outside_bbox"] / point["valid"] if point["valid"] else 0.0,
        }
    if duplicate_instances:
        errors.append(f"duplicate label instances: {duplicate_instances}")
    return {
        "status": "PASS" if not errors else "ERROR",
        "images": image_count,
        "labels": label_count,
        "instances": instance_count,
        "visibility": dict(visibility),
        "group_leakage": group_leakage,
        "cars_per_image": _describe(cars_per_image),
        "bbox": {
            "width_normalized": _describe(bbox_widths),
            "height_normalized": _describe(bbox_heights),
            "area_normalized": _describe(bbox_areas),
            "area_pixels": _describe(bbox_pixel_areas),
            "scales": dict(vehicle_scales),
            "tiny_count": tiny_vehicles,
            "tiny_ratio": tiny_vehicles / instance_count if instance_count else 0.0,
        },
        "keypoints": keypoints,
        "duplicate_instances": duplicate_instances,
        "errors": errors,
        "warnings": warnings,
    }


def main():  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/CarFusion/yolo_pose"))
    parser.add_argument("--report", type=Path, default=Path("reports/data/05_quality_results.json"))
    args = parser.parse_args()
    result = check_dataset(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
