#!/usr/bin/env python3
"""Stream only the two COCO annotation files from the nested CarFusion archive."""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path


TARGETS = {
    "carfusion/annotations/car_keypoints_train.json",
    "carfusion/annotations/car_keypoints_test.json",
}


def summarize(payload: dict) -> dict:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    groups = Counter(str(Path(image.get("file_name", "")).parent) for image in images)
    image_by_id = {image.get("id"): image for image in images}
    keypoint_lengths = Counter(len(annotation.get("keypoints", [])) for annotation in annotations)
    visibility = Counter()
    bbox_invalid = 0
    bbox_outside_image = 0
    invalid_bbox_examples = []
    for annotation in annotations:
        points = annotation.get("keypoints", [])
        visibility.update(points[2::3])
        bbox = annotation.get("bbox", [])
        invalid = len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0
        bbox_invalid += int(invalid)
        if invalid and len(invalid_bbox_examples) < 10:
            invalid_bbox_examples.append(
                {"id": annotation.get("id"), "image_id": annotation.get("image_id"), "bbox": bbox}
            )
        if len(bbox) == 4 and annotation.get("image_id") in image_by_id:
            image = image_by_id[annotation["image_id"]]
            x, y, width, height = bbox
            bbox_outside_image += int(
                x < 0 or y < 0 or x + width > image.get("width", 0) or y + height > image.get("height", 0)
            )
    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": categories,
        "group_image_counts": dict(sorted(groups.items())),
        "keypoint_value_lengths": dict(sorted(keypoint_lengths.items())),
        "visibility_counts": dict(sorted(visibility.items())),
        "invalid_bboxes": bbox_invalid,
        "bboxes_outside_image": bbox_outside_image,
        "invalid_bbox_examples": invalid_bbox_examples,
        "image_examples": [image.get("file_name") for image in images[:5]],
        "annotation_examples": [
            {
                "id": annotation.get("id"),
                "image_id": annotation.get("image_id"),
                "category_id": annotation.get("category_id"),
                "bbox": annotation.get("bbox"),
                "num_keypoints": annotation.get("num_keypoints"),
                "keypoints": annotation.get("keypoints"),
            }
            for annotation in annotations[:2]
        ],
    }


def probe(archive: Path) -> dict:
    result = {}
    with gzip.open(archive, "rb") as compressed:
        outer_headers = compressed.read(1024)
        if not outer_headers.startswith(b"CarFusion/"):
            raise ValueError("unexpected outer archive layout")
        with tarfile.open(fileobj=compressed, mode="r|") as nested:
            for member in nested:
                if member.name not in TARGETS:
                    continue
                stream = nested.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {member.name}")
                result[Path(member.name).stem] = summarize(json.load(stream))
                if len(result) == len(TARGETS):
                    break
    if len(result) != len(TARGETS):
        raise ValueError(f"missing annotations: {sorted(TARGETS - set(result))}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
