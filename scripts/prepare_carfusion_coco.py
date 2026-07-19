#!/usr/bin/env python3
"""Combine the official CarFusion train/test COCO files into one traceable manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path, PurePosixPath


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prefix(file_name: str, split: str) -> str:
    path = PurePosixPath(str(file_name).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image path: {file_name}")
    return str(PurePosixPath(split) / path)


def combine_official_splits(train_json: Path, test_json: Path, output: Path) -> dict:
    """Prefix image paths, preserve source IDs, and make annotation IDs globally unique."""
    train, test = _load(train_json), _load(test_json)
    if train.get("categories") != test.get("categories"):
        raise ValueError("train/test categories differ")

    images = []
    image_ids = set()
    for split, payload in (("train", train), ("test", test)):
        for source in payload.get("images", []):
            image = deepcopy(source)
            image_id = image.get("id")
            if image_id in image_ids:
                raise ValueError(f"duplicate image id across official splits: {image_id}")
            image_ids.add(image_id)
            image["file_name"] = _prefix(image["file_name"], split)
            image["source_split"] = split
            images.append(image)

    annotations = []
    next_id = 0
    for split, payload in (("train", train), ("test", test)):
        for source in payload.get("annotations", []):
            annotation = deepcopy(source)
            annotation["source_annotation_id"] = annotation.get("id")
            annotation["source_split"] = split
            annotation["id"] = next_id
            next_id += 1
            annotations.append(annotation)

    combined = {
        "info": {"description": "Combined official CarFusion train/test manifest"},
        "images": images,
        "annotations": annotations,
        "categories": train.get("categories", []),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "train_images": len(train.get("images", [])),
        "test_images": len(test.get("images", [])),
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(combine_official_splits(args.train_json, args.test_json, args.output), indent=2))


if __name__ == "__main__":
    main()
