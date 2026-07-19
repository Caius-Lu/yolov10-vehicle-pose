#!/usr/bin/env python3
"""Render numbered CarFusion keypoint previews from converted YOLO pose labels."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def _records(root: Path) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        for image_path in sorted((root / "images" / split).glob("*")):
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            areas = [float(line[3]) * float(line[4]) for line in lines if len(line) == 47]
            visibility = [int(float(value)) for line in lines if len(line) == 47 for value in line[7::3]]
            view_match = re.search(r"(car_[A-Za-z0-9_-]+[12])", image_path.name)
            records.append(
                {
                    "split": split,
                    "image": image_path,
                    "label": label_path,
                    "lines": lines,
                    "instances": len(lines),
                    "smallest_area": min(areas, default=1.0),
                    "occluded": 1 in visibility,
                    "view": view_match.group(1) if view_match else "unknown",
                }
            )
    return records


def _select(records: list[dict], limit: int, seed: int) -> list[dict]:
    selected = {}
    reasons = defaultdict(set)

    def add(record, reason):
        key = str(record["image"])
        if len(selected) < limit or key in selected:
            selected[key] = record
            reasons[key].add(reason)

    by_view = {}
    for record in records:
        by_view.setdefault(record["view"], record)
    for record in by_view.values():
        add(record, "view")
    for record in sorted(records, key=lambda item: item["smallest_area"])[: max(1, limit // 5)]:
        add(record, "small")
    for record in [item for item in records if item["occluded"]][: max(1, limit // 5)]:
        add(record, "occluded")
    for record in sorted(records, key=lambda item: item["instances"], reverse=True)[: max(1, limit // 5)]:
        add(record, "multi_vehicle")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    for record in shuffled:
        add(record, "random")
        if len(selected) >= limit:
            break
    output = []
    for key, record in list(selected.items())[:limit]:
        record = dict(record)
        record["reasons"] = sorted(reasons[key])
        output.append(record)
    return output


def _draw(record: dict, output: Path) -> None:
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for instance_index, fields in enumerate(record["lines"]):
        if len(fields) != 47:
            continue
        values = [float(value) for value in fields]
        center_x, center_y, box_width, box_height = values[1:5]
        box = (
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        )
        draw.rectangle(box, outline=(0, 255, 0), width=3)
        draw.text((box[0], max(0, box[1] - 12)), f"car {instance_index}", fill=(0, 255, 0))
        for keypoint_index in range(14):
            x, y, visibility = values[5 + keypoint_index * 3 : 8 + keypoint_index * 3]
            if not visibility:
                continue
            point = (x * width, y * height)
            color = (0, 180, 255) if visibility == 2 else (255, 140, 0)
            radius = 4
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
            draw.text((point[0] + 5, point[1] - 8), str(keypoint_index), fill=color)
    image.thumbnail((1280, 1280))
    image.save(output, quality=88)


def render_previews(root: Path, output: Path, limit: int = 100, seed: int = 0) -> list[dict]:
    """Render a deterministic, category-aware preview set and return its manifest."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select(_records(root), limit=limit, seed=seed)
    manifest = []
    for index, record in enumerate(selected):
        target = output / f"{index:03d}_{record['split']}_{record['image'].name}"
        _draw(record, target)
        manifest.append(
            {
                "source": str(record["image"]),
                "output": str(target),
                "split": record["split"],
                "instances": record["instances"],
                "view": record["view"],
                "reasons": record["reasons"],
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", default=Path("reports/data/preview"), type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    print(json.dumps({"previews": len(render_previews(args.root, args.output, args.limit, args.seed))}, indent=2))


if __name__ == "__main__":
    main()
