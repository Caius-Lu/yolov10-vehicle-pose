#!/usr/bin/env python3
"""Run one tiny generated-data epoch to verify Trainer/Validator plumbing; not a quality benchmark."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

from PIL import Image
import yaml

from ultralytics import YOLOv10


def make_fixture(root):
    for split, count in (("train", 4), ("val", 2)):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (64, 64), color=(20 + index, 40, 60)).save(image_dir / f"{index}.jpg")
            keypoints = " ".join("0.5 0.5 2" for _ in range(14))
            (label_dir / f"{index}.txt").write_text(f"0 0.5 0.5 0.5 0.5 {keypoints}\n", encoding="utf-8")
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "vehicle"},
        "kpt_shape": [14, 3],
        "flip_idx": [],
    }
    yaml_path = root / "fixture.yaml"
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", default="weights/yolov10n.pt")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vehicle-pose-fixture-") as directory:
        data = make_fixture(Path(directory))
        model = YOLOv10("ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml", task="pose")
        if Path(args.weights).is_file():
            model.load(args.weights)
        model.train(
            data=str(data),
            epochs=1,
            batch=2,
            imgsz=64,
            device=args.device,
            workers=0,
            amp=False,
            plots=False,
            val=True,
            seed=0,
            deterministic=True,
            fliplr=0.0,
            mosaic=0.0,
            close_mosaic=0,
            project="fixture_smoke",
            name="one_epoch",
            exist_ok=True,
        )
    print("PASS: generated fixture completed one training epoch and validation; it is not real training data")


if __name__ == "__main__":
    main()
