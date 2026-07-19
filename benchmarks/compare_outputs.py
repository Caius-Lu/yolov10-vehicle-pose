#!/usr/bin/env python3
"""Compare original and one-to-one-only YOLOv10 vehicle-pose dense outputs."""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLOv10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml")
    parser.add_argument("--checkpoint", default="weights/yolov10n.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    wrapper = YOLOv10(args.model, task="pose")
    if Path(args.checkpoint).is_file():
        wrapper.load(args.checkpoint)
    model = wrapper.model.to(args.device).eval()
    head = model.model[-1]
    tensor = torch.rand(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.inference_mode():
        head.one2one_only = False
        original = model(tensor)["one2one"][0]
        head.one2one_only = True
        optimized = model(tensor)["one2one"][0]
    difference = (original - optimized).abs()
    result = {
        "shape_equal": original.shape == optimized.shape,
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "nan_or_inf": not (torch.isfinite(original).all() and torch.isfinite(optimized).all()).item(),
        "allclose": torch.allclose(original, optimized, atol=1e-6, rtol=1e-5),
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["allclose"] and not result["nan_or_inf"] else 1)


if __name__ == "__main__":
    main()

