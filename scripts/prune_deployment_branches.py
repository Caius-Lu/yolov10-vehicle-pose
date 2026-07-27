#!/usr/bin/env python3
"""Build an inference-only Vehicle Pose checkpoint without one-to-many head towers."""

import argparse
import json
from pathlib import Path

import torch

from ultralytics.models.yolov10.prune_distill import (
    build_pruned_checkpoint,
    prune_one2many_deployment_branches,
)
from ultralytics.nn.tasks import YOLOv10PoseModel, attempt_load_one_weight


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    if checkpoint.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source checkpoint")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")

    device = torch.device(args.device)
    model, _ = attempt_load_one_weight(checkpoint, device=device, fuse=False)
    if not isinstance(model, YOLOv10PoseModel):
        raise TypeError(f"expected YOLOv10PoseModel checkpoint, got {type(model).__name__}")
    model.float().eval()
    model.model[-1].one2one_only = True
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        expected = model(image)["one2one"][0]
    model, report = prune_one2many_deployment_branches(model)
    with torch.inference_mode():
        actual = model(image)["one2one"][0]
    difference = (expected - actual).abs()
    report.update(
        source_checkpoint=str(checkpoint),
        output_checkpoint=str(output),
        dense_shape=list(actual.shape),
        dense_max_abs_error=float(difference.max()),
        dense_mean_abs_error=float(difference.mean()),
        output_exact=bool(torch.equal(expected, actual)),
    )
    if not report["output_exact"]:
        raise RuntimeError("deployment branch pruning changed one-to-one output")

    model.cpu().float()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_pruned_checkpoint(model, checkpoint, report), output)
    report["source_checkpoint_bytes"] = checkpoint.stat().st_size
    report["output_checkpoint_bytes"] = output.stat().st_size
    report["checkpoint_reduction_ratio"] = 1.0 - output.stat().st_size / checkpoint.stat().st_size
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
