#!/usr/bin/env python3
"""Paired 640/512 CUDA timing in one process to reduce clock/order bias."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from ultralytics import YOLOv10

from benchmark_torch import describe, measure


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    wrapper = YOLOv10("ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml", task="pose")
    wrapper.load(args.checkpoint)
    model = wrapper.model.to(device).eval()
    tensors = {size: torch.rand(1, 3, size, size, device=device) for size in (640, 512)}
    records = []

    # Warm both shapes before alternating AB/BA order within the same process.
    for tensor in tensors.values():
        measure(model, tensor, args.warmup, 1)
    for mode, one2one_only in (("torch_original", False), ("torch_one2one_only", True)):
        model.model[-1].one2one_only = one2one_only
        for round_index in range(args.rounds):
            order = (640, 512) if round_index % 2 == 0 else (512, 640)
            for imgsz in order:
                result = describe(measure(model, tensors[imgsz], 20, args.iterations))
                result.update(mode=mode, round=round_index + 1, imgsz=imgsz, order=list(order))
                records.append(result)

    summary = {}
    for mode in ("torch_original", "torch_one2one_only"):
        means = {
            size: statistics.fmean(r["mean_ms"] for r in records if r["mode"] == mode and r["imgsz"] == size)
            for size in (640, 512)
        }
        summary[mode] = {
            "mean_ms_640": means[640],
            "mean_ms_512": means[512],
            "latency_reduction_pct": (means[640] - means[512]) / means[640] * 100,
            "speedup": means[640] / means[512],
        }
    payload = {"environment": {"device": torch.cuda.get_device_name(device)}, "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
