#!/usr/bin/env python3
"""Measure fused FP16 deployment latency per top-level model module with CUDA events."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from ultralytics.nn.tasks import attempt_load_one_weight


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("module latency profiling requires a CUDA device")
    model, _ = attempt_load_one_weight(args.checkpoint, device=device, fuse=False)
    model = model.to(device).eval()
    model.fuse()
    model.half()
    model.model[-1].one2one_only = True
    image = torch.rand(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
    torch.cuda.synchronize(device)

    state = {"starts": {}, "ends": {}}
    handles = []
    for index, module in enumerate(model.model):
        def pre_hook(_module, _inputs, layer_index=index):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            state["starts"][layer_index] = event

        def post_hook(_module, _inputs, _output, layer_index=index):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            state["ends"][layer_index] = event

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    samples = defaultdict(list)
    total_samples = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            state["starts"] = {}
            state["ends"] = {}
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            model(image)
            total_end.record()
            torch.cuda.synchronize(device)
            total_samples.append(total_start.elapsed_time(total_end))
            for index in range(len(model.model)):
                samples[index].append(state["starts"][index].elapsed_time(state["ends"][index]))
    for handle in handles:
        handle.remove()

    backbone_count = len(model.yaml.get("backbone", []))
    rows = []
    for index, module in enumerate(model.model):
        if index < backbone_count:
            stage = "backbone"
        elif index == len(model.model) - 1:
            stage = "detect_pose_head"
        else:
            stage = "neck"
        values = samples[index]
        rows.append(
            {
                "index": index,
                "type": module.__class__.__name__,
                "stage": stage,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
                "mean_ms": statistics.fmean(values),
                "p50_ms": percentile(values, 0.5),
                "p90_ms": percentile(values, 0.9),
            }
        )
    summed_module_mean_ms = sum(row["mean_ms"] for row in rows)
    for row in rows:
        row["module_time_share_pct"] = row["mean_ms"] / summed_module_mean_ms * 100

    stages = {}
    for stage in ("backbone", "neck", "detect_pose_head"):
        selected = [row for row in rows if row["stage"] == stage]
        mean_ms = sum(row["mean_ms"] for row in selected)
        stages[stage] = {
            "parameters": sum(row["parameters"] for row in selected),
            "mean_ms": mean_ms,
            "module_time_share_pct": mean_ms / summed_module_mean_ms * 100,
        }

    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "precision": "fused FP16",
            "path": "one-to-one-only",
            "batch": 1,
            "imgsz": args.imgsz,
        },
        "checkpoint": str(args.checkpoint),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters_fused": sum(parameter.numel() for parameter in model.parameters()),
        "total": {
            "mean_ms": statistics.fmean(total_samples),
            "p50_ms": percentile(total_samples, 0.5),
            "p90_ms": percentile(total_samples, 0.9),
        },
        "summed_module_mean_ms": summed_module_mean_ms,
        "stages": stages,
        "modules": sorted(rows, key=lambda row: row["mean_ms"], reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"total": payload["total"], "stages": stages, "top_modules": payload["modules"][:10]}, indent=2))


if __name__ == "__main__":
    main()
