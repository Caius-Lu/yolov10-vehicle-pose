#!/usr/bin/env python3
"""Profile the fused FP16 one-to-one deployment path and export operator shares."""

import argparse
import json
from pathlib import Path

import torch

from ultralytics.nn.tasks import attempt_load_one_weight


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_deploy_model(checkpoint, device):
    """Load a whole-model checkpoint so structurally changed graphs are preserved."""

    model, _ = attempt_load_one_weight(checkpoint, device=device, fuse=False)
    model = model.to(device).eval()
    model.fuse()
    model.half()
    model.model[-1].one2one_only = True
    return model


def event_value(event, name):
    value = getattr(event, name, 0.0)
    return float(value() if callable(value) else value)


def first_event_value(event, *names):
    """Support both CUDA-specific and generic device-time profiler fields."""

    for name in names:
        value = event_value(event, name)
        if value:
            return value
    return 0.0


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("deploy FP16 profiling requires a CUDA device")
    model = build_deploy_model(args.checkpoint, device)
    image = torch.rand(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
    torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(args.iterations):
                model(image)
    torch.cuda.synchronize(device)

    events = profiler.key_averages(group_by_input_shape=False)
    rows = []
    for event in events:
        self_cuda_us = first_event_value(event, "self_cuda_time_total", "self_device_time_total")
        cuda_total_us = first_event_value(event, "cuda_time_total", "device_time_total")
        if self_cuda_us <= 0 and cuda_total_us <= 0:
            continue
        rows.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "self_cuda_us": self_cuda_us,
                "cuda_total_us": cuda_total_us,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    total_self_cuda_us = sum(row["self_cuda_us"] for row in rows)
    if not rows or total_self_cuda_us <= 0:
        raise RuntimeError(
            "PyTorch profiler returned no CUDA/device operator time on this build. "
            "Use benchmarks/profile_module_latency.py for CUDA Event stage shares "
            "and keep the uninstrumented deployment benchmark for absolute latency."
        )
    for row in rows:
        row["self_cuda_share_pct"] = (
            row["self_cuda_us"] / total_self_cuda_us * 100 if total_self_cuda_us else 0.0
        )

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
        "total_self_cuda_us": total_self_cuda_us,
        "top_operators": rows[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload["environment"], "top_operators": rows[:12]}, indent=2))


if __name__ == "__main__":
    main()
