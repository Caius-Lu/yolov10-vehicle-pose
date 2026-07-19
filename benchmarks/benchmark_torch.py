#!/usr/bin/env python3
"""Strict synthetic PyTorch timing for original and one-to-one-only pose paths."""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

from ultralytics import YOLOv10


def describe(samples_ms):
    ordered = sorted(samples_ms)

    def percentile(value):
        position = (len(ordered) - 1) * value
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    mean = statistics.fmean(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "std_ms": statistics.pstdev(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "fps_from_mean": 1000.0 / mean,
    }


def dense_one2one(output):
    return output["one2one"][0]


def measure(model, tensor, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
            samples = []
            for _ in range(iterations):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                model(tensor)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return samples
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            model(tensor)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml")
    parser.add_argument("--checkpoint", default="weights/yolov10n.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    wrapper = YOLOv10(args.model, task="pose")
    if Path(args.checkpoint).is_file():
        wrapper.load(args.checkpoint)
    model = wrapper.model.to(device).eval()
    head = model.model[-1]
    tensor = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device)

    head.one2one_only = False
    with torch.inference_mode():
        original = dense_one2one(model(tensor)).detach()
    head.one2one_only = True
    with torch.inference_mode():
        optimized = dense_one2one(model(tensor)).detach()
    max_error = (original - optimized).abs().max().item()
    if not torch.allclose(original, optimized, atol=1e-6, rtol=1e-5):
        raise RuntimeError(f"one2one-only output mismatch: max_abs_error={max_error}")

    records = []
    for mode, one2one_only in (("torch_original", False), ("torch_one2one_only", True)):
        head.one2one_only = one2one_only
        for round_index in range(args.rounds):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = describe(measure(model, tensor, args.warmup, args.iterations))
            result.update(
                mode=mode,
                round=round_index + 1,
                device=str(device),
                precision="fp32",
                batch=args.batch,
                imgsz=args.imgsz,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            )
            records.append(result)

    args.output.mkdir(parents=True, exist_ok=True)
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": args.checkpoint,
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size if Path(args.checkpoint).is_file() else None,
        "one2one_max_abs_error": max_error,
        "warning": "Accuracy is not established unless checkpoint points to a completed CarFusion training run.",
    }
    (args.output / "07_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (args.output / "07_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (args.output / "07_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"environment": environment, "results": records}, indent=2))


if __name__ == "__main__":
    main()

