#!/usr/bin/env python3
"""Benchmark the accuracy-preserving fused FP16 one-to-one deployment path."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from ultralytics.nn.tasks import attempt_load_one_weight

try:
    from benchmarks.benchmark_torch import describe, measure
except ModuleNotFoundError:  # Direct execution: python benchmarks/benchmark_deploy_fp16.py
    from benchmark_torch import describe, measure


def build_model(checkpoint, device, half=False, fuse=False):
    # Structural pruning changes module widths, so rebuilding the YAML graph and
    # intersecting a state_dict would silently lose the pruned architecture.
    model, _ = attempt_load_one_weight(checkpoint, device=device, fuse=False)
    model = model.to(device).eval()
    if fuse:
        model.fuse()
    if half:
        model.half()
    return model


def summarize_paired_latency(baseline_rounds_ms, candidate_rounds_ms):
    """Summarize equal-count paired fused-FP16 rounds."""
    if not baseline_rounds_ms or len(baseline_rounds_ms) != len(candidate_rounds_ms):
        raise ValueError("baseline and candidate must have the same non-zero number of rounds")
    if any(value <= 0 for value in [*baseline_rounds_ms, *candidate_rounds_ms]):
        raise ValueError("paired latency measurements must be positive")
    baseline = statistics.fmean(baseline_rounds_ms)
    candidate = statistics.fmean(candidate_rounds_ms)
    return {
        "baseline_fused_fp16_ms": baseline,
        "candidate_fused_fp16_ms": candidate,
        "latency_reduction_pct": (baseline - candidate) / baseline * 100,
        "speedup": baseline / candidate,
    }


def benchmark_paired_checkpoints(args, device):
    """Compare two whole-model checkpoints in alternating fused-FP16 rounds."""
    models = {
        "baseline": build_model(args.checkpoint, device, half=True, fuse=True),
        "candidate": build_model(args.candidate_checkpoint, device, half=True, fuse=True),
    }
    image = torch.rand(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16)
    model_stats = {}
    with torch.inference_mode():
        for name, model in models.items():
            model.model[-1].one2one_only = True
            dense = model(image)["one2one"][0]
            if not torch.isfinite(dense).all():
                raise RuntimeError(f"{name} checkpoint produced non-finite fused-FP16 output")
            model_stats[name] = {
                "checkpoint": str(args.checkpoint if name == "baseline" else args.candidate_checkpoint),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "dense_shape": list(dense.shape),
                "rounds": [],
            }
            measure(model, image, args.warmup, 1)
    if model_stats["baseline"]["dense_shape"] != model_stats["candidate"]["dense_shape"]:
        raise RuntimeError("baseline and candidate dense output shapes differ")

    for round_index in range(args.rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for name in order:
            result = describe(measure(models[name], image, args.warmup, args.iterations))
            result.update(round=round_index + 1, order=list(order))
            model_stats[name]["rounds"].append(result)
    summary = summarize_paired_latency(
        [record["mean_ms"] for record in model_stats["baseline"]["rounds"]],
        [record["mean_ms"] for record in model_stats["candidate"]["rounds"]],
    )
    return {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "precision": "fused FP16",
            "batch": 1,
            "imgsz": args.imgsz,
            "path": "one-to-one-only",
        },
        "summary": summary,
        "models": model_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", help="enable paired fused-FP16 checkpoint comparison")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.candidate_checkpoint:
        payload = benchmark_paired_checkpoints(args, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2))
        return
    fp32 = build_model(args.checkpoint, device)
    fp16 = build_model(args.checkpoint, device, half=True, fuse=True)
    fp32.model[-1].one2one_only = True
    fp16.model[-1].one2one_only = True
    image_fp32 = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)
    image_fp16 = image_fp32.half()

    with torch.inference_mode():
        reference = fp32(image_fp32)["one2one"][0]
        candidate = fp16(image_fp16)["one2one"][0].float()
    difference = (reference - candidate).abs()
    variants = {
        "fp32_one2one": (fp32, image_fp32),
        "fused_fp16_one2one": (fp16, image_fp16),
    }
    for model, image in variants.values():
        measure(model, image, args.warmup, 1)
    records = []
    for round_index in range(args.rounds):
        order = ("fp32_one2one", "fused_fp16_one2one") if round_index % 2 == 0 else tuple(reversed(variants))
        for name in order:
            model, image = variants[name]
            result = describe(measure(model, image, args.warmup, args.iterations))
            result.update(mode=name, round=round_index + 1, imgsz=args.imgsz, order=list(order))
            records.append(result)

    means = {
        name: statistics.fmean(r["mean_ms"] for r in records if r["mode"] == name)
        for name in ("fp32_one2one", "fused_fp16_one2one")
    }
    payload = {
        "environment": {"device": torch.cuda.get_device_name(device), "torch": torch.__version__},
        "correctness": {"dense_max_abs_error": difference.max().item(), "dense_mean_abs_error": difference.mean().item()},
        "summary": {
            **means,
            "latency_reduction_pct": (means["fp32_one2one"] - means["fused_fp16_one2one"])
            / means["fp32_one2one"]
            * 100,
            "speedup": means["fp32_one2one"] / means["fused_fp16_one2one"],
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"correctness": payload["correctness"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
