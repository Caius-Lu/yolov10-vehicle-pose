#!/usr/bin/env python3
"""Reject smaller YOLOv10-Pose candidates that do not produce real deployment speedups."""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.benchmark_torch import describe, measure
from ultralytics.nn.tasks import YOLOv10PoseModel
from ultralytics.utils.torch_utils import get_flops


@dataclass(frozen=True)
class GateThresholds:
    """Minimum improvements required before any recovery training is allowed."""

    min_parameter_reduction_pct: float = 10.0
    min_gflops_reduction_pct: float = 8.0
    min_latency_reduction_pct: float = 5.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _reduction_pct(baseline, candidate):
    if baseline <= 0:
        raise ValueError("baseline values must be positive")
    return (baseline - candidate) / baseline * 100.0


def evaluate_candidate(
    *,
    baseline_parameters,
    candidate_parameters,
    baseline_gflops,
    candidate_gflops,
    baseline_latency_ms,
    candidate_latency_ms,
    thresholds,
):
    """Apply independent size, compute, and measured-latency gates."""
    improvements = {
        "parameters": _reduction_pct(baseline_parameters, candidate_parameters),
        "gflops": _reduction_pct(baseline_gflops, candidate_gflops),
        "latency": _reduction_pct(baseline_latency_ms, candidate_latency_ms),
    }
    required = {
        "parameters": thresholds.min_parameter_reduction_pct,
        "gflops": thresholds.min_gflops_reduction_pct,
        "latency": thresholds.min_latency_reduction_pct,
    }
    failed = [name for name, minimum in required.items() if improvements[name] < minimum]
    return {
        "train_allowed": not failed,
        "failed_gates": failed,
        "improvement_pct": improvements,
        "required_improvement_pct": required,
    }


def build_deploy_model(config, device, imgsz):
    """Build the one-to-one deployment graph and report its dense model statistics."""
    model = YOLOv10PoseModel(config, verbose=False).eval()
    model.model[-1].one2one_only = True
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = get_flops(model, imgsz=imgsz)
    if gflops <= 0:
        raise RuntimeError("GFLOPs profiling failed; install thop before running the architecture gate")
    model = model.to(device)
    model.fuse()
    model.half()
    return model, {"parameters": parameters, "gflops": gflops}


def benchmark_configs(configs, device, imgsz, warmup, iterations, rounds):
    """Benchmark configs in alternating order to reduce thermal and ordering bias."""
    models = {}
    stats = {}
    image = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.float16)
    for name, config in configs.items():
        model, model_stats = build_deploy_model(config, device, imgsz)
        models[name] = model
        stats[name] = {**model_stats, "config": str(config), "rounds": []}
        measure(model, image, warmup, 1)

    names = list(configs)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            summary = describe(measure(models[name], image, warmup, iterations))
            summary.update(round=round_index + 1, order=order)
            stats[name]["rounds"].append(summary)

    for item in stats.values():
        item["mean_latency_ms"] = statistics.fmean(record["mean_ms"] for record in item["rounds"])
    return stats


def parse_named_config(value):
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH") from error
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_named_config, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-parameter-reduction-pct", type=float, default=10.0)
    parser.add_argument("--min-gflops-reduction-pct", type=float, default=8.0)
    parser.add_argument("--min-latency-reduction-pct", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = GateThresholds(
        min_parameter_reduction_pct=args.min_parameter_reduction_pct,
        min_gflops_reduction_pct=args.min_gflops_reduction_pct,
        min_latency_reduction_pct=args.min_latency_reduction_pct,
    )
    configs = {"baseline": args.baseline, **dict(args.candidate)}
    stats = benchmark_configs(
        configs,
        torch.device(args.device),
        args.imgsz,
        args.warmup,
        args.iterations,
        args.rounds,
    )
    baseline = stats["baseline"]
    decisions = {}
    for name, candidate in stats.items():
        if name == "baseline":
            continue
        decisions[name] = evaluate_candidate(
            baseline_parameters=baseline["parameters"],
            candidate_parameters=candidate["parameters"],
            baseline_gflops=baseline["gflops"],
            candidate_gflops=candidate["gflops"],
            baseline_latency_ms=baseline["mean_latency_ms"],
            candidate_latency_ms=candidate["mean_latency_ms"],
            thresholds=thresholds,
        )
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(torch.device(args.device)),
            "torch": torch.__version__,
            "precision": "fused FP16",
            "batch": 1,
            "imgsz": args.imgsz,
            "path": "one-to-one-only",
        },
        "thresholds": asdict(thresholds),
        "models": stats,
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"models": stats, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
