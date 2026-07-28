#!/usr/bin/env python3
"""Measure fused-FP16 deployment memory for one checkpoint in an isolated process."""

import argparse
import json
from pathlib import Path

import torch

try:
    from benchmarks.benchmark_deploy_fp16 import build_model
except ModuleNotFoundError:
    from benchmark_deploy_fp16 import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parameter_storage_bytes(model):
    """Count unique parameter and buffer tensor storage."""
    tensors = [*model.parameters(), *model.buffers()]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def validate_output_path(output):
    """Protect previously recorded memory evidence from silent replacement."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"memory output already exists: {output}")


def main():
    args = parse_args()
    validate_output_path(args.output)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("deployment memory measurement requires CUDA")
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    model = build_model(args.checkpoint, device, half=True, fuse=True)
    model.model[-1].one2one_only = True
    allocated_after_model = torch.cuda.memory_allocated(device)
    image = torch.rand(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16)
    allocated_with_input = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(image)
        torch.cuda.synchronize(device)
        peak_during_inference = torch.cuda.max_memory_allocated(device)
    dense = output["one2one"][0]
    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "precision": "fused FP16",
            "batch": 1,
            "imgsz": args.imgsz,
            "path": "one-to-one-only",
        },
        "checkpoint": str(args.checkpoint),
        "parameters_fused": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_and_buffer_storage_bytes": parameter_storage_bytes(model),
        "cuda_allocated_baseline_bytes": baseline_allocated,
        "cuda_allocated_after_model_bytes": allocated_after_model,
        "cuda_model_delta_bytes": allocated_after_model - baseline_allocated,
        "cuda_allocated_with_input_bytes": allocated_with_input,
        "cuda_peak_during_inference_bytes": peak_during_inference,
        "dense_shape": list(dense.shape),
        "dense_finite": bool(torch.isfinite(dense).all()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
