#!/usr/bin/env python3
"""Measure batch-1 image-load through pose Results end-to-end latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from benchmarks.benchmark_torch import describe
except ModuleNotFoundError:  # direct `python benchmarks/benchmark_end_to_end.py`
    from benchmark_torch import describe
from ultralytics import YOLOv10


def load_sources(source_list: Path) -> list[Path]:
    """Resolve a YOLO dataset list relative to the list file's parent."""
    source_list = Path(source_list)
    sources = []
    for value in source_list.read_text(encoding="utf-8").splitlines():
        if not value.strip():
            continue
        source = Path(value.strip())
        if not source.is_absolute():
            source = source_list.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing benchmark source: {source}")
        sources.append(source)
    if not sources:
        raise ValueError(f"no benchmark sources in {source_list}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-list", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--rounds", default=3, type=int)
    parser.add_argument("--output", default=Path("reports/benchmark/07_end_to_end_results.json"), type=Path)
    args = parser.parse_args()

    sources = load_sources(args.source_list)
    wrapper = YOLOv10(args.checkpoint, task="pose")
    wrapper.model.model[-1].one2one_only = True
    predict_args = {
        "imgsz": args.imgsz,
        "device": args.device,
        "verbose": False,
        "save": False,
        "stream": False,
    }
    for index in range(args.warmup):
        wrapper.predict(source=str(sources[index % len(sources)]), **predict_args)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()

    rounds = []
    for round_index in range(args.rounds):
        wall_samples = []
        component_samples = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
        for index in range(args.iterations):
            start = time.perf_counter_ns()
            results = wrapper.predict(source=str(sources[index % len(sources)]), **predict_args)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            wall_samples.append((time.perf_counter_ns() - start) / 1_000_000)
            speed = results[0].speed
            component_samples["preprocess_ms"].append(float(speed["preprocess"]))
            component_samples["inference_ms"].append(float(speed["inference"]))
            component_samples["postprocess_ms"].append(float(speed["postprocess"]))
        record = describe(wall_samples)
        record.update(
            round=round_index + 1,
            mode="end_to_end",
            device=args.device,
            imgsz=args.imgsz,
            batch=1,
            sources=len(sources),
            component_means_ms={key: sum(values) / len(values) for key, values in component_samples.items()},
        )
        rounds.append(record)

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_list": str(args.source_list.resolve()),
        "definition": "wall time from JPEG path input through decoded Results; batch=1",
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
