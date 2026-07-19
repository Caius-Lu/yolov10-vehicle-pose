#!/usr/bin/env python3
"""Capture a PyTorch profiler trace for the inference-only vehicle-pose path."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOv10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/yolov10n.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark/profiler"))
    args = parser.parse_args()
    wrapper = YOLOv10("ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml", task="pose")
    if Path(args.checkpoint).is_file():
        wrapper.load(args.checkpoint)
    model = wrapper.model.to(args.device).eval()
    model.model[-1].one2one_only = True
    tensor = torch.rand(1, 3, args.imgsz, args.imgsz, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(args.output)),
    ) as profiler:
        with torch.inference_mode():
            for _ in range(5):
                model(tensor)
                profiler.step()
    sort_key = "self_cuda_time_total" if str(args.device).startswith("cuda") else "self_cpu_time_total"
    table = profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=30)
    (args.output / "top30.txt").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

