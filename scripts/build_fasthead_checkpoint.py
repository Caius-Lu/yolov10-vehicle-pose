#!/usr/bin/env python3
"""Build a speed-gated architecture checkpoint from the converged 200-epoch baseline."""

import argparse
import json
from pathlib import Path

import torch

from ultralytics.models.yolov10.fasthead import transplant_compatible_weights
from ultralytics.models.yolov10.prune_distill import build_pruned_checkpoint
from ultralytics.nn.tasks import YOLOv10PoseModel, attempt_load_one_weight


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead-boxd1.yaml",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fasthead/fasthead64_boxd1_init.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/fasthead/transfer_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-imgsz", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint, model_config = resolve(args.checkpoint), resolve(args.model)
    output, report_path = resolve(args.output), resolve(args.report)
    if output.resolve() == checkpoint.resolve():
        raise ValueError("refusing to overwrite the baseline checkpoint")
    teacher, _ = attempt_load_one_weight(checkpoint, device=torch.device(args.device), fuse=False)
    student = YOLOv10PoseModel(model_config, verbose=False).to(args.device).float()
    report = transplant_compatible_weights(student, teacher.float())

    student.eval()
    student.model[-1].one2one_only = True
    example = torch.randn(1, 3, args.smoke_imgsz, args.smoke_imgsz, device=args.device)
    with torch.inference_mode():
        prediction = student(example)["one2one"][0]
    report.update(
        source_checkpoint=str(checkpoint),
        model_config=str(model_config),
        output_checkpoint=str(output),
        smoke_output_shape=list(prediction.shape),
        smoke_finite=bool(torch.isfinite(prediction).all()),
    )
    if not report["smoke_finite"]:
        raise RuntimeError("transferred architecture candidate produced non-finite outputs")
    student.model[-1].one2one_only = False
    student.cpu().float()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_pruned_checkpoint(student, checkpoint, report), output)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
