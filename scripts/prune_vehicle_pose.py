#!/usr/bin/env python3
"""Create a physically channel-pruned YOLOv10 Vehicle Pose checkpoint.

This stage needs no dataset and never overwrites the baseline checkpoint. It
uses one dummy BCHW tensor to build Torch-Pruning's dependency graph, protects
all final detection/pose outputs, verifies a real parameter reduction, runs a
single forward smoke, and saves the changed structure as a whole-model .pt.
"""

import argparse
import json
from pathlib import Path

import torch

from ultralytics.models.yolov10.prune_distill import (
    PruningConfig,
    build_pruned_checkpoint,
    prune_one2many_deployment_branches,
    structurally_prune_model,
)
from ultralytics.nn.tasks import YOLOv10PoseModel, attempt_load_one_weight


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_output_paths(checkpoint, output, report):
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    report = Path(report).resolve()
    if checkpoint == output:
        raise ValueError("refusing to overwrite the baseline checkpoint")
    if output == report:
        raise ValueError("checkpoint output and JSON report must be different paths")
    if output.exists():
        raise FileExistsError(f"output checkpoint already exists: {output}")
    if report.exists():
        raise FileExistsError(f"output report already exists: {report}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/baseline_weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pruning/yolov10n_pose_pruned_init.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/pruning/structure_report.json")
    parser.add_argument(
        "--graph-imgsz",
        "--imgsz",
        dest="graph_imgsz",
        type=int,
        default=64,
        help="small multiple-of-32 input used only to trace the fully convolutional dependency graph",
    )
    parser.add_argument("--pruning-ratio", type=float, default=0.15)
    parser.add_argument(
        "--head-pruning-ratio",
        type=float,
        default=None,
        help="independent hidden Detect/Pose head ratio; use 0 to protect all head hidden roots",
    )
    parser.add_argument("--iterative-steps", type=int, default=1)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--global-pruning", action="store_true")
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="prune active one-to-one head roots, then remove unused one-to-many deployment towers",
    )
    parser.add_argument(
        "--model-indices",
        default="",
        help="comma-separated Backbone/Neck model indices allowed to start pruning groups",
    )
    parser.add_argument("--measure-macs", action="store_true", help="also run the slower optional MAC counter")
    parser.add_argument("--importance", choices=("magnitude", "bn_scale"), default="magnitude")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or an explicit CUDA device")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint)
    output = resolve_project_path(args.output)
    report_path = resolve_project_path(args.report)
    validate_output_paths(checkpoint, output, report_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {checkpoint}")
    if args.graph_imgsz <= 0 or args.graph_imgsz % 32:
        raise ValueError("graph_imgsz must be a positive multiple of 32")

    device = torch.device(args.device)
    model, _ = attempt_load_one_weight(checkpoint, device=device, fuse=False)
    if not isinstance(model, YOLOv10PoseModel):
        raise TypeError(f"expected YOLOv10PoseModel checkpoint, got {type(model).__name__}")
    model.float().eval()
    model.model[-1].export = False
    deployment_report = None
    model.model[-1].one2one_only = False
    example = torch.randn(1, 3, args.graph_imgsz, args.graph_imgsz, device=device)
    config = PruningConfig(
        pruning_ratio=args.pruning_ratio,
        head_pruning_ratio=args.head_pruning_ratio,
        iterative_steps=args.iterative_steps,
        round_to=args.round_to,
        global_pruning=args.global_pruning,
        measure_macs=args.measure_macs,
        model_indices=tuple(int(value) for value in args.model_indices.split(",") if value.strip()),
        importance=args.importance,
        one2one_head_only=args.deployment_only,
    )
    model, report = structurally_prune_model(model, example, config)

    model.eval()
    if args.deployment_only:
        structural_parameters_after = report["parameters_after"]
        model, deployment_report = prune_one2many_deployment_branches(model)
        report.update(
            structural_parameters_after=structural_parameters_after,
            parameters_after=deployment_report["parameters_after"],
            parameters_removed=report["parameters_before"] - deployment_report["parameters_after"],
            parameter_reduction_ratio=(
                (report["parameters_before"] - deployment_report["parameters_after"])
                / report["parameters_before"]
            ),
        )
    model.model[-1].one2one_only = True
    with torch.inference_mode():
        prediction = model(example)
    one2one = prediction["one2one"][0]
    report.update(
        source_checkpoint=str(checkpoint),
        output_checkpoint=str(output),
        example_shape=list(example.shape),
        smoke_output_shape=list(one2one.shape),
        smoke_finite=bool(torch.isfinite(one2one).all().item()),
        deployment_only=args.deployment_only,
        deployment_branch_pruning=deployment_report,
    )
    if not report["smoke_finite"]:
        raise RuntimeError("pruned-model forward smoke produced non-finite values")
    model.model[-1].one2one_only = args.deployment_only
    model.cpu().float()

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_pruned_checkpoint(model, checkpoint, report), output)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
