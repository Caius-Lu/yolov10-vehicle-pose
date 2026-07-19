#!/usr/bin/env python3
"""Fine-tune a structurally pruned Vehicle Pose student with output KD."""

import argparse
from pathlib import Path

from ultralytics.models.yolov10.prune_distill import (
    OutputDistillationConfig,
    upgrade_for_pruning_distillation,
)
from ultralytics.models.yolov10.train import YOLOv10PoseTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import LOGGER, yaml_load, yaml_save


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_experiment_config(path):
    """Load and validate the independent pruning-distillation settings."""

    settings = yaml_load(path)
    required = {"pruned_weights", "teacher_weights", "data", "imgsz", "teacher_imgsz", "distill_weight"}
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"missing pruning-distillation settings: {', '.join(missing)}")
    if int(settings["teacher_imgsz"]) != int(settings["imgsz"]):
        raise ValueError("pruned student and teacher must use the same resolution for raw-output distillation")
    return settings


class YOLOv10PosePrunedDistillationTrainer(YOLOv10PoseTrainer):
    """Load a whole-model pruned checkpoint without rebuilding its changed shapes."""

    def __init__(self, pruned_weights, teacher_weights, distillation_config, *args, **kwargs):
        self.pruned_weights = Path(pruned_weights)
        self.teacher_weights = Path(teacher_weights)
        self.distillation_config = distillation_config
        super().__init__(*args, **kwargs)

    def setup_model(self):
        student, checkpoint = attempt_load_one_weight(self.pruned_weights, fuse=False)
        teacher, _ = attempt_load_one_weight(self.teacher_weights, fuse=False)
        self.model = upgrade_for_pruning_distillation(student, teacher, self.distillation_config)
        LOGGER.info(
            f"Pruned output distillation enabled: student={self.pruned_weights}, "
            f"teacher={self.teacher_weights}, weight={self.distillation_config.weight}, "
            f"temperature={self.distillation_config.temperature}"
        )
        return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pose_prune_distill.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_experiment_config(args.config)
    for key in ("epochs", "batch", "name", "fraction", "device"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value

    pruned_weights = resolve_project_path(settings.pop("pruned_weights"))
    teacher_weights = resolve_project_path(settings.pop("teacher_weights"))
    settings["model"] = str(pruned_weights)
    settings["data"] = str(resolve_project_path(settings["data"]))
    settings["project"] = str(resolve_project_path(settings.get("project", "vehicle_pose_runs_pruned")))
    distillation_config = OutputDistillationConfig(
        weight=float(settings.pop("distill_weight")),
        teacher_imgsz=int(settings.pop("teacher_imgsz")),
        temperature=float(settings.pop("temperature", 2.0)),
        detection_weight=float(settings.pop("detection_distill_weight", 1.0)),
        pose_weight=float(settings.pop("pose_distill_weight", 1.0)),
    )
    settings.update(
        task="pose",
        deterministic=True,
        amp=True,
        fliplr=0.0,
        close_mosaic=0,
        exist_ok=False,
        plots=True,
        val=True,
    )
    for label, path in (
        ("pruned student checkpoint", pruned_weights),
        ("teacher checkpoint", teacher_weights),
        ("dataset configuration", Path(settings["data"])),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    trainer = YOLOv10PosePrunedDistillationTrainer(
        pruned_weights=pruned_weights,
        teacher_weights=teacher_weights,
        distillation_config=distillation_config,
        overrides=settings,
    )
    yaml_save(
        trainer.save_dir / "pruning_distillation.yaml",
        {
            "pruned_student": str(pruned_weights),
            "teacher": str(teacher_weights),
            "imgsz": settings["imgsz"],
            "distill_weight": distillation_config.weight,
            "temperature": distillation_config.temperature,
            "detection_distill_weight": distillation_config.detection_weight,
            "pose_distill_weight": distillation_config.pose_weight,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
