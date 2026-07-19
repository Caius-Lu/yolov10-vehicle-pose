#!/usr/bin/env python3
"""Fine-tune a 512px YOLOv10 pose student from a frozen 640px teacher."""

import argparse
from pathlib import Path

from ultralytics.models.yolov10.distill import (
    DistillationConfig,
    YOLOv10PoseDistillationModel,
)
from ultralytics.models.yolov10.train import YOLOv10PoseTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import LOGGER, RANK, yaml_load, yaml_save


ROOT = Path(__file__).resolve().parents[1]


def load_experiment_config(path):
    """Load and validate the independent distillation experiment settings."""

    settings = yaml_load(path)
    required = {"student_weights", "teacher_weights", "data", "imgsz", "teacher_imgsz", "distill_weight"}
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"missing distillation settings: {', '.join(missing)}")
    if int(settings["teacher_imgsz"]) < int(settings["imgsz"]):
        raise ValueError("teacher_imgsz must be greater than or equal to student imgsz")
    return settings


def resolve_project_path(value):
    """Resolve a configured input path relative to the repository root."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class YOLOv10PoseDistillationTrainer(YOLOv10PoseTrainer):
    """Opt-in trainer that builds a distilled student and frozen teacher."""

    def __init__(self, teacher_weights, distillation_config, *args, **kwargs):
        self.teacher_weights = Path(teacher_weights)
        self.distillation_config = distillation_config
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        student = YOLOv10PoseDistillationModel(
            cfg,
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            student.load(weights)
        teacher, _ = attempt_load_one_weight(self.teacher_weights, device=self.device)
        student.configure_distillation(teacher, self.distillation_config)
        LOGGER.info(
            f"Distillation enabled: teacher={self.teacher_weights}, "
            f"teacher_imgsz={self.distillation_config.teacher_imgsz}, weight={self.distillation_config.weight}"
        )
        return student


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pose_distill_512.yaml")
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

    teacher_weights = resolve_project_path(settings.pop("teacher_weights"))
    settings["model"] = str(resolve_project_path(settings.pop("student_weights")))
    settings["data"] = str(resolve_project_path(settings["data"]))
    settings["project"] = str(resolve_project_path(settings.get("project", "vehicle_pose_runs_distill")))
    distillation_config = DistillationConfig(
        weight=float(settings.pop("distill_weight")), teacher_imgsz=int(settings.pop("teacher_imgsz"))
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

    if not teacher_weights.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_weights}")
    if not Path(settings["model"]).is_file():
        raise FileNotFoundError(f"student initialization checkpoint not found: {settings['model']}")

    trainer = YOLOv10PoseDistillationTrainer(
        teacher_weights=teacher_weights,
        distillation_config=distillation_config,
        overrides=settings,
    )
    yaml_save(
        trainer.save_dir / "distillation.yaml",
        {
            "teacher_weights": str(teacher_weights),
            "student_initialization": settings["model"],
            "student_imgsz": settings["imgsz"],
            "teacher_imgsz": distillation_config.teacher_imgsz,
            "distill_weight": distillation_config.weight,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
