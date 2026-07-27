#!/usr/bin/env python3
"""Fine-tune Vehicle Pose with BN-scale sparsity before physical channel pruning."""

import argparse
import os
from pathlib import Path

from ultralytics.models.yolov10.sparsity import (
    SparsityConfig,
    YOLOv10PoseSparsityModel,
    summarize_bn_scales,
)
from ultralytics.models.yolov10.train import YOLOv10PoseTrainer
from ultralytics.utils import RANK, yaml_load, yaml_save


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_experiment_config(path):
    settings = yaml_load(path)
    required = {"student_weights", "data", "imgsz", "sparsity_weight"}
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"missing sparsity settings: {', '.join(missing)}")
    if float(settings["sparsity_weight"]) <= 0:
        raise ValueError("sparsity_weight must be positive")
    return settings


class YOLOv10PoseSparsityTrainer(YOLOv10PoseTrainer):
    """Build the opt-in sparse model while preserving the standard pose trainer."""

    def __init__(self, sparsity_config, *args, **kwargs):
        self.sparsity_config = sparsity_config
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOv10PoseSparsityModel(
            cfg,
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model.configure_sparsity(self.sparsity_config)


def setup_sparsity_model(trainer):
    """Materialize a checkpoint path before reading configured BN layers."""
    if not hasattr(trainer.model, "sparsity_bn_layers"):
        trainer.setup_model()
    if not hasattr(trainer.model, "sparsity_bn_layers"):
        raise TypeError("trainer did not build a sparsity-configured model")
    return trainer.model.sparsity_bn_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_experiment_config(args.config)
    os.environ.setdefault("WANDB_MODE", str(settings.pop("wandb_mode", "disabled")))
    for key in ("epochs", "batch", "name", "device"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    student_weights = resolve_project_path(settings.pop("student_weights"))
    settings["model"] = str(student_weights)
    settings["data"] = str(resolve_project_path(settings["data"]))
    settings["project"] = str(resolve_project_path(settings.get("project", "vehicle_pose_runs_sparsity")))
    config = SparsityConfig(
        weight=float(settings.pop("sparsity_weight")),
        model_indices=tuple(int(index) for index in settings.pop("sparsity_model_indices", ())),
        include_head_hidden=bool(settings.pop("sparsity_include_head_hidden", True)),
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
    if not student_weights.is_file():
        raise FileNotFoundError(f"student initialization checkpoint not found: {student_weights}")

    trainer = YOLOv10PoseSparsityTrainer(sparsity_config=config, overrides=settings)
    sparsity_bn_layers = setup_sparsity_model(trainer)
    yaml_save(
        trainer.save_dir / "sparsity.yaml",
        {
            "student_initialization": str(student_weights),
            "sparsity_weight": config.weight,
            "sparsity_model_indices": list(config.model_indices),
            "sparsity_include_head_hidden": config.include_head_hidden,
            "initial_bn_scales": summarize_bn_scales(sparsity_bn_layers),
        },
    )
    trainer.train()
    yaml_save(
        trainer.save_dir / "final_bn_scales.yaml",
        summarize_bn_scales(trainer.model.sparsity_bn_layers),
    )


if __name__ == "__main__":
    main()
