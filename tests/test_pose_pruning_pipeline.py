"""Configuration and safety tests for the pruning/distillation command-line pipeline."""

from pathlib import Path

import pytest
import torch
import yaml

from scripts.prune_vehicle_pose import validate_output_paths
from scripts.train_pose_pruned_distillation import (
    apply_cli_overrides,
    apply_trainable_parameter_patterns,
    load_experiment_config,
    parse_args as parse_distillation_args,
    set_frozen_normalization_eval,
    validate_new_run_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trainable_parameter_patterns_freeze_everything_except_matching_branches():
    model = torch.nn.ModuleDict(
        {
            "backbone": torch.nn.Linear(4, 4),
            "box": torch.nn.Linear(4, 4),
            "pose": torch.nn.Linear(4, 4),
        }
    )

    trainable = apply_trainable_parameter_patterns(model, ("box.",))

    assert trainable == ["box.weight", "box.bias"]
    assert all(not parameter.requires_grad for parameter in model["backbone"].parameters())
    assert all(parameter.requires_grad for parameter in model["box"].parameters())
    assert all(not parameter.requires_grad for parameter in model["pose"].parameters())


def test_trainable_parameter_patterns_reject_an_empty_match():
    model = torch.nn.Linear(4, 4)

    with pytest.raises(ValueError, match="did not match"):
        apply_trainable_parameter_patterns(model, ("missing.",))


def test_frozen_normalization_uses_saved_statistics_during_selective_recovery():
    frozen = torch.nn.BatchNorm2d(4)
    trainable = torch.nn.BatchNorm2d(4)
    model = torch.nn.ModuleDict({"frozen": frozen, "trainable": trainable}).train()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)

    frozen_names = set_frozen_normalization_eval(model)

    assert frozen_names == ["frozen"]
    assert frozen.training is False
    assert trainable.training is True


def test_pruning_cli_exposes_independent_head_ratio():
    source = (ROOT / "scripts/prune_vehicle_pose.py").read_text(encoding="utf-8")

    assert "--head-pruning-ratio" in source


def test_pruning_refuses_to_overwrite_the_baseline_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    report = tmp_path / "report.json"

    with pytest.raises(ValueError, match="baseline"):
        validate_output_paths(checkpoint, checkpoint, report)
    with pytest.raises(ValueError, match="different"):
        validate_output_paths(checkpoint, report, report)


def test_pruning_refuses_to_overwrite_existing_outputs(tmp_path):
    checkpoint = tmp_path / "best.pt"
    output = tmp_path / "student.pt"
    report = tmp_path / "report.json"
    output.touch()

    with pytest.raises(FileExistsError, match="output checkpoint"):
        validate_output_paths(checkpoint, output, report)

    output.unlink()
    report.touch()
    with pytest.raises(FileExistsError, match="report"):
        validate_output_paths(checkpoint, output, report)


def test_pruned_distillation_config_requires_same_resolution_and_inputs(tmp_path):
    path = tmp_path / "experiment.yaml"
    settings = {
        "pruned_weights": "pruned.pt",
        "teacher_weights": "best.pt",
        "data": "data.yaml",
        "imgsz": 640,
        "teacher_imgsz": 512,
        "distill_weight": 0.25,
    }
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    with pytest.raises(ValueError, match="same resolution"):
        load_experiment_config(path)

    settings["teacher_imgsz"] = 640
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    assert load_experiment_config(path)["pruned_weights"] == "pruned.pt"

    del settings["data"]
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_experiment_config(path)


def test_distillation_cli_can_reuse_one_config_for_s_or_l_checkpoints():
    args = parse_distillation_args(
        [
            "--config",
            "configs/pose_prune_recovery.yaml",
            "--pruned-weights",
            "outputs/pruning/yolov10l_pose_pruned.pt",
            "--teacher-weights",
            "runs/yolov10l_pose/weights/best.pt",
            "--data",
            "ultralytics/cfg/datasets/CarFusion-pose.yaml",
            "--name",
            "yolov10l_prune10_recovery",
        ]
    )
    settings = apply_cli_overrides(
        {
            "pruned_weights": "old-student.pt",
            "teacher_weights": "old-teacher.pt",
            "data": "old-data.yaml",
            "name": "old-name",
        },
        args,
    )

    assert settings["pruned_weights"] == "outputs/pruning/yolov10l_pose_pruned.pt"
    assert settings["teacher_weights"] == "runs/yolov10l_pose/weights/best.pt"
    assert settings["data"] == "ultralytics/cfg/datasets/CarFusion-pose.yaml"
    assert settings["name"] == "yolov10l_prune10_recovery"


def test_distillation_cli_exposes_true_checkpoint_resume():
    args = parse_distillation_args(
        [
            "--config",
            "configs/pose_prune_recovery.yaml",
            "--teacher-weights",
            "runs/yolov10s_baseline/weights/best.pt",
            "--resume",
            "runs/yolov10s_prune10/weights/last.pt",
        ]
    )
    settings = apply_cli_overrides({"pruned_weights": "init.pt"}, args)

    assert settings["resume"] == "runs/yolov10s_prune10/weights/last.pt"


def test_new_recovery_run_refuses_automatic_name_increment(tmp_path):
    existing = tmp_path / "pruned" / "run"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        validate_new_run_directory(tmp_path / "pruned", "run", resume=False)
    validate_new_run_directory(tmp_path / "pruned", "run", resume=True)
