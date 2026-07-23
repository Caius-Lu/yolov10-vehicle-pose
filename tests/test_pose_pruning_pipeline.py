"""Configuration and safety tests for the pruning/distillation command-line pipeline."""

from pathlib import Path

import pytest
import yaml

from scripts.prune_vehicle_pose import validate_output_paths
from scripts.train_pose_pruned_distillation import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


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
