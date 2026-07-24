from pathlib import Path

import pytest

from benchmarks.pretrain_architecture_gate import GateThresholds, evaluate_candidate
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"
FAST_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead.yaml"
FAST_BOX_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead-boxd1.yaml"


def _parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_fast_pose_head_is_explicit_and_structurally_smaller():
    baseline = YOLOv10PoseModel(BASELINE_CFG, verbose=False)
    candidate = YOLOv10PoseModel(FAST_CFG, verbose=False)
    head = candidate.model[-1]

    assert head.pose_head_channels == 64
    assert head.pose_head_depth == 1
    assert _parameters(candidate) < _parameters(baseline) * 0.85


def test_fast_box_and_pose_head_removes_one_spatial_layer_per_tower():
    baseline = YOLOv10PoseModel(BASELINE_CFG, verbose=False)
    candidate = YOLOv10PoseModel(FAST_BOX_CFG, verbose=False)
    head = candidate.model[-1]

    assert head.pose_head_channels == 64
    assert head.pose_head_depth == 1
    assert head.box_head_depth == 1
    assert all(len(tower) == 2 for tower in [*head.cv2, *head.one2one_cv2])
    assert _parameters(candidate) < _parameters(baseline) * 0.80


def test_gate_requires_both_parameter_and_latency_reduction():
    thresholds = GateThresholds(
        min_parameter_reduction_pct=10.0,
        min_gflops_reduction_pct=8.0,
        min_latency_reduction_pct=5.0,
    )

    passed = evaluate_candidate(
        baseline_parameters=4_000_000,
        candidate_parameters=3_400_000,
        baseline_gflops=10.0,
        candidate_gflops=8.5,
        baseline_latency_ms=6.0,
        candidate_latency_ms=5.4,
        thresholds=thresholds,
    )
    slow = evaluate_candidate(
        baseline_parameters=4_000_000,
        candidate_parameters=3_400_000,
        baseline_gflops=10.0,
        candidate_gflops=8.5,
        baseline_latency_ms=6.0,
        candidate_latency_ms=6.1,
        thresholds=thresholds,
    )
    too_large = evaluate_candidate(
        baseline_parameters=4_000_000,
        candidate_parameters=3_800_000,
        baseline_gflops=10.0,
        candidate_gflops=8.5,
        baseline_latency_ms=6.0,
        candidate_latency_ms=5.4,
        thresholds=thresholds,
    )
    too_many_flops = evaluate_candidate(
        baseline_parameters=4_000_000,
        candidate_parameters=3_400_000,
        baseline_gflops=10.0,
        candidate_gflops=9.5,
        baseline_latency_ms=6.0,
        candidate_latency_ms=5.4,
        thresholds=thresholds,
    )

    assert passed["train_allowed"] is True
    assert slow["train_allowed"] is False
    assert "latency" in slow["failed_gates"]
    assert too_large["train_allowed"] is False
    assert "parameters" in too_large["failed_gates"]
    assert too_many_flops["train_allowed"] is False
    assert "gflops" in too_many_flops["failed_gates"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_parameter_reduction_pct": -1.0}, "parameter"),
        ({"min_gflops_reduction_pct": -1.0}, "gflops"),
        ({"min_latency_reduction_pct": -1.0}, "latency"),
    ],
)
def test_gate_thresholds_reject_negative_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GateThresholds(**kwargs)
