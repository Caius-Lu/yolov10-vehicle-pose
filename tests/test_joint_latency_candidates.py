"""TDD contracts for joint Backbone/Neck/Head latency candidates."""

from pathlib import Path

import pytest
import torch

from benchmarks.pretrain_architecture_gate import GateThresholds, evaluate_candidate
from ultralytics.models.yolov10.fasthead import transplant_compatible_weights
from ultralytics.nn.modules.block import C1, C2f, C2fCIB, LitePSA, RepC3
from ultralytics.nn.modules.conv import RepConv
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics/cfg/models/v10"
BASELINE = MODEL_DIR / "yolov10n-vehicle-pose.yaml"
CANDIDATES = {
    "e4375_head96": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-e4375-head96.yaml",
        96,
    ),
    "e375_head96": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-e375-head96.yaml",
        96,
    ),
    "e375_head88": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-e375-head88.yaml",
        88,
    ),
    "e375_head80": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-e375-head80.yaml",
        80,
    ),
}
CHANNEL_SLIM_CANDIDATES = {
    "w875_head96": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-w875-head96.yaml",
        (56, 112, 224),
    ),
    "w75_head96": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-w75-head96.yaml",
        (48, 96, 192),
    ),
}
KERNEL_SLIM_CANDIDATES = {
    "repc3": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-repc3-head96.yaml",
        RepC3,
    ),
    "c1neck": (
        MODEL_DIR / "yolov10n-vehicle-pose-joint-repc3-c1neck-head96.yaml",
        C1,
    ),
}


def build(path):
    return YOLOv10PoseModel(path, verbose=False)


def test_latency_gate_rejects_any_candidate_below_fifteen_percent():
    thresholds = GateThresholds(
        min_parameter_reduction_pct=0.0,
        min_gflops_reduction_pct=0.0,
        min_latency_reduction_pct=15.0,
    )

    rejected = evaluate_candidate(
        baseline_parameters=100,
        candidate_parameters=90,
        baseline_gflops=10,
        candidate_gflops=9,
        baseline_latency_ms=10,
        candidate_latency_ms=8.51,
        thresholds=thresholds,
    )
    accepted = evaluate_candidate(
        baseline_parameters=100,
        candidate_parameters=90,
        baseline_gflops=10,
        candidate_gflops=9,
        baseline_latency_ms=10,
        candidate_latency_ms=8.50,
        thresholds=thresholds,
    )

    assert rejected["failed_gates"] == ["latency"]
    assert accepted["train_allowed"] is True


@pytest.mark.parametrize(("path", "pose_channels"), CANDIDATES.values())
def test_joint_candidate_reduces_backbone_neck_and_head(path, pose_channels):
    baseline = build(BASELINE)
    candidate = build(path)

    assert isinstance(candidate.model[10], LitePSA)
    assert isinstance(candidate.model[6], C2f)
    assert len(candidate.model[6].m) == 1
    assert len(baseline.model[6].m) == 2
    assert candidate.model[6].c < baseline.model[6].c

    for index in (13, 16, 19):
        assert isinstance(candidate.model[index], C2f)
        assert candidate.model[index].c < baseline.model[index].c
    assert isinstance(candidate.model[22], C2fCIB)
    assert candidate.model[22].c < baseline.model[22].c

    head = candidate.model[-1]
    assert head.pose_head_channels == pose_channels
    assert head.pose_head_depth == 2
    assert head.box_head_depth == 1
    assert sum(p.numel() for p in candidate.parameters()) < sum(p.numel() for p in baseline.parameters())


@pytest.mark.parametrize(("path", "_pose_channels"), CANDIDATES.values())
def test_joint_candidate_preserves_dual_assignment_output_contract(path, _pose_channels):
    model = build(path).train()
    image = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(image)

    assert set(output) == {"one2many", "one2one"}
    for features, keypoints in output.values():
        assert len(features) == 3
        assert keypoints.shape == (2, 42, 84)
        assert all(torch.isfinite(feature).all() for feature in features)
        assert torch.isfinite(keypoints).all()


@pytest.mark.parametrize(
    "path",
    [CANDIDATES["e4375_head96"][0], CANDIDATES["e375_head96"][0]],
)
def test_head96_candidates_transfer_the_complete_pose_tower(path):
    teacher = build(BASELINE)
    student = build(path)

    transplant_compatible_weights(student, teacher)

    teacher_pose = {
        name: value
        for name, value in teacher.model[-1].state_dict().items()
        if name.startswith(("cv4.", "one2one_cv4."))
    }
    student_pose = {
        name: value
        for name, value in student.model[-1].state_dict().items()
        if name.startswith(("cv4.", "one2one_cv4."))
    }
    assert teacher_pose.keys() == student_pose.keys()
    assert all(torch.equal(student_pose[name], teacher_pose[name]) for name in teacher_pose)


@pytest.mark.parametrize(("path", "feature_channels"), CHANNEL_SLIM_CANDIDATES.values())
def test_channel_slim_candidates_reduce_backbone_neck_and_box_head(path, feature_channels):
    baseline = build(BASELINE)
    candidate = build(path)

    p3_channels, p4_channels, p5_channels = feature_channels
    assert candidate.model[4].cv2.conv.out_channels == p3_channels
    assert candidate.model[6].cv2.conv.out_channels == p4_channels
    assert candidate.model[10].cv2.conv.out_channels == p5_channels
    assert candidate.model[16].cv2.conv.out_channels == p3_channels
    assert candidate.model[19].cv2.conv.out_channels == p4_channels
    assert candidate.model[22].cv2.conv.out_channels == p5_channels
    assert isinstance(candidate.model[10], LitePSA)

    head = candidate.model[-1]
    assert head.pose_head_channels == 96
    assert head.pose_head_depth == 2
    assert head.box_head_depth == 1
    assert tuple(head.cv4[i][0].conv.in_channels for i in range(3)) == feature_channels
    assert sum(p.numel() for p in candidate.parameters()) < sum(p.numel() for p in baseline.parameters())


@pytest.mark.parametrize(("path", "_feature_channels"), CHANNEL_SLIM_CANDIDATES.values())
def test_channel_slim_candidates_preserve_dual_assignment_output_contract(path, _feature_channels):
    model = build(path).train()
    image = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(image)

    assert set(output) == {"one2many", "one2one"}
    for features, keypoints in output.values():
        assert len(features) == 3
        assert keypoints.shape == (2, 42, 84)
        assert all(torch.isfinite(feature).all() for feature in features)
        assert torch.isfinite(keypoints).all()


@pytest.mark.parametrize(("path", "neck_block"), KERNEL_SLIM_CANDIDATES.values())
def test_kernel_slim_candidates_reduce_serial_kernels_and_keep_full_width(path, neck_block):
    baseline = build(BASELINE)
    candidate = build(path)

    assert all(isinstance(candidate.model[index], RepC3) for index in (4, 6, 8))
    assert all(isinstance(candidate.model[index], neck_block) for index in (13, 16, 19, 22))
    assert isinstance(candidate.model[10], LitePSA)
    assert tuple(candidate.model[index].cv3.conv.out_channels for index in (4, 6, 8)) == (64, 128, 256)

    head = candidate.model[-1]
    assert head.pose_head_channels == 96
    assert head.pose_head_depth == 2
    assert head.box_head_depth == 1

    baseline_convs = sum(isinstance(module, torch.nn.Conv2d) for module in baseline.modules())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_parameters < baseline_parameters
    candidate.fuse()
    candidate_convs = sum(isinstance(module, torch.nn.Conv2d) for module in candidate.modules())
    assert not any(isinstance(module, RepConv) and hasattr(module, "conv1") for module in candidate.modules())
    assert candidate_convs <= baseline_convs - 7


@pytest.mark.parametrize(("path", "_neck_block"), KERNEL_SLIM_CANDIDATES.values())
def test_kernel_slim_candidates_preserve_dual_assignment_output_contract(path, _neck_block):
    model = build(path).train()
    image = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(image)

    assert set(output) == {"one2many", "one2one"}
    for features, keypoints in output.values():
        assert len(features) == 3
        assert keypoints.shape == (2, 42, 84)
        assert all(torch.isfinite(feature).all() for feature in features)
        assert torch.isfinite(keypoints).all()
