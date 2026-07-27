"""Tests for opt-in structural pruning followed by pose-output distillation."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.models.yolov10.prune_distill import (
    OutputDistillationConfig,
    PruningConfig,
    build_pruned_checkpoint,
    build_pruning_report,
    collect_head_output_layers,
    collect_ignored_root_layers,
    collect_prunable_root_layers,
    count_parameters,
    feature_distillation_loss,
    pose_output_distillation_loss,
    prune_one2many_deployment_branches,
    upgrade_for_pruning_distillation,
)
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
POSE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"


def _raw_predictions(batch=1, anchors=(16, 4, 1), reg_max=16, keypoints=14):
    detections = [torch.randn(batch, reg_max * 4 + 1, size, 1) for size in anchors]
    pose = torch.randn(batch, keypoints * 3, sum(anchors))
    return detections, pose


def _dual_predictions(requires_grad=False):
    predictions = {
        "one2many": _raw_predictions(),
        "one2one": _raw_predictions(),
    }
    if requires_grad:
        for detections, pose in predictions.values():
            for tensor in [*detections, pose]:
                tensor.requires_grad_(True)
    return predictions


def _training_batch():
    keypoints = torch.zeros(1, 14, 3)
    keypoints[..., :2] = 0.5
    keypoints[..., 2] = 2
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "keypoints": keypoints,
    }


def test_pruning_config_rejects_destructive_or_invalid_settings():
    config = PruningConfig(pruning_ratio=0.15, iterative_steps=1, round_to=8)
    assert config.pruning_ratio == 0.15
    assert config.head_pruning_ratio is None
    assert config.measure_macs is False
    for ratio in (0.0, -0.1, 0.5, 1.0):
        with pytest.raises(ValueError, match="pruning_ratio"):
            PruningConfig(pruning_ratio=ratio)
    with pytest.raises(ValueError, match="iterative_steps"):
        PruningConfig(pruning_ratio=0.15, iterative_steps=0)
    with pytest.raises(ValueError, match="round_to"):
        PruningConfig(pruning_ratio=0.15, round_to=0)
    with pytest.raises(ValueError, match="head_pruning_ratio"):
        PruningConfig(pruning_ratio=0.15, head_pruning_ratio=-0.01)
    with pytest.raises(ValueError, match="head_pruning_ratio"):
        PruningConfig(pruning_ratio=0.15, head_pruning_ratio=0.5)
    with pytest.raises(ValueError, match="model_indices"):
        PruningConfig(pruning_ratio=0.15, model_indices=(-1,))
    with pytest.raises(ValueError, match="model_indices"):
        PruningConfig(pruning_ratio=0.15, model_indices=(4, 4))


def test_output_distillation_config_validates_weights_and_resolution():
    config = OutputDistillationConfig(
        weight=0.25,
        teacher_imgsz=640,
        temperature=2.0,
        feature_weight=0.5,
        feature_layers=(4, 6, 8, 13, 16, 19, 22),
    )
    assert config.weight == 0.25
    assert config.feature_weight == 0.5
    assert config.feature_layers[-1] == 22
    with pytest.raises(ValueError, match="weight"):
        OutputDistillationConfig(weight=-0.1)
    with pytest.raises(ValueError, match="teacher_imgsz"):
        OutputDistillationConfig(teacher_imgsz=0)
    with pytest.raises(ValueError, match="temperature"):
        OutputDistillationConfig(temperature=0)
    with pytest.raises(ValueError, match="feature_weight"):
        OutputDistillationConfig(feature_weight=-0.1)
    with pytest.raises(ValueError, match="feature_layers"):
        OutputDistillationConfig(feature_weight=0.5, feature_layers=())


def test_feature_distillation_is_normalized_and_only_backpropagates_into_student():
    student = [torch.randn(2, channels, 8, 8, requires_grad=True) for channels in (64, 128, 256)]
    teacher = [tensor.detach().clone().requires_grad_(True) for tensor in student]

    identical = feature_distillation_loss(student, teacher)
    assert identical.item() == pytest.approx(0.0, abs=1e-7)

    teacher = [torch.randn_like(tensor, requires_grad=True) for tensor in student]
    loss = feature_distillation_loss(student, teacher)
    loss.backward()

    assert torch.isfinite(loss) and loss.item() > 0
    assert all(tensor.grad is not None for tensor in student)
    assert all(tensor.grad is None for tensor in teacher)


def test_output_distillation_is_zero_for_identical_raw_predictions():
    student = _dual_predictions()
    teacher = deepcopy(student)

    loss = pose_output_distillation_loss(student, teacher, reg_max=16, kpt_shape=(14, 3), temperature=2.0)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_output_distillation_only_backpropagates_into_student():
    student = _dual_predictions(requires_grad=True)
    teacher = _dual_predictions(requires_grad=True)

    loss = pose_output_distillation_loss(student, teacher, reg_max=16, kpt_shape=(14, 3), temperature=2.0)
    loss.backward()

    assert torch.isfinite(loss) and loss.item() > 0
    assert all(t.grad is not None for branch in student.values() for t in [*branch[0], branch[1]])
    assert all(t.grad is None for branch in teacher.values() for t in [*branch[0], branch[1]])


def test_output_distillation_rejects_missing_branch_or_shape_mismatch():
    student = _dual_predictions()
    teacher = _dual_predictions()
    teacher.pop("one2one")
    with pytest.raises(ValueError, match="one2one"):
        pose_output_distillation_loss(student, teacher, reg_max=16, kpt_shape=(14, 3))

    teacher = _dual_predictions()
    teacher["one2many"][0][0] = torch.randn(1, 65, 8, 1)
    with pytest.raises(ValueError, match="shape"):
        pose_output_distillation_loss(student, teacher, reg_max=16, kpt_shape=(14, 3))


def test_all_detection_and_pose_output_convolutions_are_protected():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)
    head = model.model[-1]

    protected = collect_head_output_layers(head)

    assert len(protected) == head.nl * 6
    assert len({id(layer) for layer in protected}) == len(protected)
    assert {layer.out_channels for layer in protected} == {head.reg_max * 4, head.nc, head.nk}


def test_only_conservative_convolutions_are_available_as_pruning_roots():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)
    roots = collect_prunable_root_layers(model)

    ignored = collect_ignored_root_layers(model, roots)

    all_convolutions = {module for module in model.modules() if isinstance(module, torch.nn.Conv2d)}
    assert set(roots).isdisjoint(ignored)
    assert set(roots) | set(ignored) == all_convolutions
    assert set(collect_head_output_layers(model.model[-1])).issubset(ignored)


def test_head_hidden_roots_can_be_fully_protected_for_accuracy_repair():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)
    all_roots = collect_prunable_root_layers(model)
    protected_head_roots = collect_prunable_root_layers(model, include_head_hidden=False)
    head_hidden = set(all_roots) - set(protected_head_roots)

    assert head_hidden
    assert set(protected_head_roots).issubset(all_roots)
    assert head_hidden.isdisjoint(protected_head_roots)
    assert head_hidden.issubset(set(collect_ignored_root_layers(model, protected_head_roots)))


def test_backbone_neck_pruning_roots_can_be_limited_to_profiled_model_indices():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)

    roots = collect_prunable_root_layers(
        model,
        include_head_hidden=False,
        model_indices=(4, 6, 10, 13, 16, 19, 22),
    )

    assert len(roots) == 7
    assert roots == [
        model.model[4].cv2.conv,
        model.model[6].cv2.conv,
        model.model[10].cv2.conv,
        model.model[13].cv2.conv,
        model.model[16].cv2.conv,
        model.model[19].cv2.conv,
        model.model[22].cv2.conv,
    ]


def test_pruning_report_requires_real_structural_reduction():
    baseline = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 4, 1))
    pruned = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 4, 1))

    report = build_pruning_report(count_parameters(baseline), count_parameters(pruned))

    assert report["parameters_after"] < report["parameters_before"]
    assert 0 < report["parameter_reduction_ratio"] < 1
    with pytest.raises(RuntimeError, match="did not reduce"):
        build_pruning_report(count_parameters(baseline), count_parameters(baseline))


def test_pruned_model_can_be_upgraded_for_distillation_without_registering_teacher():
    student = YOLOv10PoseModel(POSE_CFG, verbose=False)
    teacher = YOLOv10PoseModel(POSE_CFG, verbose=False)
    upgraded = upgrade_for_pruning_distillation(
        student,
        teacher,
        OutputDistillationConfig(weight=0.25, teacher_imgsz=64),
    ).train()

    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not any(name.startswith("distillation_teacher") for name, _ in upgraded.named_parameters())

    loss, items = upgraded.loss(_training_batch())

    assert torch.isfinite(loss) and items.shape == (10,)
    assert upgraded.last_distillation_loss.item() >= 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in upgraded.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_checkpoint_copy_drops_pruning_distillation_teacher():
    student = YOLOv10PoseModel(POSE_CFG, verbose=False)
    teacher = YOLOv10PoseModel(POSE_CFG, verbose=False)
    upgraded = upgrade_for_pruning_distillation(student, teacher, OutputDistillationConfig())

    checkpoint_copy = deepcopy(upgraded)

    assert checkpoint_copy.distillation_teacher is None
    assert checkpoint_copy.distillation_config is None


def test_upgrade_drops_stale_loss_criterion_before_device_migration():
    student = YOLOv10PoseModel(POSE_CFG, verbose=False)
    teacher = YOLOv10PoseModel(POSE_CFG, verbose=False)
    stale_criterion = object()
    student.criterion = stale_criterion

    upgraded = upgrade_for_pruning_distillation(student, teacher, OutputDistillationConfig())

    assert not hasattr(upgraded, "criterion")


def test_pruned_checkpoint_drops_stale_loss_criterion_before_recovery_training():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)
    model.criterion = object()

    checkpoint = build_pruned_checkpoint(
        model,
        source_checkpoint="baseline.pt",
        report={"parameters_before": 2, "parameters_after": 1},
    )

    assert not hasattr(checkpoint["model"], "criterion")


def test_deployment_branch_pruning_preserves_one2one_output_and_removes_parameters():
    baseline = YOLOv10PoseModel(POSE_CFG, verbose=False).eval()
    baseline.model[-1].one2one_only = True
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = baseline(image)["one2one"][0]
    parameters_before = count_parameters(baseline)

    pruned, report = prune_one2many_deployment_branches(deepcopy(baseline))

    with torch.inference_mode():
        actual = pruned(image)["one2one"][0]
    assert torch.equal(actual, expected)
    assert report["parameters_before"] == parameters_before
    assert report["parameters_after"] == count_parameters(pruned) < parameters_before
    assert report["deployment_only"] is True
    head = pruned.model[-1]
    assert not head.cv2 and not head.cv3 and not head.cv4
    assert head.one2one_only is True

    pruned.train()
    with pytest.raises(RuntimeError, match="deployment-only"):
        pruned(image)
