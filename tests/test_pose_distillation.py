"""Tests for the isolated resolution-aware YOLOv10 pose distillation experiment."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.models.yolov10.distill import (
    DistillationConfig,
    YOLOv10PoseDistillationModel,
    feature_pyramid_distillation_loss,
)
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
POSE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"


def _training_batch(batch_size=1):
    keypoints = torch.zeros(1, 14, 3)
    keypoints[..., :2] = 0.5
    keypoints[..., 2] = 2
    return {
        "img": torch.rand(batch_size, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "keypoints": keypoints,
    }


def test_feature_distillation_is_zero_for_identical_features():
    features = (torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4))
    loss = feature_pyramid_distillation_loss(features, tuple(x.clone() for x in features))
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_feature_distillation_aligns_resolution_and_only_updates_student():
    student = (torch.randn(2, 8, 8, 8, requires_grad=True), torch.randn(2, 16, 4, 4, requires_grad=True))
    teacher = (torch.randn(2, 8, 10, 10, requires_grad=True), torch.randn(2, 16, 5, 5, requires_grad=True))

    loss = feature_pyramid_distillation_loss(student, teacher)
    loss.backward()

    assert torch.isfinite(loss) and loss.item() > 0
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in student)
    assert all(x.grad is None for x in teacher)


def test_feature_distillation_rejects_incompatible_pyramids():
    with pytest.raises(ValueError, match="same number"):
        feature_pyramid_distillation_loss((torch.rand(1, 8, 4, 4),), ())
    with pytest.raises(ValueError, match="channel"):
        feature_pyramid_distillation_loss((torch.rand(1, 8, 4, 4),), (torch.rand(1, 16, 4, 4),))


def test_distillation_config_validates_ranges():
    assert DistillationConfig(weight=0.5, teacher_imgsz=640).weight == 0.5
    with pytest.raises(ValueError, match="weight"):
        DistillationConfig(weight=-0.1, teacher_imgsz=640)
    with pytest.raises(ValueError, match="teacher_imgsz"):
        DistillationConfig(weight=0.5, teacher_imgsz=0)


def test_distillation_model_adds_frozen_teacher_loss_without_registering_teacher():
    student = YOLOv10PoseDistillationModel(POSE_CFG, verbose=False).train()
    teacher = YOLOv10PoseModel(POSE_CFG, verbose=False).eval()
    student.configure_distillation(teacher, DistillationConfig(weight=0.5, teacher_imgsz=96))

    assert all(not p.requires_grad for p in teacher.parameters())
    assert not any(name.startswith("distillation_teacher") for name, _ in student.named_parameters())

    loss, items = student.loss(_training_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert items.shape == (10,)
    assert student.last_distillation_loss.item() >= 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_distillation_model_behaves_like_baseline_when_not_configured():
    model = YOLOv10PoseDistillationModel(POSE_CFG, verbose=False).train()
    loss, items = model.loss(_training_batch())
    assert torch.isfinite(loss)
    assert items.shape == (10,)
    assert model.last_distillation_loss.item() == 0


def test_checkpoint_copy_does_not_embed_the_teacher():
    student = YOLOv10PoseDistillationModel(POSE_CFG, verbose=False)
    teacher = YOLOv10PoseModel(POSE_CFG, verbose=False)
    student.configure_distillation(teacher)

    checkpoint_copy = deepcopy(student)

    assert checkpoint_copy.distillation_teacher is None
    assert checkpoint_copy.distillation_config is None
