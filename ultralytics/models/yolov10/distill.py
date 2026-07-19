"""Resolution-aware feature distillation for the vehicle-pose model.

This module is intentionally opt-in.  The regular YOLOv10 pose model and
trainer do not import it, so existing baseline training and checkpoints keep
their original behaviour.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import YOLOv10PoseModel


@dataclass(frozen=True)
class DistillationConfig:
    """Settings for feature-pyramid distillation."""

    weight: float = 0.5
    teacher_imgsz: int = 640

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("distillation weight must be non-negative")
        if self.teacher_imgsz <= 0:
            raise ValueError("teacher_imgsz must be positive")


def feature_pyramid_distillation_loss(student_features, teacher_features):
    """Return channel-normalized MSE across corresponding FPN levels.

    Teacher features are detached and resized to the student spatial shape.
    Keeping the loss in normalized feature space makes its scale less
    sensitive to the number of channels or mixed-precision activation range.
    """

    if len(student_features) != len(teacher_features):
        raise ValueError("student and teacher must expose the same number of feature levels")
    if not student_features:
        raise ValueError("feature pyramids must not be empty")

    losses = []
    for level, (student, teacher) in enumerate(zip(student_features, teacher_features)):
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError(f"feature level {level} must contain BCHW tensors")
        if student.shape[:2] != teacher.shape[:2]:
            raise ValueError(
                f"feature level {level} batch/channel mismatch: {tuple(student.shape[:2])} vs "
                f"{tuple(teacher.shape[:2])}"
            )
        teacher = teacher.detach()
        if teacher.shape[-2:] != student.shape[-2:]:
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)
        student = F.normalize(student.float(), p=2, dim=1, eps=1e-6)
        teacher = F.normalize(teacher.float(), p=2, dim=1, eps=1e-6)
        losses.append((student - teacher).square().sum(dim=1).mean())
    return torch.stack(losses).mean()


@contextmanager
def _capture_head_features(model):
    """Capture the P3/P4/P5 tensors passed into a YOLO detection head."""

    captured = []

    def capture(_module, inputs):
        features = inputs[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError("YOLO head input must be a feature list or tuple")
        captured[:] = list(features)

    handle = model.model[-1].register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


class YOLOv10PoseDistillationModel(YOLOv10PoseModel):
    """YOLOv10 pose student with an optional, unregistered frozen teacher."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("last_distillation_loss", torch.tensor(0.0), persistent=False)
        self.__dict__["distillation_teacher"] = None
        self.distillation_config = None

    def configure_distillation(self, teacher, config=None):
        """Attach a frozen teacher without adding it to the student optimizer."""

        if not isinstance(teacher, YOLOv10PoseModel):
            raise TypeError("teacher must be a YOLOv10PoseModel")
        config = config or DistillationConfig()
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        # Direct __dict__ storage deliberately avoids nn.Module child registration.
        self.__dict__["distillation_teacher"] = teacher
        self.distillation_config = config
        return self

    def _apply(self, fn):
        """Move the hidden teacher together with the student (CPU/CUDA/dtype)."""

        super()._apply(fn)
        teacher = self.__dict__.get("distillation_teacher")
        if teacher is not None:
            teacher._apply(fn)
        return self

    def __getstate__(self):
        """Exclude the training-only teacher from EMA and checkpoint copies."""

        state = super().__getstate__().copy()
        state["distillation_teacher"] = None
        state["distillation_config"] = None
        return state

    def loss(self, batch, preds=None):
        """Add feature KD to the unchanged ground-truth pose loss during training."""

        teacher = self.__dict__.get("distillation_teacher")
        config = self.distillation_config
        if teacher is None or config is None or not self.training or preds is not None:
            self.last_distillation_loss.zero_()
            return super().loss(batch, preds)

        with _capture_head_features(self) as student_features:
            student_predictions = self.forward(batch["img"])
        supervised_loss, loss_items = super().loss(batch, student_predictions)

        teacher_image = batch["img"]
        if teacher_image.shape[-2:] != (config.teacher_imgsz, config.teacher_imgsz):
            teacher_image = F.interpolate(
                teacher_image, size=(config.teacher_imgsz, config.teacher_imgsz), mode="bilinear", align_corners=False
            )
        teacher.eval()
        with torch.no_grad(), _capture_head_features(teacher) as teacher_features:
            teacher(teacher_image)

        distillation_loss = feature_pyramid_distillation_loss(student_features, teacher_features)
        self.last_distillation_loss.copy_(distillation_loss.detach())
        total_loss = supervised_loss + config.weight * distillation_loss * batch["img"].shape[0]
        return total_loss, loss_items
