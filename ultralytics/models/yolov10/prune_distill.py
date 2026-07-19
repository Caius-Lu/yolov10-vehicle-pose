"""Opt-in structural pruning and output distillation for YOLOv10 Vehicle Pose.

The baseline model and trainer do not import this module. Structural pruning is
performed through Torch-Pruning's dependency graph so channels are physically
removed instead of masked. A full-width frozen teacher then supervises the raw
detection and pose outputs of the pruned student at the same resolution.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f, C2fCIB, PSA, SCDown, SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import YOLOv10PoseModel


@dataclass(frozen=True)
class PruningConfig:
    """Conservative dependency-aware channel-pruning settings."""

    pruning_ratio: float = 0.15
    iterative_steps: int = 1
    round_to: int = 8
    global_pruning: bool = False
    measure_macs: bool = False

    def __post_init__(self):
        if not 0.0 < self.pruning_ratio < 0.5:
            raise ValueError("pruning_ratio must be greater than 0 and less than 0.5")
        if self.iterative_steps <= 0:
            raise ValueError("iterative_steps must be positive")
        if self.round_to <= 0:
            raise ValueError("round_to must be positive")


@dataclass(frozen=True)
class OutputDistillationConfig:
    """Weights for same-resolution raw-output knowledge distillation."""

    weight: float = 0.25
    teacher_imgsz: int = 640
    temperature: float = 2.0
    detection_weight: float = 1.0
    pose_weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("distillation weight must be non-negative")
        if self.teacher_imgsz <= 0:
            raise ValueError("teacher_imgsz must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.detection_weight < 0 or self.pose_weight < 0:
            raise ValueError("detection and pose distillation weights must be non-negative")


def count_parameters(model):
    """Return the physical parameter count of a model."""

    return sum(parameter.numel() for parameter in model.parameters())


def build_pruning_report(parameters_before, parameters_after, **extra):
    """Build a report and reject masking/no-op pruning."""

    parameters_before = int(parameters_before)
    parameters_after = int(parameters_after)
    if parameters_before <= 0:
        raise ValueError("parameters_before must be positive")
    if parameters_after >= parameters_before:
        raise RuntimeError("structural pruning did not reduce the physical parameter count")
    report = {
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "parameters_removed": parameters_before - parameters_after,
        "parameter_reduction_ratio": (parameters_before - parameters_after) / parameters_before,
    }
    report.update(extra)
    return report


def collect_head_output_layers(head):
    """Return final box/class/pose convolutions that must keep their output dimensions."""

    protected = []
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3", "cv4", "one2one_cv4"):
        towers = getattr(head, name, None)
        if towers is None:
            raise TypeError(f"pose head is missing required branch {name}")
        for tower in towers:
            output = tower[-1]
            if not isinstance(output, nn.Conv2d):
                raise TypeError(f"{name} tower must end with Conv2d, got {type(output).__name__}")
            protected.append(output)
    return protected


def _head_hidden_convolutions(head):
    protected_ids = {id(layer) for layer in collect_head_output_layers(head)}
    roots = []
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3", "cv4", "one2one_cv4"):
        for tower in getattr(head, name):
            for module in tower.modules():
                if isinstance(module, Conv) and module.conv.groups == 1 and id(module.conv) not in protected_ids:
                    roots.append(module.conv)
    return roots


def collect_prunable_root_layers(model):
    """Select conservative roots whose output width has no stale split attribute.

    Internal C2f/PSA split convolutions and depthwise roots are deliberately not
    selected. They may still be adjusted by DepGraph as dependent layers.
    """

    if not isinstance(model, YOLOv10PoseModel):
        raise TypeError("model must be a YOLOv10PoseModel")
    roots = []
    for block in model.model[:-1]:
        if isinstance(block, Conv) and block.conv.groups == 1:
            roots.append(block.conv)
        elif isinstance(block, (C2f, C2fCIB)):
            roots.append(block.cv2.conv)
        elif isinstance(block, SPPF):
            roots.append(block.cv2.conv)
        elif isinstance(block, SCDown):
            roots.append(block.cv1.conv)
        elif isinstance(block, PSA):
            roots.append(block.cv2.conv)
    roots.extend(_head_hidden_convolutions(model.model[-1]))
    unique = []
    seen = set()
    for layer in roots:
        if id(layer) not in seen:
            unique.append(layer)
            seen.add(id(layer))
    return unique


def _load_pruning_backend():
    try:
        import torch_pruning as tp
    except ImportError as error:
        raise RuntimeError(
            "Torch-Pruning is required for structural pruning; install requirements-pruning.txt"
        ) from error
    return tp


def structurally_prune_model(model, example_input, config=None, pruning_backend=None):
    """Physically remove coupled channels and return the modified model plus evidence."""

    config = config or PruningConfig()
    backend = pruning_backend or _load_pruning_backend()
    if not isinstance(example_input, torch.Tensor) or example_input.ndim != 4:
        raise ValueError("example_input must be a BCHW tensor")
    head = model.model[-1]
    protected = collect_head_output_layers(head)
    protected_widths = [layer.out_channels for layer in protected]
    roots = collect_prunable_root_layers(model)
    if not roots:
        raise RuntimeError("no conservative pruning roots were found")

    parameters_before = count_parameters(model)
    # Ultralytics inference loading prefers EMA weights, whose parameters are
    # frozen. DepGraph relies on Autograd to discover coupled channels, and the
    # pruned student must be trainable for recovery fine-tuning.
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)
    macs_before = None
    if config.measure_macs and hasattr(backend, "utils") and hasattr(backend.utils, "count_ops_and_params"):
        macs_before, _ = backend.utils.count_ops_and_params(model, example_input)

    importance_cls = getattr(backend.importance, "GroupMagnitudeImportance", None)
    if importance_cls is None:
        importance_cls = backend.importance.MagnitudeImportance
    importance = importance_cls(p=2)
    pruner_cls = getattr(backend.pruner, "MetaPruner", None)
    if pruner_cls is None:
        pruner_cls = backend.pruner.MagnitudePruner

    original_training = model.training
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    pruner = pruner_cls(
        model,
        example_input,
        importance=importance,
        # Keep the global default at zero. Only conservative roots receive a
        # non-zero ratio; coupled inputs/BN/depthwise layers are then adjusted
        # by DepGraph instead of becoming independent pruning roots.
        pruning_ratio=0.0,
        pruning_ratio_dict={layer: config.pruning_ratio for layer in roots},
        iterative_steps=config.iterative_steps,
        global_pruning=config.global_pruning,
        ignored_layers=protected,
        round_to=config.round_to,
        root_module_types=[nn.Conv2d],
    )
    for _ in range(config.iterative_steps):
        pruner.step()
    model.train(original_training)

    if [layer.out_channels for layer in protected] != protected_widths:
        raise RuntimeError("pruning changed a protected detection or pose output dimension")
    parameters_after = count_parameters(model)
    macs_after = None
    if config.measure_macs and hasattr(backend, "utils") and hasattr(backend.utils, "count_ops_and_params"):
        macs_after, _ = backend.utils.count_ops_and_params(model, example_input)
    model.zero_grad(set_to_none=True)
    report = build_pruning_report(
        parameters_before,
        parameters_after,
        pruning_config=asdict(config),
        prunable_root_count=len(roots),
        protected_output_count=len(protected),
        macs_before=float(macs_before) if macs_before is not None else None,
        macs_after=float(macs_after) if macs_after is not None else None,
    )
    if macs_before:
        report["mac_reduction_ratio"] = (float(macs_before) - float(macs_after)) / float(macs_before)
    model.pruning_metadata = report
    return model, report


def _validate_branch(branch, name):
    if not isinstance(branch, (tuple, list)) or len(branch) != 2:
        raise ValueError(f"{name} must be a (detections, keypoints) pair")
    detections, keypoints = branch
    if not isinstance(detections, (tuple, list)) or not detections:
        raise ValueError(f"{name} detections must be a non-empty feature list")
    if not isinstance(keypoints, torch.Tensor) or keypoints.ndim != 3:
        raise ValueError(f"{name} keypoints must be a BCN tensor")
    return detections, keypoints


def _bernoulli_logit_mse(student, teacher, temperature):
    return F.mse_loss(torch.sigmoid(student / temperature), torch.sigmoid(teacher / temperature)) * temperature**2


def pose_output_distillation_loss(
    student_predictions,
    teacher_predictions,
    reg_max,
    kpt_shape,
    temperature=2.0,
    detection_weight=1.0,
    pose_weight=1.0,
):
    """Distill raw DFL/class/pose logits while allowing different hidden widths."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    branches = ("one2many", "one2one")
    losses = []
    for branch_name in branches:
        if branch_name not in student_predictions or branch_name not in teacher_predictions:
            raise ValueError(f"student and teacher must both contain {branch_name}")
        student_detections, student_pose = _validate_branch(student_predictions[branch_name], branch_name)
        teacher_detections, teacher_pose = _validate_branch(teacher_predictions[branch_name], branch_name)
        if len(student_detections) != len(teacher_detections):
            raise ValueError(f"{branch_name} detection pyramid shape mismatch")

        detection_losses = []
        for student_level, teacher_level in zip(student_detections, teacher_detections):
            teacher_level = teacher_level.detach()
            if student_level.shape != teacher_level.shape:
                raise ValueError(
                    f"{branch_name} detection shape mismatch: {tuple(student_level.shape)} vs "
                    f"{tuple(teacher_level.shape)}"
                )
            box_channels = int(reg_max) * 4
            if student_level.shape[1] <= box_channels:
                raise ValueError("detection tensor has no class-logit channels")
            student_box = student_level[:, :box_channels].reshape(student_level.shape[0], 4, reg_max, -1)
            teacher_box = teacher_level[:, :box_channels].reshape(teacher_level.shape[0], 4, reg_max, -1)
            dfl = F.kl_div(
                F.log_softmax(student_box / temperature, dim=2),
                F.softmax(teacher_box / temperature, dim=2),
                reduction="none",
            ).sum(dim=2).mean() * temperature**2
            cls = _bernoulli_logit_mse(
                student_level[:, box_channels:], teacher_level[:, box_channels:], temperature
            )
            detection_losses.append(dfl + cls)

        teacher_pose = teacher_pose.detach()
        if student_pose.shape != teacher_pose.shape:
            raise ValueError(
                f"{branch_name} pose shape mismatch: {tuple(student_pose.shape)} vs {tuple(teacher_pose.shape)}"
            )
        keypoint_count, dimensions = map(int, kpt_shape)
        if student_pose.shape[1] != keypoint_count * dimensions:
            raise ValueError("pose tensor channel count does not match kpt_shape")
        student_pose = student_pose.reshape(student_pose.shape[0], keypoint_count, dimensions, -1)
        teacher_pose = teacher_pose.reshape(teacher_pose.shape[0], keypoint_count, dimensions, -1)
        xy_loss = F.smooth_l1_loss(student_pose[:, :, :2], teacher_pose[:, :, :2])
        visibility_loss = (
            _bernoulli_logit_mse(student_pose[:, :, 2:3], teacher_pose[:, :, 2:3], temperature)
            if dimensions == 3
            else student_pose.new_zeros(())
        )
        detection_loss = torch.stack(detection_losses).mean()
        losses.append(detection_weight * detection_loss + pose_weight * (xy_loss + visibility_loss))
    return torch.stack(losses).mean()


@contextmanager
def _raw_teacher_predictions(model):
    """Make only the YOLO head emit training-format raw tensors; keep teacher BN frozen."""

    head = model.model[-1]
    original_training = head.training
    head.training = True
    try:
        yield
    finally:
        head.training = original_training


class YOLOv10PosePruningDistillationModel(YOLOv10PoseModel):
    """A structurally pruned pose model with an unregistered full-width teacher."""

    def configure_output_distillation(self, teacher, config=None):
        if not isinstance(teacher, YOLOv10PoseModel):
            raise TypeError("teacher must be a YOLOv10PoseModel")
        config = config or OutputDistillationConfig()
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.__dict__["distillation_teacher"] = teacher
        self.distillation_config = config
        return self

    def _apply(self, fn):
        super()._apply(fn)
        teacher = self.__dict__.get("distillation_teacher")
        if teacher is not None:
            teacher._apply(fn)
        return self

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["distillation_teacher"] = None
        state["distillation_config"] = None
        return state

    def loss(self, batch, preds=None):
        teacher = self.__dict__.get("distillation_teacher")
        config = self.distillation_config
        if teacher is None or config is None or not self.training or preds is not None:
            self.last_distillation_loss.zero_()
            return super().loss(batch, preds)

        student_predictions = self.forward(batch["img"])
        supervised_loss, loss_items = super().loss(batch, student_predictions)
        teacher_image = batch["img"]
        if teacher_image.shape[-2:] != (config.teacher_imgsz, config.teacher_imgsz):
            teacher_image = F.interpolate(
                teacher_image,
                size=(config.teacher_imgsz, config.teacher_imgsz),
                mode="bilinear",
                align_corners=False,
            )
        if teacher_image.shape[-2:] != batch["img"].shape[-2:]:
            raise ValueError("pruning distillation requires teacher_imgsz to match the student training resolution")
        teacher.eval()
        with torch.no_grad(), _raw_teacher_predictions(teacher):
            teacher_predictions = teacher(teacher_image)
        distillation_loss = pose_output_distillation_loss(
            student_predictions,
            teacher_predictions,
            reg_max=self.model[-1].reg_max,
            kpt_shape=self.model[-1].kpt_shape,
            temperature=config.temperature,
            detection_weight=config.detection_weight,
            pose_weight=config.pose_weight,
        )
        self.last_distillation_loss.copy_(distillation_loss.detach())
        total_loss = supervised_loss + config.weight * distillation_loss * batch["img"].shape[0]
        return total_loss, loss_items


def upgrade_for_pruning_distillation(model, teacher, config=None):
    """Add KD behaviour to an already-pruned model without rebuilding its structure."""

    if not isinstance(model, YOLOv10PoseModel):
        raise TypeError("student must be a YOLOv10PoseModel")
    if not isinstance(model, YOLOv10PosePruningDistillationModel):
        model.__class__ = YOLOv10PosePruningDistillationModel
    if "last_distillation_loss" not in model._buffers:
        model.register_buffer("last_distillation_loss", torch.tensor(0.0), persistent=False)
    model.__dict__["distillation_teacher"] = None
    model.distillation_config = None
    return model.configure_output_distillation(teacher, config)


def build_pruned_checkpoint(model, source_checkpoint, report):
    """Create an Ultralytics-compatible whole-model checkpoint for changed shapes."""

    checkpoint_model = deepcopy(model).half()
    checkpoint_model.zero_grad(set_to_none=True)
    return {
        "epoch": -1,
        "best_fitness": None,
        "model": checkpoint_model,
        "ema": None,
        "optimizer": None,
        "train_args": getattr(model, "args", {}),
        "pruning": {**report, "source_checkpoint": str(source_checkpoint)},
        "date": datetime.now().isoformat(),
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
    }
