"""Weight transfer helpers for explicitly narrowed YOLOv10 Vehicle Pose heads."""

from copy import deepcopy

import torch

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.tasks import YOLOv10PoseModel


def fold_c2f_output_projection(target, source):
    """Fold one omitted residual feature group into the last retained C2f group."""

    if (
        target.ndim != 4
        or source.ndim != 4
        or target.shape[0] != source.shape[0]
        or target.shape[2:] != source.shape[2:]
        or source.shape[1] <= target.shape[1]
    ):
        raise ValueError(f"incompatible C2f projection shapes: target={target.shape}, source={source.shape}")
    omitted = source.shape[1] - target.shape[1]
    if omitted > target.shape[1]:
        raise ValueError("omitted C2f group is wider than the retained projection")
    folded = source[:, : target.shape[1]].clone()
    folded[:, -omitted:] += source[:, target.shape[1] :]
    return folded


def slice_compatible_state_tensors(target_state, source_state):
    """Crop same-name tensors when every student dimension fits in the teacher.

    Channel dimensions use prefix selection to preserve existing narrowed-model
    behavior. Spatial convolution kernels use a centered crop so a 5x5 filter
    becomes its aligned 3x3 core instead of an incorrectly shifted top-left tile.
    """
    sliced = {}
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is None or source.shape == target.shape or source.ndim != target.ndim or source.dtype != target.dtype:
            continue
        if source.ndim and all(target_size <= source_size for target_size, source_size in zip(target.shape, source.shape)):
            slices = []
            for dimension, (target_size, source_size) in enumerate(zip(target.shape, source.shape)):
                if source.ndim == 4 and dimension >= 2 and target_size < source_size:
                    offset = (source_size - target_size) // 2
                    slices.append(slice(offset, offset + target_size))
                else:
                    slices.append(slice(0, target_size))
            sliced[name] = source[tuple(slices)].clone()
    return sliced


def pad_compatible_state_tensors(target_state, source_state):
    """Copy a source prefix into same-name tensors widened in every dimension."""

    padded = {}
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is None or source.shape == target.shape or source.ndim != target.ndim or source.dtype != target.dtype:
            continue
        if source.ndim and all(source_size <= target_size for target_size, source_size in zip(target.shape, source.shape)):
            widened = target.clone()
            widened[tuple(slice(0, size) for size in source.shape)] = source
            padded[name] = widened
    return padded


def transplant_compatible_weights(student, teacher):
    """Load identical/narrowed tensors and remap outputs when a tower loses a layer."""
    if not isinstance(student, YOLOv10PoseModel) or not isinstance(teacher, YOLOv10PoseModel):
        raise TypeError("student and teacher must both be YOLOv10PoseModel instances")

    source_state = teacher.state_dict()
    target_state = student.state_dict()
    direct = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    sliced = slice_compatible_state_tensors(target_state, source_state)
    padded = pad_compatible_state_tensors(target_state, source_state)
    incompatible = student.load_state_dict({**direct, **sliced, **padded}, strict=False)

    folded = []
    for student_layer, teacher_layer in zip(student.model, teacher.model):
        if not isinstance(student_layer, C2f) or not isinstance(teacher_layer, C2f):
            continue
        if len(teacher_layer.m) - len(student_layer.m) != 1:
            continue
        target_projection = student_layer.cv2.conv.weight
        source_projection = teacher_layer.cv2.conv.weight
        if source_projection.shape[1] - target_projection.shape[1] != student_layer.c:
            continue
        target_projection.data.copy_(fold_c2f_output_projection(target_projection, source_projection))
        folded.append(f"model.{student_layer.i}.cv2.conv.weight")

    remapped = []
    student_head = student.model[-1]
    teacher_head = teacher.model[-1]
    if getattr(student_head, "class_head_depth", 2) == 3 and getattr(teacher_head, "class_head_depth", 2) == 2:
        for branch_name in ("cv3", "one2one_cv3"):
            for level, (target_tower, source_tower) in enumerate(
                zip(getattr(student_head, branch_name), getattr(teacher_head, branch_name))
            ):
                target_state = target_tower[1].state_dict()
                source_state = source_tower[1][1].state_dict()
                direct_pointwise = {
                    name: tensor
                    for name, tensor in source_state.items()
                    if name in target_state and target_state[name].shape == tensor.shape
                }
                sliced_pointwise = slice_compatible_state_tensors(target_state, source_state)
                target_tower[1].load_state_dict({**direct_pointwise, **sliced_pointwise}, strict=False)
                remapped.append(f"{branch_name}.{level}.lite_pointwise")
    for branch_name in ("cv2", "one2one_cv2", "cv3", "one2one_cv3", "cv4", "one2one_cv4"):
        for level, (target_tower, source_tower) in enumerate(
            zip(getattr(student_head, branch_name), getattr(teacher_head, branch_name))
        ):
            target_output, source_output = target_tower[-1], source_tower[-1]
            for tensor_name in ("weight", "bias"):
                target_tensor = getattr(target_output, tensor_name)
                source_tensor = getattr(source_output, tensor_name)
                if target_tensor is not None and source_tensor is not None and target_tensor.shape == source_tensor.shape:
                    target_tensor.data.copy_(source_tensor.data)
                    remapped.append(f"{branch_name}.{level}.output.{tensor_name}")
                elif (
                    tensor_name == "weight"
                    and target_tensor is not None
                    and source_tensor is not None
                    and target_tensor.ndim == source_tensor.ndim == 4
                    and target_tensor.shape[0] == source_tensor.shape[0]
                    and target_tensor.shape[1] >= source_tensor.shape[1]
                    and target_tensor.shape[2:] == source_tensor.shape[2:]
                ):
                    target_tensor.data.zero_()
                    target_tensor.data[:, : source_tensor.shape[1]].copy_(source_tensor.data)
                    remapped.append(f"{branch_name}.{level}.output.{tensor_name}.padded")

    if hasattr(teacher, "names"):
        student.names = deepcopy(teacher.names)
    if hasattr(teacher, "args"):
        student.args = deepcopy(teacher.args)
    return {
        "direct_tensor_count": len(direct),
        "sliced_tensor_count": len(sliced),
        "sliced_tensors": sorted(sliced),
        "padded_tensor_count": len(padded),
        "padded_tensors": sorted(padded),
        "folded_tensor_count": len(folded),
        "folded_tensors": folded,
        "remapped_tensor_count": len(remapped),
        "remapped_tensors": remapped,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
    }
