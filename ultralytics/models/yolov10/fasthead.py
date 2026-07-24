"""Weight transfer helpers for explicitly narrowed YOLOv10 Vehicle Pose heads."""

from copy import deepcopy

import torch

from ultralytics.nn.tasks import YOLOv10PoseModel


def transplant_compatible_weights(student, teacher):
    """Load identical tensors and remap final outputs when a tower loses a hidden layer."""
    if not isinstance(student, YOLOv10PoseModel) or not isinstance(teacher, YOLOv10PoseModel):
        raise TypeError("student and teacher must both be YOLOv10PoseModel instances")

    source_state = teacher.state_dict()
    target_state = student.state_dict()
    direct = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    incompatible = student.load_state_dict(direct, strict=False)

    remapped = []
    student_head = student.model[-1]
    teacher_head = teacher.model[-1]
    for branch_name in ("cv2", "one2one_cv2", "cv4", "one2one_cv4"):
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

    if hasattr(teacher, "names"):
        student.names = deepcopy(teacher.names)
    if hasattr(teacher, "args"):
        student.args = deepcopy(teacher.args)
    return {
        "direct_tensor_count": len(direct),
        "remapped_tensor_count": len(remapped),
        "remapped_tensors": remapped,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
    }
