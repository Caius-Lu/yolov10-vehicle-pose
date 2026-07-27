from pathlib import Path

import torch

from ultralytics.models.yolov10.fasthead import (
    fold_c2f_output_projection,
    pad_compatible_state_tensors,
    slice_compatible_state_tensors,
    transplant_compatible_weights,
)
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"
FAST_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead-boxd1.yaml"


def test_slice_compatible_state_tensors_crops_all_narrowed_dimensions():
    source = {
        "conv.weight": torch.arange(8 * 6 * 3 * 3).reshape(8, 6, 3, 3).float(),
        "bn.weight": torch.arange(8).float(),
        "wider.weight": torch.ones(2, 2),
    }
    target = {
        "conv.weight": torch.zeros(4, 3, 3, 3),
        "bn.weight": torch.zeros(4),
        "wider.weight": torch.zeros(3, 2),
    }

    sliced = slice_compatible_state_tensors(target, source)

    assert torch.equal(sliced["conv.weight"], source["conv.weight"][:4, :3])
    assert torch.equal(sliced["bn.weight"], source["bn.weight"][:4])
    assert "wider.weight" not in sliced


def test_slice_compatible_state_tensors_center_crops_spatial_kernels():
    source = {"conv.weight": torch.arange(25).reshape(1, 1, 5, 5).float()}
    target = {"conv.weight": torch.zeros(1, 1, 3, 3)}

    sliced = slice_compatible_state_tensors(target, source)

    assert torch.equal(sliced["conv.weight"], source["conv.weight"][:, :, 1:4, 1:4])


def test_fold_c2f_output_projection_merges_the_removed_residual_group():
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1)
    target = torch.empty(2, 3, 1, 1)

    folded = fold_c2f_output_projection(target, source)

    assert torch.equal(folded[:, :2], source[:, :2])
    assert torch.equal(folded[:, 2], source[:, 2] + source[:, 3])


def test_pad_compatible_state_tensors_preserves_new_capacity_and_copies_source_prefix():
    source = {"weight": torch.ones(2, 3)}
    target = {"weight": torch.full((2, 5), 7.0)}

    padded = pad_compatible_state_tensors(target, source)

    assert torch.equal(padded["weight"][:, :3], source["weight"])
    assert torch.equal(padded["weight"][:, 3:], target["weight"][:, 3:])


def test_fasthead_transfer_loads_shared_graph_and_remaps_compatible_outputs():
    teacher = YOLOv10PoseModel(BASELINE_CFG, verbose=False)
    student = YOLOv10PoseModel(FAST_CFG, verbose=False)
    teacher_head = teacher.model[-1]
    student_head = student.model[-1]
    torch.nn.init.constant_(teacher_head.one2one_cv2[0][-1].bias, 0.375)

    report = transplant_compatible_weights(student, teacher)

    assert report["direct_tensor_count"] > 0
    assert report["sliced_tensor_count"] > 0
    assert report["remapped_tensor_count"] > 0
    assert torch.equal(student.model[0].conv.weight, teacher.model[0].conv.weight)
    assert torch.all(student_head.one2one_cv2[0][-1].bias == 0.375)
    assert student_head.one2one_cv4[0][-1].weight.shape != teacher_head.one2one_cv4[0][-1].weight.shape


def test_transferred_fasthead_has_finite_training_outputs():
    teacher = YOLOv10PoseModel(BASELINE_CFG, verbose=False)
    student = YOLOv10PoseModel(FAST_CFG, verbose=False)
    transplant_compatible_weights(student, teacher)

    outputs = student.train()(torch.rand(1, 3, 64, 64))

    assert all(
        torch.isfinite(tensor).all()
        for detections, keypoints in outputs.values()
        for tensor in [*detections, keypoints]
    )
