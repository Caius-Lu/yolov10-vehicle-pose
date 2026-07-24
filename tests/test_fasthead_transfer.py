from pathlib import Path

import torch

from ultralytics.models.yolov10.fasthead import transplant_compatible_weights
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"
FAST_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead-boxd1.yaml"


def test_fasthead_transfer_loads_shared_graph_and_remaps_compatible_outputs():
    teacher = YOLOv10PoseModel(BASELINE_CFG, verbose=False)
    student = YOLOv10PoseModel(FAST_CFG, verbose=False)
    teacher_head = teacher.model[-1]
    student_head = student.model[-1]
    torch.nn.init.constant_(teacher_head.one2one_cv2[0][-1].bias, 0.375)

    report = transplant_compatible_weights(student, teacher)

    assert report["direct_tensor_count"] > 0
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
