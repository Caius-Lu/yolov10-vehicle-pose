"""TDD contracts for the second PSA/C2f latency-search round."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.block import C2f, LitePSA
from ultralytics.models.yolov10.fasthead import transplant_compatible_weights
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics/cfg/models/v10"
LITE_PSA = MODEL_DIR / "yolov10n-vehicle-pose-litepsa.yaml"
C2F_P4D1 = MODEL_DIR / "yolov10n-vehicle-pose-c2f-p4d1.yaml"
COMBINED = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1.yaml"


def build(path):
    return YOLOv10PoseModel(path, verbose=False)


def test_lite_psa_preserves_shape_and_finite_residual_path():
    block = LitePSA(256, 256, e=0.5, kernel_size=5).eval()
    image = torch.randn(2, 256, 20, 20)

    with torch.inference_mode():
        output = block(image)

    assert output.shape == image.shape
    assert block.c == 128
    assert block.spatial.conv.groups == 128
    assert block.spatial.conv.kernel_size == (5, 5)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("path", [LITE_PSA, COMBINED])
def test_lite_psa_yaml_uses_lite_block_and_preserves_full_pose_head(path):
    model = build(path)

    assert isinstance(model.model[10], LitePSA)
    assert model.model[-1].pose_head_channels == 96
    assert model.model[-1].pose_head_depth == 2
    assert model.model[-1].box_head_depth == 2


@pytest.mark.parametrize("path", [C2F_P4D1, COMBINED])
def test_p4_depth_candidate_removes_exactly_one_scaled_bottleneck(path):
    baseline = build(MODEL_DIR / "yolov10n-vehicle-pose.yaml")
    candidate = build(path)

    assert isinstance(candidate.model[6], C2f)
    assert len(baseline.model[6].m) == 2
    assert len(candidate.model[6].m) == 1
    assert len(candidate.model[4].m) == len(baseline.model[4].m)
    assert candidate.model[-1].pose_head_channels == 96


@pytest.mark.parametrize("path", [LITE_PSA, C2F_P4D1, COMBINED])
def test_second_round_variants_keep_training_pose_contract(path):
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


def test_combined_candidate_transfers_the_complete_pose_head_from_baseline():
    teacher = build(MODEL_DIR / "yolov10n-vehicle-pose.yaml")
    student = build(COMBINED)

    report = transplant_compatible_weights(student, teacher)

    teacher_head = teacher.model[-1].state_dict()
    student_head = student.model[-1].state_dict()
    assert teacher_head.keys() == student_head.keys()
    assert all(torch.equal(student_head[name], teacher_head[name]) for name in teacher_head)
    assert report["direct_tensor_count"] > 250
    assert report["student_parameters"] < report["teacher_parameters"]
