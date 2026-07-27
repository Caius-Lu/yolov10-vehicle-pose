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
COMBINED_BOXD1 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-boxd1.yaml"
COMBINED_POSED1 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-posed1.yaml"
COMBINED_P3P4D1 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p3p4d1-head96.yaml"
COMBINED_POSE_CLASS_D1 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-posed1-classd1.yaml"
COMBINED_CLASS_D1 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-classd1.yaml"
COMBINED_CLASS_D1W96 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-classd1w96.yaml"
COMBINED_CLASS_LITE = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-classlite.yaml"
COMBINED_CLASS_LITE56 = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-head96-classlite56.yaml"
COMBINED_P5_ZERO = MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-p5zero-head96.yaml"
COMBINED_P4_NECK_P5_ZERO = (
    MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-p4neck0-p5zero-head96.yaml"
)
COMBINED_P4_NECK_P5_ZERO_K3 = (
    MODEL_DIR / "yolov10n-vehicle-pose-litepsa-k3-c2f-p4d1-p4neck0-p5zero-head96.yaml"
)
COMBINED_P4_BOTH_P5_ZERO = (
    MODEL_DIR / "yolov10n-vehicle-pose-litepsa-c2f-p4d1-p4both0-p5zero-head96.yaml"
)


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


def test_combined_boxd1_candidate_only_shortens_box_towers():
    baseline = build(COMBINED)
    candidate = build(COMBINED_BOXD1)
    head = candidate.model[-1]

    assert isinstance(candidate.model[10], LitePSA)
    assert len(candidate.model[6].m) == 1
    assert head.pose_head_channels == 96
    assert head.pose_head_depth == 2
    assert head.box_head_depth == 1
    assert sum(parameter.numel() for parameter in candidate.parameters()) < sum(
        parameter.numel() for parameter in baseline.parameters()
    )


def test_combined_posed1_candidate_preserves_detection_and_only_shortens_pose_towers():
    baseline = build(COMBINED)
    candidate = build(COMBINED_POSED1)
    head = candidate.model[-1]

    assert head.pose_head_channels == 96
    assert head.pose_head_depth == 1
    assert head.box_head_depth == 2
    assert len(head.cv2[0]) == len(baseline.model[-1].cv2[0])
    assert len(head.one2one_cv2[0]) == len(baseline.model[-1].one2one_cv2[0])
    assert len(head.cv4[0]) + 1 == len(baseline.model[-1].cv4[0])
    assert sum(parameter.numel() for parameter in candidate.parameters()) < sum(
        parameter.numel() for parameter in baseline.parameters()
    )


def test_combined_p3p4_depth_candidate_removes_one_more_backbone_bottleneck():
    baseline = build(COMBINED)
    candidate = build(COMBINED_P3P4D1)

    assert len(candidate.model[4].m) + 1 == len(baseline.model[4].m)
    assert len(candidate.model[6].m) == len(baseline.model[6].m)
    assert candidate.model[-1].pose_head_depth == 2
    assert candidate.model[-1].box_head_depth == 2


def test_pose_and_class_depth_candidate_preserves_complete_box_towers():
    baseline = build(COMBINED)
    candidate = build(COMBINED_POSE_CLASS_D1)
    head = candidate.model[-1]

    assert head.pose_head_depth == 1
    assert head.class_head_depth == 1
    assert head.box_head_depth == 2
    assert len(head.cv2[0]) == len(baseline.model[-1].cv2[0])
    assert len(head.one2one_cv2[0]) == len(baseline.model[-1].one2one_cv2[0])
    assert len(head.cv3[0]) + 1 == len(baseline.model[-1].cv3[0])
    assert len(head.cv4[0]) + 1 == len(baseline.model[-1].cv4[0])


def test_class_depth_candidate_preserves_complete_box_and_pose_towers():
    baseline = build(COMBINED)
    candidate = build(COMBINED_CLASS_D1)
    head = candidate.model[-1]

    assert head.pose_head_depth == 2
    assert head.class_head_depth == 1
    assert head.box_head_depth == 2
    assert len(head.cv2[0]) == len(baseline.model[-1].cv2[0])
    assert len(head.cv4[0]) == len(baseline.model[-1].cv4[0])
    assert len(head.one2one_cv4[0]) == len(baseline.model[-1].one2one_cv4[0])
    assert len(head.cv3[0]) + 1 == len(baseline.model[-1].cv3[0])


def test_wide_class_depth_candidate_uses_96_hidden_channels():
    candidate = build(COMBINED_CLASS_D1W96)
    head = candidate.model[-1]

    assert head.pose_head_depth == 2
    assert head.class_head_depth == 1
    assert head.class_head_channels == 96
    assert head.cv3[0][0][1].conv.out_channels == 96
    assert head.one2one_cv3[0][0][1].conv.out_channels == 96


def test_class_lite_candidate_replaces_second_depthwise_block_with_pointwise():
    baseline = build(COMBINED)
    candidate = build(COMBINED_CLASS_LITE)
    head = candidate.model[-1]
    torch.nn.init.constant_(baseline.model[-1].one2one_cv3[0][1][1].conv.weight, 0.25)

    transplant_compatible_weights(candidate, baseline)

    assert head.pose_head_depth == 2
    assert head.box_head_depth == 2
    assert head.class_head_depth == 3
    assert len(head.cv3[0]) == len(baseline.model[-1].cv3[0])
    assert head.cv3[0][1].conv.kernel_size == (1, 1)
    assert head.cv3[0][1].conv.groups == 1
    assert head.cv3[0][1].conv.out_channels == 48
    assert torch.all(head.one2one_cv3[0][1].conv.weight == 0.25)


def test_class_lite_width_can_be_set_independently():
    head = build(COMBINED_CLASS_LITE56).model[-1]

    assert head.class_head_channels == 64
    assert head.class_lite_channels == 56
    assert head.cv3[0][1].conv.out_channels == 56
    assert head.one2one_cv3[0][1].conv.out_channels == 56


def test_p5_zero_candidate_only_removes_low_resolution_bottlenecks():
    """Protect P3/P4 and every prediction tower while reducing both P5 stages."""

    baseline = build(COMBINED)
    candidate = build(COMBINED_P5_ZERO)

    # Backbone P5 and neck P5 lose one internal residual/CIB block.
    assert len(candidate.model[8].m) + 1 == len(baseline.model[8].m)
    assert len(candidate.model[22].m) + 1 == len(baseline.model[22].m)

    # Higher-resolution P3/P4 paths and the complete multi-task head stay intact.
    for layer_index in (4, 6, 13, 16, 19):
        assert len(candidate.model[layer_index].m) == len(baseline.model[layer_index].m)
    assert candidate.model[-1].box_head_depth == 2
    assert candidate.model[-1].class_head_depth == 2
    assert candidate.model[-1].pose_head_depth == 2


def test_p4_neck_p5_zero_candidate_preserves_backbone_p3_p4_and_full_head():
    """Add one neck-P4 removal without touching backbone detail or task towers."""

    baseline = build(COMBINED)
    candidate = build(COMBINED_P4_NECK_P5_ZERO)

    for layer_index in (8, 19, 22):
        assert len(candidate.model[layer_index].m) + 1 == len(baseline.model[layer_index].m)
    for layer_index in (4, 6, 13, 16):
        assert len(candidate.model[layer_index].m) == len(baseline.model[layer_index].m)
    assert candidate.model[-1].box_head_depth == 2
    assert candidate.model[-1].class_head_depth == 2
    assert candidate.model[-1].pose_head_depth == 2


def test_p4_neck_p5_zero_k3_only_reduces_litepsa_spatial_kernel():
    previous = build(COMBINED_P4_NECK_P5_ZERO)
    candidate = build(COMBINED_P4_NECK_P5_ZERO_K3)

    assert previous.model[10].spatial.conv.kernel_size == (5, 5)
    assert candidate.model[10].spatial.conv.kernel_size == (3, 3)
    assert candidate.model[10].spatial.conv.groups == previous.model[10].spatial.conv.groups
    for layer_index in (4, 6, 8, 13, 16, 19, 22):
        assert len(candidate.model[layer_index].m) == len(previous.model[layer_index].m)


def test_p4_both_p5_zero_preserves_all_p3_paths_and_prediction_towers():
    baseline = build(COMBINED)
    candidate = build(COMBINED_P4_BOTH_P5_ZERO)

    for layer_index in (8, 13, 19, 22):
        assert len(candidate.model[layer_index].m) + 1 == len(baseline.model[layer_index].m)
    for layer_index in (4, 6, 16):
        assert len(candidate.model[layer_index].m) == len(baseline.model[layer_index].m)
    assert candidate.model[-1].box_head_depth == 2
    assert candidate.model[-1].class_head_depth == 2
    assert candidate.model[-1].pose_head_depth == 2
