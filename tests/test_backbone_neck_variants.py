"""Architecture contracts for latency-gated PSA/C2f backbone-neck variants."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.block import C2f, C2fCIB, PSA
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics/cfg/models/v10"
BASELINE = MODEL_DIR / "yolov10n-vehicle-pose.yaml"
PSA025 = MODEL_DIR / "yolov10n-vehicle-pose-psa025.yaml"
CIB_P5 = MODEL_DIR / "yolov10n-vehicle-pose-c2fcib-p5.yaml"
CIB_P45 = MODEL_DIR / "yolov10n-vehicle-pose-c2fcib-p45.yaml"
COMBINED = MODEL_DIR / "yolov10n-vehicle-pose-psa025-c2fcib-p5.yaml"


def build(path):
    return YOLOv10PoseModel(path, verbose=False)


def test_psa025_halves_attention_channels_without_narrowing_pose_head():
    baseline = build(BASELINE)
    candidate = build(PSA025)

    assert isinstance(baseline.model[10], PSA)
    assert isinstance(candidate.model[10], PSA)
    assert candidate.model[10].c * 2 == baseline.model[10].c
    assert candidate.model[-1].pose_head_channels == 96
    assert candidate.model[-1].pose_head_depth == 2
    assert candidate.model[-1].box_head_depth == 2


@pytest.mark.parametrize(
    ("path", "cib_indices", "c2f_indices"),
    [
        (CIB_P5, (8,), (6,)),
        (CIB_P45, (6, 8), (4,)),
        (COMBINED, (8,), (6,)),
    ],
)
def test_c2fcib_variants_only_replace_declared_backbone_stages(path, cib_indices, c2f_indices):
    candidate = build(path)

    for index in cib_indices:
        assert isinstance(candidate.model[index], C2fCIB)
    for index in c2f_indices:
        assert isinstance(candidate.model[index], C2f)
        assert not isinstance(candidate.model[index], C2fCIB)
    assert candidate.model[-1].pose_head_channels == 96
    assert candidate.model[-1].pose_head_depth == 2
    assert candidate.model[-1].box_head_depth == 2


@pytest.mark.parametrize("path", [PSA025, CIB_P5, CIB_P45, COMBINED])
def test_backbone_neck_variant_preserves_dual_assignment_pose_outputs(path):
    model = build(path).train()
    image = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(image)

    assert isinstance(output, dict)
    assert set(output) == {"one2many", "one2one"}
    for branch in output.values():
        assert len(branch) == 2
        features, keypoints = branch
        assert len(features) == 3
        assert keypoints.shape == (2, 14 * 3, 84)
        assert all(torch.isfinite(feature).all() for feature in features)
        assert torch.isfinite(keypoints).all()
