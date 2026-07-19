"""Behavior tests for the YOLOv10 vehicle-pose extension.

User journey: as a training engineer I can build, train, save, and run the
14-keypoint model without changing the existing YOLOv10 detection model.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics.models.yolov10.predict import YOLOv10PosePredictor
from ultralytics.nn.modules.head import v10Pose
from ultralytics.nn.tasks import YOLOv10DetectionModel, YOLOv10PoseModel, guess_model_task
from ultralytics.utils.ops import v10_pose_postprocess


ROOT = Path(__file__).resolve().parents[1]
POSE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"
DETECT_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n.yaml"
OFFICIAL_WEIGHTS = ROOT / "weights/yolov10n.pt"


def _pose_model():
    return YOLOv10PoseModel(POSE_CFG, verbose=False)


def _training_batch(batch_size=2, with_target=True):
    batch = {"img": torch.rand(batch_size, 3, 64, 64)}
    if with_target:
        keypoints = torch.zeros(1, 14, 3)
        keypoints[..., :2] = 0.5
        keypoints[..., 2] = 2
        batch.update(
            batch_idx=torch.tensor([0.0]),
            cls=torch.tensor([[0.0]]),
            bboxes=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            keypoints=keypoints,
        )
    else:
        batch.update(
            batch_idx=torch.empty(0),
            cls=torch.empty(0, 1),
            bboxes=torch.empty(0, 4),
            keypoints=torch.empty(0, 14, 3),
        )
    return batch


def test_pose_yaml_selects_pose_task_and_builds_expected_head():
    assert guess_model_task(POSE_CFG) == "pose"
    model = _pose_model()
    head = model.model[-1]
    assert isinstance(head, v10Pose)
    assert head.kpt_shape == [14, 3]
    assert head.nk == 42
    assert len(head.cv4) == len(head.one2one_cv4) == 3


def test_training_forward_has_both_detection_and_keypoint_branches():
    model = _pose_model().train()
    output = model(torch.rand(2, 3, 64, 64))
    assert set(output) == {"one2many", "one2one"}
    for branch in output.values():
        features, keypoints = branch
        assert len(features) == 3
        assert keypoints.shape == (2, 42, 84)
        assert all(torch.isfinite(x).all() for x in features)
        assert torch.isfinite(keypoints).all()


def test_one2one_only_matches_original_one2one_dense_output():
    model = _pose_model().eval()
    head = model.model[-1]
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        head.one2one_only = False
        original = model(image)
        head.one2one_only = True
        optimized = model(image)
    assert original["one2many"] is not None
    assert optimized["one2many"] is None
    assert torch.allclose(original["one2one"][0], optimized["one2one"][0], atol=1e-6, rtol=1e-5)


def test_pose_postprocess_keeps_keypoints_from_selected_anchor():
    # Dense layout: xywh, one class score, then two keypoints x/y/v.
    dense = torch.tensor(
        [
            [
                [10.0, 20.0, 4.0, 8.0, 0.10, 1.0, 2.0, 0.1, 3.0, 4.0, 0.2],
                [30.0, 40.0, 6.0, 10.0, 0.90, 5.0, 6.0, 0.7, 7.0, 8.0, 0.8],
            ]
        ]
    )
    result = v10_pose_postprocess(dense, max_det=1, nc=1, kpt_shape=(2, 3))
    assert result.shape == (1, 1, 12)
    assert torch.equal(result[0, 0, :4], dense[0, 1, :4])
    assert result[0, 0, 4].item() == dense[0, 1, 4].item()
    assert result[0, 0, 5].item() == 0
    assert torch.equal(result[0, 0, 6:], dense[0, 1, 5:])


def test_pose_predictor_converts_xywh_boxes_before_scaling(monkeypatch):
    """Raw-image prediction must scale xyxy boxes, matching the validator path."""
    predictor = YOLOv10PosePredictor.__new__(YOLOv10PosePredictor)
    predictor.args = SimpleNamespace(conf=0.25, classes=None, max_det=1)
    predictor.model = SimpleNamespace(kpt_shape=(2, 3), names={0: "vehicle"})
    predictor.batch = (["frame.jpg"],)
    captured = {}

    def capture_boxes(_img_shape, boxes, _orig_shape):
        captured["boxes"] = boxes.clone()
        return boxes

    monkeypatch.setattr("ultralytics.models.yolov10.predict.ops.scale_boxes", capture_boxes)
    monkeypatch.setattr(
        "ultralytics.models.yolov10.predict.ops.scale_coords", lambda _img_shape, coords, _orig_shape: coords
    )
    prediction = torch.tensor([[[10.0, 20.0, 4.0, 6.0, 0.9, 0.0, 1.0, 2.0, 0.8, 3.0, 4.0, 0.7]]])

    predictor.postprocess(prediction, torch.zeros(1, 3, 64, 64), [np.zeros((64, 64, 3), dtype=np.uint8)])

    assert torch.equal(captured["boxes"][0], torch.tensor([8.0, 17.0, 12.0, 23.0]))


def test_pose_loss_backward_is_finite_for_target_and_empty_batch():
    for with_target in (True, False):
        model = _pose_model().train()
        batch = _training_batch(with_target=with_target)
        loss, items = model.loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert items.shape == (10,)
        assert torch.isfinite(items).all()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_checkpoint_roundtrip_preserves_pose_output(tmp_path):
    model = _pose_model().eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = model(image)["one2one"][0]
    checkpoint = tmp_path / "pose_state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _pose_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        actual = restored(image)["one2one"][0]
    assert torch.equal(expected, actual)


def test_original_yolov10_detect_model_still_builds_and_runs():
    model = YOLOv10DetectionModel(DETECT_CFG, verbose=False).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))
    assert set(output) == {"one2many", "one2one"}
    assert output["one2one"][0].shape[1] == 84


def test_official_checkpoint_loads_with_current_pytorch():
    if not OFFICIAL_WEIGHTS.is_file():
        return
    from ultralytics import YOLOv10

    model = YOLOv10(OFFICIAL_WEIGHTS)
    assert isinstance(model.model, YOLOv10DetectionModel)
