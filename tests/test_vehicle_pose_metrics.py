import math

import pytest
import torch

from scripts.evaluate_vehicle_pose import compute_metrics
from ultralytics.utils.vehicle_pose_metrics import match_pose_pairs


def test_vehicle_pose_metrics_separate_visible_occluded_and_size_groups():
    keypoints = [[10.0, 10.0, 2.0], [20.0, 20.0, 1.0]] + [[0.0, 0.0, 0.0]] * 12
    predictions = [[11.0, 10.0, 0.9], [24.0, 20.0, 0.8]] + [[0.0, 0.0, 0.0]] * 12
    pairs = [{"bbox": [0.0, 0.0, 100.0, 50.0], "gt_keypoints": keypoints, "pred_keypoints": predictions}]
    metrics = compute_metrics(pairs)
    assert math.isclose(metrics["nme"], 0.025)
    assert metrics["pck@0.05"] == 1.0
    assert metrics["pck@0.10"] == 1.0
    assert metrics["visible_pck@0.05"] == 1.0
    assert metrics["occluded_pck@0.05"] == 1.0
    assert metrics["size_groups"]["medium"]["count"] == 2
    assert metrics["valid_keypoint_count_groups"]["2"]["count"] == 2
    assert metrics["per_keypoint"][0]["count"] == 1
    assert metrics["per_keypoint"][2]["count"] == 0


def test_vehicle_pose_metrics_reject_invalid_pairs_and_handles_no_valid_points():
    empty = compute_metrics(
        [{"bbox": [0, 0, 10, 10], "gt_keypoints": [[0, 0, 0]] * 14, "pred_keypoints": [[0, 0, 0]] * 14}]
    )
    assert empty["valid_keypoints"] == 0
    assert empty["nme"] is None
    with pytest.raises(ValueError, match="bbox scale"):
        compute_metrics([{"bbox": [0, 0, 0, 0], "gt_keypoints": [], "pred_keypoints": []}])
    with pytest.raises(ValueError, match="exactly 14"):
        compute_metrics([{"bbox": [0, 0, 10, 10], "gt_keypoints": [], "pred_keypoints": []}])


def test_pose_pair_matching_is_greedy_unique_and_iou_gated():
    gt_boxes = torch.tensor([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=torch.float32)
    pred_boxes = torch.tensor([[2, 2, 98, 98], [205, 205, 295, 295], [400, 400, 450, 450]], dtype=torch.float32)
    gt_keypoints = torch.zeros(2, 14, 3)
    pred_keypoints = torch.ones(3, 14, 3)
    gt_keypoints[..., 2] = 2
    pairs = match_pose_pairs(
        gt_boxes,
        gt_keypoints,
        pred_boxes,
        pred_keypoints,
        pred_scores=torch.tensor([0.8, 0.7, 0.99]),
        iou_threshold=0.5,
    )
    assert len(pairs) == 2
    assert pairs[0]["gt_index"] == 0
    assert pairs[0]["pred_index"] == 0
    assert pairs[0]["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert len(pairs[0]["gt_keypoints"]) == 14
