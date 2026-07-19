"""Vehicle-specific pose pairing plus PCK/NME metrics."""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from ultralytics.utils.metrics import box_iou


def _aggregate(errors, threshold):
    return None if not errors else sum(error <= threshold for error in errors) / len(errors)


def compute_vehicle_pose_metrics(pairs):
    """Compute metrics normalized by the GT vehicle bbox long side."""
    all_errors, visible_errors, occluded_errors = [], [], []
    per_keypoint = defaultdict(list)
    size_errors = defaultdict(list)
    valid_count_errors = defaultdict(list)
    for pair in pairs:
        _, _, width, height = map(float, pair["bbox"])
        scale = max(width, height)
        if scale <= 0:
            raise ValueError("bbox scale must be positive")
        area = width * height
        size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
        ground_truth = pair["gt_keypoints"]
        prediction = pair["pred_keypoints"]
        if len(ground_truth) != 14 or len(prediction) != 14:
            raise ValueError("each pair must contain exactly 14 keypoints")
        pair_errors = []
        valid_count = sum(int(target[2]) in (1, 2) for target in ground_truth)
        for index, (target, estimate) in enumerate(zip(ground_truth, prediction)):
            visibility = int(target[2])
            if visibility == 0:
                continue
            if visibility not in (1, 2):
                raise ValueError(f"invalid visibility {visibility}")
            error = math.hypot(float(estimate[0]) - float(target[0]), float(estimate[1]) - float(target[1])) / scale
            all_errors.append(error)
            pair_errors.append(error)
            per_keypoint[index].append(error)
            size_errors[size].append(error)
            (visible_errors if visibility == 2 else occluded_errors).append(error)
        valid_count_errors[str(valid_count)].extend(pair_errors)

    per_keypoint_metrics = []
    for index in range(14):
        errors = per_keypoint[index]
        per_keypoint_metrics.append(
            {
                "index": index,
                "count": len(errors),
                "pck@0.05": _aggregate(errors, 0.05),
                "pck@0.10": _aggregate(errors, 0.10),
                "nme": None if not errors else sum(errors) / len(errors),
            }
        )
    size_groups = {
        name: {
            "count": len(size_errors[name]),
            "pck@0.05": _aggregate(size_errors[name], 0.05),
            "pck@0.10": _aggregate(size_errors[name], 0.10),
            "nme": None if not size_errors[name] else sum(size_errors[name]) / len(size_errors[name]),
        }
        for name in ("small", "medium", "large")
    }
    valid_keypoint_count_groups = {
        name: {
            "count": len(errors),
            "pck@0.05": _aggregate(errors, 0.05),
            "pck@0.10": _aggregate(errors, 0.10),
            "nme": None if not errors else sum(errors) / len(errors),
        }
        for name, errors in sorted(valid_count_errors.items(), key=lambda item: int(item[0]))
    }
    return {
        "normalization": "GT bbox long side",
        "valid_keypoints": len(all_errors),
        "pck@0.05": _aggregate(all_errors, 0.05),
        "pck@0.10": _aggregate(all_errors, 0.10),
        "nme": None if not all_errors else sum(all_errors) / len(all_errors),
        "visible_pck@0.05": _aggregate(visible_errors, 0.05),
        "visible_pck@0.10": _aggregate(visible_errors, 0.10),
        "occluded_pck@0.05": _aggregate(occluded_errors, 0.05),
        "occluded_pck@0.10": _aggregate(occluded_errors, 0.10),
        "per_keypoint": per_keypoint_metrics,
        "size_groups": size_groups,
        "valid_keypoint_count_groups": valid_keypoint_count_groups,
    }


def match_pose_pairs(
    gt_boxes, gt_keypoints, pred_boxes, pred_keypoints, pred_scores=None, iou_threshold=0.5
):
    """Greedily create unique GT/prediction pairs ordered by box IoU then confidence."""
    gt_boxes = torch.as_tensor(gt_boxes).detach().float().cpu()
    pred_boxes = torch.as_tensor(pred_boxes).detach().float().cpu()
    gt_keypoints = torch.as_tensor(gt_keypoints).detach().float().cpu()
    pred_keypoints = torch.as_tensor(pred_keypoints).detach().float().cpu()
    pred_scores = (
        torch.ones(len(pred_boxes), dtype=torch.float32)
        if pred_scores is None
        else torch.as_tensor(pred_scores).detach().float().cpu()
    )
    if not len(gt_boxes) or not len(pred_boxes):
        return []
    overlaps = box_iou(gt_boxes, pred_boxes)
    candidates = []
    for gt_index, pred_index in torch.nonzero(overlaps >= iou_threshold, as_tuple=False).tolist():
        candidates.append(
            (float(overlaps[gt_index, pred_index]), float(pred_scores[pred_index]), gt_index, pred_index)
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_gt, used_pred, pairs = set(), set(), []
    for overlap, score, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        x1, y1, x2, y2 = gt_boxes[gt_index].tolist()
        pairs.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "bbox_iou": overlap,
                "score": score,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "gt_keypoints": gt_keypoints[gt_index].tolist(),
                "pred_keypoints": pred_keypoints[pred_index].tolist(),
            }
        )
    return pairs
