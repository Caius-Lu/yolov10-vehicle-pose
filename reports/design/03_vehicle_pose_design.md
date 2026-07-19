# 03 Vehicle Pose Design

## Decision

Implement `v10Pose` as a YOLOv10-native, NMS-free pose head. It preserves the upstream backbone/neck and detection towers and adds independent pure-convolution keypoint towers for one-to-many and one-to-one assignments.

## Head comparison

| Candidate | Parameters / FLOPs | Compatibility | Training and inference risk |
|---|---:|---|---|
| A: two 3×3 Conv + 1×1 | Highest of the three | Excellent | Stable, simple, but duplicates substantial work |
| B: Conv + DWConv + 1×1 | Lower | Good on PyTorch/ONNX; backend-dependent DW speed | Moderate |
| C: adapt existing Pose `cv4` to YOLOv10 | Known implementation, configurable width/depth | Excellent; Conv/BN/SiLU/reshape/concat only | Lowest integration risk |

Select C for version 1. Each branch uses configurable `pose_head_channels=96` and `pose_head_depth=2`. P3/P4/P5 are retained. The two keypoint towers are independent because the two assigners optimize different anchor sets; sharing remains a later controlled ablation.

## Training contract

- Head output per branch: `(raw_detection_features, raw_keypoints)`.
- `v10PoseLoss` owns two `v8PoseLoss` instances with `tal_topk=10` and `tal_topk=1`.
- Each loss uses its own assigner's `target_gt_idx` to gather the corresponding 14-point ground truth. No bbox-index heuristic is used.
- Total loss is the configured weighted sum of one-to-many and one-to-one loss vectors.

## Visibility and coordinate loss

- Input `v=0`: unlabelled/invalid; excluded from coordinate loss and visibility target is 0.
- Input `v=1`: labelled but occluded; included in coordinate loss and visibility target is 1.
- Input `v=2`: visible; included in coordinate loss and visibility target is 1.
- Version 1 uses a binary visibility logit for deployability and compatibility with the existing pose pipeline. It does not distinguish occluded vs visible at inference.
- Coordinates use the existing OKS-like loss normalized by matched bbox area. Because no authoritative vehicle sigmas are available, 14 uniform configurable sigmas are treated as an engineering loss/experimental metric, not standard COCO vehicle OKS AP.
- Existing `box`, `cls`, `dfl`, `pose`, and `kobj` gains remain in configuration. New `one2many_loss_weight` and `one2one_loss_weight` live in the model YAML.

## Inference contract

- Decode box and 14×3 keypoints at all fixed-shape anchors without Python loops over targets or keypoints.
- Apply YOLOv10 two-stage top-k selection to class scores and gather box/keypoint tensors using the same anchor indices.
- Default prediction is one-to-one/NMS-free.
- `one2one_only=False` reproduces the upstream-style PyTorch path for correctness and timing comparison.
- `one2one_only=True` skips unused one-to-many detection and pose towers in eval mode; one-to-one tensors must match within `atol=1e-6, rtol=1e-5`.

## Weight initialization

Build the vehicle-pose YAML, load `weights/yolov10n.pt` through the existing intersect-by-name-and-shape loader, report transferred/missing layers, and retain normal initialization for new keypoint towers. No silent custom `strict=False` load is introduced.

## Benchmark naming

- `torch_original`: eval with `one2one_only=False`.
- `torch_one2one_only`: eval with unused one-to-many towers skipped.
- `forward_only`: raw tower execution.
- `forward_decode`: model plus dense box/keypoint decode.
- `end_to_end`: file read through D2H/result construction.

