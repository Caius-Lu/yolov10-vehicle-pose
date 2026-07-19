# 03 Repository Audit

## Scope and revision

- Upstream: `https://github.com/THU-MIG/yolov10.git`
- Revision: `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`
- Audit date: 2026-07-16

## YAML-to-module construction path

1. `ultralytics.engine.model.Model._new()` loads YAML with `yaml_model_load()` and infers the task with `guess_model_task()`.
2. The task map selects the PyTorch model class.
3. `DetectionModel.__init__()` calls `parse_model()`.
4. `parse_model()` resolves layer names imported from `ultralytics.nn.modules`, scales depth/width, appends feature-channel lists to head arguments, instantiates modules, and records their routing metadata.
5. A dummy 256×256 forward establishes P3/P4/P5 strides, then head biases are initialized.

## YOLOv10 dual assignments and paths

- `v10Detect` owns regular `cv2/cv3` one-to-many branches and deep-copied `one2one_cv2/one2one_cv3` branches.
- The one-to-one branch consumes detached neck features; the one-to-many branch consumes normal features.
- Training returns a dictionary containing raw `one2many` and `one2one` feature maps.
- `v10DetectLoss` applies `v8DetectionLoss(tal_topk=10)` and `v8DetectionLoss(tal_topk=1)` and sums both losses.
- Evaluation decodes the one-to-one branch, but the current non-export PyTorch path also computes the one-to-many branch before returning it. Export skips that branch.
- YOLOv10 prediction/validation uses NMS-free `ops.v10postprocess()` over the one-to-one dense output.

## Existing pose capability

The repository already has:

- `Pose` pure-convolution head with a per-scale `cv4` keypoint branch.
- `v8PoseLoss`, including assigner-provided `target_gt_idx`; keypoints are gathered from the same ground-truth instance as each matched box.
- `PoseModel`, `PoseTrainer`, `PoseValidator`, and `PosePredictor`.
- Dataset parsing, 2D/3D keypoint validation, augmentation hooks, plotting, and COCO-style pose metrics.

It does not have a YOLOv10 pose head, dual pose loss, YOLOv10 pose model/task map, or NMS-free keypoint postprocessing.

## Reusable modules and gaps

- Reuse `Pose.cv4` structure and vectorized keypoint decode semantics.
- Reuse `v8PoseLoss` for each assignment branch by making `tal_topk` configurable.
- Reuse generic pose data loading, plotting, and most validator behavior.
- Add a YOLOv10-specific pose head and NMS-free top-k gather so keypoints remain attached to the selected anchor.
- Add an explicit `one2one_only` flag that skips one-to-many detection and keypoint convolution only in eval mode.

## Performance finding

In `ultralytics/nn/modules/head.py`, `v10Detect.forward()` invokes `super().forward(x)` whenever `export` is false, including `eval()`. This executes the one-to-many `cv2/cv3` towers even though predictor and validator discard that result. The vehicle-pose head will expose the original path for comparison and an inference-only path that returns identical one-to-one dense output while setting `one2many` to `None`.

