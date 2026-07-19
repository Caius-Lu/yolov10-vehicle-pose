# 03 Planned Change List

## Core files

- `ultralytics/nn/modules/head.py`: add `v10Pose` and eval-only branch control.
- `ultralytics/nn/modules/__init__.py`: export the new head.
- `ultralytics/nn/tasks.py`: parse the new head, infer pose task, add `YOLOv10PoseModel`, connect loss.
- `ultralytics/utils/loss.py`: allow pose assigner top-k configuration and add dual `v10PoseLoss`.
- `ultralytics/utils/ops.py`: add vectorized NMS-free pose top-k postprocess.
- `ultralytics/models/yolov10/model.py`: add pose task mapping.
- `ultralytics/models/yolov10/train.py`: add pose trainer.
- `ultralytics/models/yolov10/predict.py`: add pose predictor.
- `ultralytics/models/yolov10/val.py`: add pose validator.

## Configuration and tools

- `ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml`
- `ultralytics/cfg/datasets/CarFusion-pose.yaml`
- `configs/vehicle_pose_loss.yaml`
- `configs/benchmark.yaml`
- Data conversion/quality scripts, training scripts, benchmarks, validation script, and reproduction documentation.

## Tests

- Build and task inference.
- P3/P4/P5 and 14×3 shapes at batch 1 and >1.
- Dual assignment outputs and finite forward/backward for target/empty batches.
- Visibility edge cases and GT-index matching.
- Original vs one2one-only equivalence.
- NMS-free keypoint gather.
- Checkpoint roundtrip, CPU/CUDA forward, ONNX smoke when available.
- Original YOLOv10 Detect regression.

