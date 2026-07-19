# 06 CarFusion Baseline Report

Status: **PASS WITH DOCUMENTED DATA LIMITATION**

## Run

- Model: YOLOv10n Vehicle Pose, single `vehicle` class, `kpt_shape=[14,3]`.
- Hardware: RTX 3090 Ti 24 GB under WSL2.
- Training: 200 epochs, batch 16, image size 640, AMP, 8 workers, deterministic seed 0.
- Dataset: 20,187 official-train images converted with group-aware train/val/test lists; final validation split contains 3,606 images and 11,569 instances.
- Checkpoint: `vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt`.
- Resume: WSL reboot interrupted epoch 176; training resumed from `last.pt` and completed epochs 176–200 without restarting the experiment.

## Final best-checkpoint validation

| Metric | Value |
|---|---:|
| Box precision / recall | 0.840 / 0.711 |
| Box mAP50 / mAP50-95 | 0.818 / 0.665 |
| Pose precision / recall | 0.822 / 0.683 |
| Pose mAP50 / mAP50-95 | 0.765 / 0.515 |
| PCK@0.05 / PCK@0.10 | 0.6686 / 0.8525 |
| NME | 0.05960 |
| Visible / occluded PCK@0.05 | 0.6875 / 0.5470 |

PCK/NME use the matched GT bounding-box long side for normalization. Raw evidence is in `reports/evaluation/carfusion_baseline_best_metrics.json` and the training curves are in `reports/training/final_assets/`.

## Time evidence

- The uninterrupted resume segment completed 25 epochs in 1.230 hours, or about 2.95 minutes per epoch.
- At that observed rate, 200 epochs represent about 9.84 GPU-hours of active training. This is an extrapolated compute estimate, not the end-to-end wall-clock duration.
- The actual run crossed a WSL reboot and temporary Windows E-drive mount loss, so wall-clock completion was substantially longer. Checkpoint resume prevented loss of the first 175 completed epochs.

## Visual gate and predictor regression

Validator plots showed aligned boxes and keypoints, but the first raw-image predictor review produced oversized boxes on 16:9 images. A test-first diagnosis showed that `YOLOv10PosePredictor.postprocess` passed `xywh` boxes into `scale_boxes` as though they were `xyxy`. The predictor now performs `xywh2xyxy` before scaling. This changes predictor visualization/end-to-end output only; it does not change training or validation metrics.

After the fix:

- `tests/test_vehicle_pose.py`: 9/9 passed;
- 100 fixed raw-image predictions were regenerated;
- representative wide, edge-clipped, occluded and multi-vehicle scenes were inspected;
- boxes and keypoints were consistent with the final metrics.

## Data limitation

The model output is 14×3, but source-derived quality statistics show keypoint index 13 has visibility 0 for all 62,434 converted instances. Final validation therefore has 0 valid observations for that point and cannot report its PCK/NME. The defensible description is **14-point tensor output with 13 supervised/evaluable points in this CarFusion release**. This should be resolved at the dataset-contract level before claiming complete 14-point accuracy.
