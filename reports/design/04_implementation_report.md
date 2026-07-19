# 04 Implementation Report

Status: **PASS_FOR_IMPLEMENTATION_AND_SMOKE**

Implemented a native `v10Pose` head for 14×3 vehicle keypoints with configurable 96-channel, depth-2 pure-convolution towers on P3/P4/P5. Training produces separate one-to-many and one-to-one detection/keypoint outputs. `v10PoseLoss` reuses the actual assigner `target_gt_idx` with top-k 10/1, so keypoints follow the matched vehicle GT.

The NMS-free postprocessor gathers keypoints using the same anchor/class top-k indices as boxes. The eval-only `one2one_only` flag skips unused one-to-many detection and pose towers. Dense one-to-one outputs match the original path exactly in the tested environment (max absolute error 0).

Official YOLOv10n Detect weights transfer 493/679 state entries into the vehicle-pose model. New pose parameters use initialized values. CUDA 640×640 forward passed on RTX 3090 Ti with finite `(1, 47, 8400)` dense output.

Real training is intentionally not part of this stage and remains blocked by CarFusion authorization.

