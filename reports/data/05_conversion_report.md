# 05 Conversion Report

Status: **IMPLEMENTATION READY / WAITING FOR VERIFIED REMOTE EXTRACTION**

The production converter is implemented at `scripts/convert_carfusion.py`. It validates COCO JSON, requires exactly 14×3 keypoints, maps COCO visibility 0/1/2 without inventing coordinates, invalidates out-of-image visible points with an issue record, validates images, preserves image/annotation IDs, and writes deterministic group-level train/val/test splits.

The user-supplied archive contains official-style COCO files with 20,187 train images / 40,642 annotations and 12,761 test images / 22,531 annotations. All 63,173 annotations contain exactly 14×3 keypoint values. The source also contains 739 explicit zero bboxes; conversion therefore requires the explicit `drop` policy and records every dropped source annotation instead of silently clipping it.

Ten pipeline tests pass. Combined branch coverage across conversion, manifest preparation, quality checking, and preview generation is 84%. The fixture contains generated 100×80 images solely to test parser behavior and is not training data. Real conversion waits only for verified remote extraction.

Reproducible command after authorization:

```bash
.venv/bin/python scripts/prepare_carfusion_coco.py \
  --train-json /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion/annotations/car_keypoints_train.json \
  --test-json /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion/annotations/car_keypoints_test.json \
  --output datasets/CarFusion/coco/car_keypoints_all.json
.venv/bin/python scripts/convert_carfusion.py \
  --coco-json datasets/CarFusion/coco/car_keypoints_all.json \
  --images-root /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion \
  --output datasets/CarFusion/yolo_pose --seed 0 \
  --invalid-bbox-policy drop --fixed-test-group car_penn
.venv/bin/python scripts/check_carfusion.py --root datasets/CarFusion/yolo_pose
.venv/bin/python scripts/visualize_carfusion.py \
  --root datasets/CarFusion/yolo_pose --output reports/data/preview --limit 100 --seed 0
```
