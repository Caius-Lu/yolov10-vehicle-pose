#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCHIVE="${CARFUSION_ARCHIVE:-/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/CarFusion.tar.gz.00}"
EXTRACTED="${CARFUSION_EXTRACTED:-/home/caius/projects/yolopose/data/CarFusion/extracted}"
SOURCE="$EXTRACTED/carfusion"
OUTPUT="datasets/CarFusion/yolo_pose"

scripts/extract_carfusion_remote.sh "$ARCHIVE" "$EXTRACTED"

.venv/bin/python scripts/prepare_carfusion_coco.py \
  --train-json "$SOURCE/annotations/car_keypoints_train.json" \
  --test-json "$SOURCE/annotations/car_keypoints_test.json" \
  --output datasets/CarFusion/coco/car_keypoints_all.json

.venv/bin/python scripts/convert_carfusion.py \
  --coco-json datasets/CarFusion/coco/car_keypoints_all.json \
  --images-root "$SOURCE" --output "$OUTPUT" --seed 0 \
  --invalid-bbox-policy drop --fixed-test-group car_penn --skip-image-verify

.venv/bin/python scripts/check_carfusion.py --root "$OUTPUT"
.venv/bin/python scripts/visualize_carfusion.py \
  --root "$OUTPUT" --output reports/data/preview --limit 100 --seed 0
.venv/bin/python scripts/create_carfusion_lists.py \
  --root "$OUTPUT" --fixed-eval-output reports/training/fixed_eval_images.txt --seed 0

echo "PASS CarFusion extraction, conversion, quality gate, previews and fixed lists"
