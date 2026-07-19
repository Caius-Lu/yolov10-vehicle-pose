#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"
NAME="${NAME:-overfit}"
MODEL_CONFIG="ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"
DATA_TEMPLATE="ultralytics/cfg/datasets/CarFusion-pose-overfit.yaml"
DATA_CONFIG="reports/training/runtime_configs/CarFusion-pose-overfit.yaml"
test -f datasets/CarFusion/yolo_pose/conversion_summary.json || { echo "BLOCKED: converted CarFusion data is missing"; exit 2; }
test -f datasets/CarFusion/yolo_pose/overfit.txt || { echo "BLOCKED: run scripts/create_carfusion_lists.py"; exit 2; }
.venv/bin/python scripts/materialize_dataset_config.py --template "$DATA_TEMPLATE" \
  --output "$DATA_CONFIG" --dataset-root datasets/CarFusion/yolo_pose
.venv/bin/python scripts/build_training_manifest.py --output "reports/training/manifests/${NAME}.json" \
  --model-config "$MODEL_CONFIG" --data-config "$DATA_CONFIG" --checkpoint weights/yolov10n.pt
.venv/bin/yolo pose train model="$MODEL_CONFIG" \
  pretrained=weights/yolov10n.pt data="$DATA_CONFIG" \
  epochs="${EPOCHS:-100}" batch="${BATCH:-8}" imgsz="${IMGSZ:-640}" device="${DEVICE:-0}" \
  seed="${SEED:-0}" deterministic=True amp=True fliplr=0.0 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 project=vehicle_pose_runs name="$NAME"
