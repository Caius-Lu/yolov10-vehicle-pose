#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m pytest tests/test_vehicle_pose.py tests/test_carfusion_pipeline.py \
  tests/test_vehicle_pose_metrics.py tests/test_benchmark_tools.py \
  tests/test_pose_pruning_distillation.py tests/test_pose_pruning_pipeline.py \
  -q --disable-warnings
"$PYTHON" benchmarks/compare_outputs.py --device "${DEVICE:-cpu}" --imgsz "${IMGSZ:-128}"

if [[ -f datasets/CarFusion/yolo_pose/conversion_summary.json ]]; then
  "$PYTHON" scripts/check_carfusion.py --root datasets/CarFusion/yolo_pose
else
  echo "BLOCKED data validation: authorized/converted CarFusion data is absent"
fi

"$PYTHON" - <<'PY'
from pathlib import Path
import torch
from ultralytics import YOLOv10

weights = Path("weights/yolov10n.pt")
assert weights.is_file(), "official YOLOv10n weights are missing"
detect = YOLOv10(weights)
with torch.inference_mode():
    output = detect.model.eval()(torch.zeros(1, 3, 128, 128))
assert torch.isfinite(output["one2one"][0]).all()
print("PASS official Detect checkpoint and forward")
PY

if "$PYTHON" -c 'import onnx' >/dev/null 2>&1; then
  echo "ONNX dependency available; run the dedicated export command before release"
else
  echo "SKIP ONNX smoke: onnx is not installed in this environment"
fi
