#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"
CHECKPOINT="${1:?usage: scripts/resume_vehicle_pose.sh path/to/last.pt}"
test -f "$CHECKPOINT" || { echo "checkpoint not found: $CHECKPOINT"; exit 2; }
.venv/bin/yolo pose train resume model="$CHECKPOINT" device="${DEVICE:-0}"
