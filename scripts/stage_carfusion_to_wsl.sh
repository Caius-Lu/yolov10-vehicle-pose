#!/usr/bin/env bash
# Stage the converted CarFusion YOLO tree on WSL ext4 without changing the baseline dataset.

set -euo pipefail

SOURCE_ROOT="${1:-datasets/CarFusion/yolo_pose}"
DEST_ROOT="${2:-/home/caius/datasets/CarFusion/yolo_pose}"
RUNTIME_YAML="${3:-reports/optimization/runtime_configs/CarFusion-pose-wsl.yaml}"
TEMPLATE_YAML="${TEMPLATE_YAML:-ultralytics/cfg/datasets/CarFusion-pose.yaml}"
VERIFY_REPORT="${RUNTIME_YAML%.yaml}_verification.json"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "source dataset does not exist: $SOURCE_ROOT" >&2
  exit 2
fi

case "$DEST_ROOT" in
  /mnt/*)
    echo "destination must be on the WSL Linux filesystem, not under /mnt: $DEST_ROOT" >&2
    exit 2
    ;;
esac

mkdir -p "$DEST_ROOT" "$(dirname "$RUNTIME_YAML")"

# -L materializes image symlinks that currently point into the Windows-mounted drive.
# The destination is independent, so the baseline dataset and outputs remain untouched.
rsync -rltD --copy-links --human-readable --progress "$SOURCE_ROOT/" "$DEST_ROOT/"

python scripts/verify_staged_tree.py \
  --source "$SOURCE_ROOT" \
  --destination "$DEST_ROOT" \
  --output "$VERIFY_REPORT"

python scripts/materialize_dataset_config.py \
  --template "$TEMPLATE_YAML" \
  --output "$RUNTIME_YAML" \
  --dataset-root "$DEST_ROOT"

printf 'PASS: staged dataset verified\nsource=%s\ndestination=%s\nruntime_yaml=%s\nverification=%s\n' \
  "$SOURCE_ROOT" "$DEST_ROOT" "$RUNTIME_YAML" "$VERIFY_REPORT"
