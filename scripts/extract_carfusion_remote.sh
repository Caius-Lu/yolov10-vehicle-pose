#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/CarFusion.tar.gz.00}"
DESTINATION="${2:-/home/caius/projects/yolopose/data/CarFusion/extracted}"
EXPECTED_SHA256="${CARFUSION_SHA256:-34ae638404d724a50b744dc7cba359b4f38cc336d9c45e5f0cea6a996fe90eb4}"

test -f "$ARCHIVE" || { echo "archive not found: $ARCHIVE"; exit 2; }
test ! -e "$DESTINATION/CarFusion/carfusion.tar" || {
  echo "refusing to overwrite existing inner archive: $DESTINATION/CarFusion/carfusion.tar"
  exit 2
}
test ! -e "$DESTINATION/carfusion" || {
  echo "refusing to overwrite existing extracted dataset: $DESTINATION/carfusion"
  exit 2
}

ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256" || {
  echo "SHA-256 mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256"
  exit 3
}

mkdir -p "$DESTINATION"
echo "PASS archive SHA-256: $ACTUAL_SHA256"
echo "Extracting outer gzip/tar..."
tar -xzf "$ARCHIVE" -C "$DESTINATION"

INNER="$DESTINATION/CarFusion/carfusion.tar"
test -f "$INNER" || { echo "inner archive not found: $INNER"; exit 4; }
echo "Extracting inner tar..."
tar -xf "$INNER" -C "$DESTINATION"

ROOT="$DESTINATION/carfusion"
test -f "$ROOT/annotations/car_keypoints_train.json" || { echo "train annotations missing"; exit 5; }
test -f "$ROOT/annotations/car_keypoints_test.json" || { echo "test annotations missing"; exit 5; }
test -d "$ROOT/train" || { echo "train images missing"; exit 5; }
test -d "$ROOT/test" || { echo "test images missing"; exit 5; }

echo "PASS two-stage extraction"
du -sh "$DESTINATION"
find "$ROOT" -type f | awk '
  /\/annotations\/.*\.json$/ {annotations++}
  /\/train\/.*\.jpg$/ {train_images++}
  /\/test\/.*\.jpg$/ {test_images++}
  END {printf "annotation_files=%d train_images=%d test_images=%d\n", annotations, train_images, test_images}
'
