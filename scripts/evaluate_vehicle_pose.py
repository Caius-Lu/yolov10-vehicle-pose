#!/usr/bin/env python3
"""Compute PCK/NME for already box-matched vehicle pose prediction/GT pairs."""

import argparse
import csv
import json
from pathlib import Path

from ultralytics.utils.vehicle_pose_metrics import compute_vehicle_pose_metrics


def compute_metrics(pairs):
    """Compatibility wrapper used by the documented standalone CLI and tests."""
    return compute_vehicle_pose_metrics(pairs)


def main():  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=Path("reports/training/06_baseline_metrics.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("reports/training/06_baseline_metrics.csv"))
    args = parser.parse_args()
    pairs = json.loads(args.pairs_json.read_text(encoding="utf-8"))
    metrics = compute_metrics(pairs)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("index", "count", "pck@0.05", "pck@0.10", "nme"))
        writer.writeheader()
        writer.writerows(metrics["per_keypoint"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
