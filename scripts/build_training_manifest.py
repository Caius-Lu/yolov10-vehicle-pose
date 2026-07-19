#!/usr/bin/env python3
"""Record Git, environment, config, weight, and dataset hashes before a training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def _hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _git(project: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(
    project: Path, output: Path, model_config: Path, data_config: Path, checkpoint: Path | None = None
) -> dict:
    """Write a reproducibility manifest for one training entry point."""
    project, output = Path(project), Path(output)
    candidates = {
        "model_config": Path(model_config),
        "data_config": Path(data_config),
        "checkpoint": Path(checkpoint) if checkpoint else None,
        "conversion_summary": project / "datasets/CarFusion/yolo_pose/conversion_summary.json",
        "split_summary": project / "datasets/CarFusion/yolo_pose/split_summary.json",
        "subset_summary": project / "datasets/CarFusion/yolo_pose/subset_summary.json",
    }
    files = {name: _hash(path) for name, path in candidates.items() if path is not None and path.is_file()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project.resolve()),
        "git_commit": _git(project, "rev-parse", "HEAD"),
        "git_status_short": _git(project, "status", "--short"),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_manifest(args.project, args.output, args.model_config, args.data_config, args.checkpoint),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
