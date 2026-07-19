#!/usr/bin/env python3
"""Verify that a staged dataset tree has the same files and SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def iter_files(root: Path):
    for directory, _, filenames in os.walk(root, followlinks=True):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.is_file():
                yield path.relative_to(root), path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tree(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    source_files = dict(iter_files(source))
    destination_files = dict(iter_files(destination))
    missing = sorted(str(path) for path in source_files.keys() - destination_files.keys())
    extra = sorted(str(path) for path in destination_files.keys() - source_files.keys())
    mismatches = []
    total_bytes = 0
    for relative in sorted(source_files.keys() & destination_files.keys()):
        source_path = source_files[relative]
        destination_path = destination_files[relative]
        total_bytes += source_path.stat().st_size
        if source_path.stat().st_size != destination_path.stat().st_size:
            mismatches.append(str(relative))
        elif sha256(source_path) != sha256(destination_path):
            mismatches.append(str(relative))
    return {
        "source": str(source),
        "destination": str(destination),
        "source_file_count": len(source_files),
        "destination_file_count": len(destination_files),
        "verified_bytes": total_bytes,
        "missing": missing,
        "extra": extra,
        "sha256_mismatches": mismatches,
        "passed": not (missing or extra or mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify_tree(args.source, args.destination)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
