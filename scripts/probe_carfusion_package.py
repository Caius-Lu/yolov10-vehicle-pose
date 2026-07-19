#!/usr/bin/env python3
"""Safely inventory a CarFusion package without extracting or loading its payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_prefix(path: Path, limit: int = 1024 * 1024) -> str:
    """Hash only the first ``limit`` bytes for fast format/provenance diagnostics."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(limit))
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    suffixes = Counter("".join(path.suffixes).lower() or "<none>" for path in files)
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:10]
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "suffix_counts": dict(sorted(suffixes.items())),
        "largest_files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256_first_1mib": sha256_prefix(path),
            }
            for path in largest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"not a directory: {args.root}")
    print(json.dumps(inventory(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
