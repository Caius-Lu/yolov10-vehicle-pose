#!/usr/bin/env python3
"""Create deterministic smoke, overfit, and fixed-evaluation image lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _read(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty source list: {path}")
    return values


def _sample(values: list[str], count: int, seed: int) -> list[str]:
    if count > len(values):
        raise ValueError(f"requested {count} entries from a {len(values)}-entry list")
    return random.Random(seed).sample(values, count)


def _write(path: Path, values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def create_lists(
    root: Path,
    fixed_eval_output: Path,
    smoke_train: int = 128,
    smoke_val: int = 32,
    overfit: int = 32,
    fixed_eval_count: int = 100,
    seed: int = 0,
) -> dict:
    """Write all fixed lists and return counts plus SHA-256 hashes."""
    root = Path(root)
    train, val, test = (_read(root / f"{split}.txt") for split in ("train", "val", "test"))
    selected = {
        "smoke_train": _sample(train, smoke_train, seed),
        "smoke_val": _sample(val, smoke_val, seed + 1),
        "overfit": _sample(train, overfit, seed + 2),
        "fixed_eval": _sample(test, fixed_eval_count, seed + 3),
    }
    hashes = {
        "smoke_train": _write(root / "smoke_train.txt", selected["smoke_train"]),
        "smoke_val": _write(root / "smoke_val.txt", selected["smoke_val"]),
        "overfit": _write(root / "overfit.txt", selected["overfit"]),
        "fixed_eval": _write(
            Path(fixed_eval_output),
            [str((root / value).resolve()) if not Path(value).is_absolute() else value for value in selected["fixed_eval"]],
        ),
    }
    summary = {
        "seed": seed,
        "counts": {name: len(values) for name, values in selected.items()},
        "sha256": hashes,
        "fixed_eval_output": str(Path(fixed_eval_output).resolve()),
    }
    (root / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path("datasets/CarFusion/yolo_pose"), type=Path)
    parser.add_argument("--fixed-eval-output", default=Path("reports/training/fixed_eval_images.txt"), type=Path)
    parser.add_argument("--smoke-train", default=128, type=int)
    parser.add_argument("--smoke-val", default=32, type=int)
    parser.add_argument("--overfit", default=32, type=int)
    parser.add_argument("--fixed-eval-count", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            create_lists(
                args.root,
                args.fixed_eval_output,
                args.smoke_train,
                args.smoke_val,
                args.overfit,
                args.fixed_eval_count,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
