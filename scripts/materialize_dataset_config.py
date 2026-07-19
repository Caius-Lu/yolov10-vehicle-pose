#!/usr/bin/env python3
"""Materialize a portable dataset YAML with an absolute runtime data root."""

import argparse
from pathlib import Path


def materialize_dataset_config(template, output, dataset_root):
    template, output = Path(template), Path(output)
    dataset_root = Path(dataset_root).resolve()
    lines = template.read_text(encoding="utf-8").splitlines()
    path_indexes = [index for index, line in enumerate(lines) if line.startswith("path:")]
    if len(path_indexes) != 1:
        raise ValueError(f"expected exactly one top-level path entry in {template}")
    lines[path_indexes[0]] = f"path: {dataset_root}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    print(materialize_dataset_config(args.template, args.output, args.dataset_root))


if __name__ == "__main__":
    main()
