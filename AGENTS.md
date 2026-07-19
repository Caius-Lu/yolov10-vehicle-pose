# Project working agreement

- Work from the repository root and keep commands reproducible on Linux and macOS.
- Preserve the upstream YOLOv10 detection task while adding vehicle pose support.
- Never commit CarFusion authorization scripts, access tokens, raw data, generated weights, or `.venv`.
- Do not report training, accuracy, or latency results unless the corresponding command actually ran.
- Keep model, data, training, and benchmark settings in versioned YAML/CLI configuration; do not hard-code local paths or GPU IDs.
- Run focused tests first, then `scripts/validate_project.sh` before delivery.
- Store stage evidence in `reports/` and append material commands and decisions to `reports/execution_journal.md`.

