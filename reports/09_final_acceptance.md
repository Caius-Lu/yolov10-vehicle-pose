# 09 Final Acceptance

Overall status: **PASS WITH ONE SOURCE-DATA LIMITATION**

## Completed gates

- Fixed upstream YOLOv10 baseline, project layout, local/remote synchronization and WSL CUDA environment.
- Native 14×3 Vehicle Pose head, dual-assignment loss, NMS-free one-to-one postprocess and eval-only fast path.
- Deterministic CarFusion conversion, explicit invalid-bbox policy, group-aware split, image/label checks and rendered review set.
- Real-data smoke, 32-image overfit and 200-epoch baseline.
- Resume recovery after WSL reboot and E-drive mount disappearance; no duplicate baseline was started.
- Final bbox, pose, PCK/NME and visibility-sliced evaluation.
- Validator plots plus 100-image raw predictor visual gate.
- Test-first predictor box-format fix; full pytest exit code 0 and Vehicle Pose regression 9/9.
- Fixed-checkpoint end-to-end benchmark and independent WSL ext4 storage optimization validation.
- Reproducible staging and SHA-256 verification scripts.
- Codex-focused department share manuscript and PowerPoint deliverables.

## Final baseline evidence

| Evidence | Result |
|---|---|
| Validation set | 3,606 images / 11,569 instances |
| Box mAP50 / mAP50-95 | 0.818 / 0.665 |
| Pose mAP50 / mAP50-95 | 0.765 / 0.515 |
| PCK@0.05 / PCK@0.10 | 0.6686 / 0.8525 |
| NME | 0.05960 |
| Raw-image visual gate | PASS after predictor `xywh→xyxy` fix |
| E-drive → WSL ext4 end-to-end latency | 60.10 → 16.63 ms (−72.3%, 3.61×) |

## Accepted limitation

Keypoint index 13 is visibility 0 in every one of the 62,434 converted source instances, so the 14th output coordinate has no supervision or evaluation evidence. The implementation remains 14×3 for schema compatibility, but this dataset release supports accuracy claims for 13 effective points only. Horizontal flip remains disabled until the full semantic left/right mapping is confirmed.

## Evidence map

- Training: `reports/training/06_baseline_report.md`
- Evaluation: `reports/evaluation/carfusion_baseline_best_metrics.json`
- Visuals: `reports/visualization/review_boxfix/`
- Baseline benchmark: `reports/benchmark/carfusion_baseline_best_boxfix_e2e.json`
- Optimization: `reports/optimization/08_final_recommendation.md`
- Process journal: `reports/execution_journal.md`
- Department manuscript: `docs/部门分享_从YOLOv10到车辆关键点训练系统.md`
