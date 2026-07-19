# 06 Smoke Training Report

Status: **WAITING_FOR_VERIFIED REAL-DATA CONVERSION**

The model forward/backward, optimizer-compatible gradients, empty-target behavior and checkpoint roundtrip are covered by synthetic fixture tests. In addition, `scripts/smoke_fixture_train.py` completed one CUDA epoch on 4 generated train + 2 generated validation images, including two optimizer steps, all 10 finite/non-zero dual-branch Loss components, validation, last/best checkpoint save, optimizer stripping and final checkpoint validation.

The fixture rerun also verified automatic `matched_pose_pairs.json` and `vehicle_pose_metrics.json` output. No prediction reached bbox IoU≥0.5, so PCK/NME remained `null`; no accuracy was fabricated.

This only proves Trainer/Validator plumbing. Its AP values are meaningless and are not baseline results. The user-supplied real package is transferring to the server. After hash verification/conversion, `create_carfusion_lists.py` fixes 128 train + 32 val images and their hashes; `scripts/train_vehicle_pose_smoke.sh` enforces both dataset READY and list gates.
