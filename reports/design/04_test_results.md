# 04 Test Results

Remote environment: RTX 3090 Ti, PyTorch 2.6.0+cu124.

- Vehicle Pose core + Detect regression: 8 tests passed.
- CarFusion manifest/converter/quality/preview fixture: 10 tests passed.
- Vehicle pose PCK/NME metrics: 2 tests passed.
- Combined branch coverage for the four CarFusion data scripts: 84%.
- CPU forward/backward including empty targets: PASS.
- Checkpoint roundtrip: PASS.
- Official YOLOv10n checkpoint load under PyTorch 2.6: PASS.
- CUDA 640 forward: PASS.
- End-to-end custom Predictor result construction: PASS.
- Generated fixture integration: one complete CUDA epoch, two optimizer steps, validation, last/best checkpoint save, optimizer stripping and final checkpoint validation all PASS. All 10 dual-branch Loss components were finite and non-zero. This fixture is not accuracy evidence.
- Validator PCK/NME pairing/export integration and batch=1 end-to-end benchmark helpers: 3 additional tests passed; generated fixture produced traceable empty-pair/`null` metrics rather than fabricated accuracy.
- Original vs one2one-only: max/mean absolute error 0.
- ONNX: not run; `onnx` is not installed in the remote compatibility environment.

No test was skipped silently. Fixture tests and their zero AP are not real training evidence.
