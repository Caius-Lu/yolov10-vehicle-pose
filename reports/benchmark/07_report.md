# 07 PyTorch Benchmark Report

Status: **SMOKE_ONLY; FORMAL_BENCHMARK_BLOCKED_BY_TRAINED_CHECKPOINT**

A short execution smoke used batch=1, 640×640 FP32 on RTX 3090 Ti with 5 warmups, 10 timed iterations and one round:

| Mode | Mean | P50 | P95 | P99 | FPS from mean |
|---|---:|---:|---:|---:|---:|
| torch_original | 8.545 ms | 8.558 ms | 8.794 ms | 8.815 ms | 117.0 |
| torch_one2one_only | 6.415 ms | 6.368 ms | 6.735 ms | 6.781 ms | 155.9 |

The one-to-one tensors were exactly equal. These numbers are not formal results: the pose head is untrained, the run is 10×1 rather than 1000×3, synthetic input was used, and forward/decode/end-to-end modes plus FP16/compile/channels-last remain for the trained-checkpoint stage.

Raw evidence is in `07_results.json`, `07_results.csv`, and `07_environment.json`.

