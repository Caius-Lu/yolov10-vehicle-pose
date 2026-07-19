# 08 Final Time-Optimization Recommendation

Status: **VALIDATED / BASELINE UNCHANGED**

## Recommendation

For the next training or inference run, stage the converted YOLO dataset from the Windows-mounted path (`/mnt/e/...`) onto the WSL Linux filesystem (for example `/home/caius/datasets/CarFusion/yolo_pose`) before starting GPU work. Use the independent entry point:

```bash
scripts/stage_carfusion_to_wsl.sh \
  datasets/CarFusion/yolo_pose \
  /home/caius/datasets/CarFusion/yolo_pose \
  reports/optimization/runtime_configs/CarFusion-pose-wsl.yaml
```

The script dereferences existing image symlinks, copies into an independent directory, verifies every file by SHA-256, and creates a new runtime YAML. It does not edit the baseline dataset, model code, checkpoint, training configuration or result directory.

## Controlled result

Same RTX 3090 Ti, same `best.pt`, same 100 JPEG files, same predictor, batch 1, image size 640, 20 warmups and 3 rounds × 1000 iterations:

| Storage path | Mean latency | Mean P95 | FPS from mean |
|---|---:|---:|---:|
| Windows E drive through `/mnt/e` | 60.10 ms | 109.25 ms | 16.64 |
| WSL ext4 staged copy | 16.63 ms | 18.62 ms | 60.12 |
| Change | **−72.3%** | **−83.0%** | **3.61×** |

The staged 100-file subset passed SHA-256 identity with 0 mismatches. Because the optimization changes only file location, model weights, decoded image bytes and numerical precision remain unchanged; no accuracy trade-off is introduced.

Raw evidence:

- `reports/benchmark/carfusion_baseline_best_boxfix_e2e.json`
- `reports/optimization/carfusion_wsl_local_e2e.json`
- `scripts/stage_carfusion_to_wsl.sh`
- `scripts/verify_staged_tree.py`

## Measurement boundary

The reliable comparison is the end-to-end wall time measured with `perf_counter_ns`. One internal component sample reported a negative preprocess value because the upstream profiler uses `time.time()` around synchronized GPU regions; system clock adjustment can make that component invalid. It does not enter the wall-time calculation or the 3.61× conclusion.

## Why this is the first action

Microsoft recommends keeping Linux-command-line workloads in the WSL filesystem rather than mounted Windows drives for best performance: [Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems). PyTorch also recommends measuring `num_workers`, prefetching, pinned memory and persistent workers for the actual workload rather than assuming one universal setting: [DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html) and [data-loading optimization tutorial](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html).

After storage locality is fixed, the next low-risk experiment is a small controlled sweep of `workers`, `persistent_workers` and `prefetch_factor`. TensorRT FP16 can further reduce inference time, but it is a separate deployment optimization and requires numerical/accuracy regression because reduced precision can affect outputs; see NVIDIA's [TensorRT performance best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) and [accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html).
