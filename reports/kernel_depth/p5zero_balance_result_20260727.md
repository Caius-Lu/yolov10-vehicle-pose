# P5-Zero 速度—精度平衡方案（2026-07-27）

## 结论

在不使用 TensorRT 的前提下，P5-Zero 首次同时通过本项目的速度、精度和
CUDA 内存门禁。该方案基于已恢复的 LitePSA + C2f-P4D1 模型，只移除
backbone P5 与 neck P5 两个 C2f 的内部 bottleneck，完整保留 P3、P4
以及 Box、Class、Pose 三个任务塔。

| 指标 | 原始 baseline | P5-Zero | 变化 | 门禁 |
|---|---:|---:|---:|---:|
| paired fused-FP16 latency | 5.94537 ms | 5.06345 ms | **-14.83371%** | ≥10%，通过 |
| Box mAP50-95 | — | **0.66254** | 门槛余量 +0.01493 | ≥0.64761，通过 |
| Pose mAP50-95 | — | **0.49592** | 门槛余量 +0.00137 | ≥0.49455，通过 |
| fused 参数 | 3,992,210 | 3,387,794 | **-15.13989%** | 记录项 |
| 参数/Buffer 存储 | 7,984,420 B | 6,775,592 B | **-15.13984%** | 记录项 |
| 模型驻留 CUDA 内存 | 57,721,856 B | 54,093,312 B | **-6.28626%** | 下降，通过 |
| 推理峰值 CUDA 内存 | 72,237,568 B | 68,609,024 B | **-5.02307%** | 下降，通过 |

精度来自最佳 checkpoint 的完整 3,606 图验证记录；最终 fused-FP16
复验显示 Box/Pose 为 0.662/0.496，PCK@0.05 为 0.64366，
PCK@0.10 为 0.82799，NME 为 0.06729。延迟为 RTX 3090 Ti、
batch=1、640、one-to-one-only、同进程 paired AB/BA 结果，不应将绝对
毫秒数直接外推到其他硬件。

## 结构与权重迁移

- 模型配置：
  `ultralytics/cfg/models/v10/yolov10n-vehicle-pose-litepsa-c2f-p4d1-p5zero-head96.yaml`
- 恢复配置：
  `configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml`
- 姿态精修配置：
  `configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml`
- 同源教师：
  `vehicle_pose_runs_backbone_neck/carfusion_yolov10n_litepsa_c2f_p4d1_kd_adaptive_60e_r3/weights/best.pt`
- 最终 checkpoint：
  `vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt`

迁移器直接复制 605 个张量，对 layer 8 和 layer 22 的两个 C2f 投影做语义
折叠；missing/unexpected key 均为 0。卷积核尺寸缩小时使用中心裁剪，
通道缩小时保持前缀切片，这两类行为均有回归测试覆盖。

## 候选筛选

候选先通过 paired latency gate，再决定是否训练。最终门禁的比较基准始终
是原始完整 baseline；“相对同源恢复模型的增量”仅用于诊断，不能代替最终
门禁。

| 候选 | paired latency | 相对比较对象收益 | 处理 |
|---|---:|---:|---|
| recovered source | 5.84769→5.42568 ms | +7.21673%，相对原始 baseline | 作为同源教师与结构起点 |
| P5-Zero | 5.30245→4.93782 ms | +6.8766%，相对 recovered source | 对原始 baseline 实为 +14.72%，进入恢复 |
| P4-neck0 + P5-Zero | 5.68929→5.16267 ms | +9.25633% | 增益不足且更激进，停止 |
| P4-neck0 + P5-Zero + LitePSA k3 | 5.47342→5.00268 ms | +8.60052% | k3 未带来稳定增益，停止 |
| P4-both0 + P5-Zero | 5.43736→4.86304 ms | +10.56234% | 进入恢复，但精度平台，拒绝 |

P4-both0 + P5-Zero 的同源 KD 恢复在前 3 轮得到
0.55074/0.35087、0.60047/0.41162、0.60121/0.41628，明显低于精度线，
第 4 轮早期终止；训练显存约 14.8–15.4 GB。它证明继续压缩 P4 会破坏
高分辨率定位与姿态容量。

## 精度恢复轨迹

P5-Zero 的投影折叠初始化完整验证为 Box/Pose 0.551/0.387。全模型同源 KD
恢复轨迹如下：

| epoch | Box mAP50-95 | Pose mAP50-95 | 结论 |
|---:|---:|---:|---|
| 1 | 0.60299 | 0.43531 | 快速恢复 |
| 2 | **0.66245** | **0.49369** | Box 通过；Pose 距门槛 0.00086 |
| 3 | 0.65026 | 0.49116 | 回落 |
| 4 | 0.64727 | 0.48279 | 平台，停止 |

随后仅训练 `model.23.cv4.*` 与 `model.23.one2one_cv4.*` 共 48 个姿态塔张量，
冻结其余参数并固定冻结 BN 行为。第 1 轮即达到 Box/Pose
**0.66254/0.49592**，训练显存约 4.47 GB；门禁通过后立即停止后续轮次，
避免无信息增益的重复训练。

## 可靠性与已知边界

- 延迟证据：
  `p5zero_final_vs_original_paired_fp16.json`
- 权重迁移证据：
  `p5zero_transfer.json`
- 内存证据：
  `original_baseline_memory_final.json`、`p5zero_final_memory.json`
- 其他候选证据：
  `p5zero_paired_fp16.json`、`p4neck0_p5zero_paired_fp16.json`、
  `p4neck0_p5zero_k3_paired_fp16.json`、`p4both0_p5zero_paired_fp16.json`

最终验证第一次使用 `workers=8` 时，并非模型或 CUDA 卡死，而是
Ultralytics 尝试联网下载 Arial 字体并阻塞在 TLS。将系统 DejaVuSans
写入本机 Ultralytics 字体缓存后，以 `workers=0` 重启并完成验证。另有
W&B 因绝对项目路径失败，恢复训练改用 `WANDB_MODE=disabled`；两项基础设施
问题均未被计入模型结论。

Pose 门槛余量为 0.00137，属于通过但余量较小。部署到新 GPU、切换 PyTorch/
CUDA 版本、修改数据集或验证代码后，都必须重新执行 paired latency、完整
精度和独立进程 CUDA memory 三门禁。

## 最小复现流程

```bash
# 1. 结构与迁移回归
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_fasthead_transfer.py tests/test_lite_backbone_blocks.py

# 2. 同源恢复
WANDB_MODE=disabled .venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml

# 3. 姿态塔选择性精修
WANDB_MODE=disabled .venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml

# 4. 使用 benchmarks/benchmark_deploy_fp16.py 做原始 baseline 与最终
#    checkpoint 的同进程 paired fused-FP16 对比。
```

原始数据、训练权重和大体积运行目录不进入 Git；仓库提交配置、实现、测试、
JSON 证据和复盘文档，checkpoint 按项目既有制品流程保存。
