# YOLOPose 网络耗时优化：分辨率感知特征蒸馏

## 1. 目标与边界

- 目标：在 RTX 3090 Ti、batch=1 的相同 PyTorch 前向口径下明显降低网络耗时，同时尽量维持车辆框与 13 个有效关键点的精度。
- 基线：`carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt`，输入 640。
- 隔离原则：不修改基线权重、基线结果和默认 Trainer；所有新代码、配置、权重与 benchmark 输出均使用独立路径。
- 本轮是 3 epoch 可行性验证，不把短跑结果冒充充分收敛结果。

## 2. 方法调研与选择

| 方法 | 推理收益 | 精度风险 | 本轮处理 |
|---|---:|---:|---|
| FP16/TensorRT | 高，且不改变网络语义 | 低 | 属于部署优化，建议与本方案叠加，但不是本轮“网络计算量”主变量 |
| 输入 640→512 | 理论像素/FLOPs 约减少 36% | 中等，关键点更敏感 | 作为学生网络的主要加速来源 |
| 640 教师→512 学生蒸馏 | 推理时无额外开销 | 低至中，取决于损失设计 | 本轮实现并验证 |
| 通道剪枝/更小宽度学生 | 可进一步减少参数与 MACs | 中高，需要重新搜索与充分训练 | 作为第二阶段，不在 3 epoch 中贸然执行 |
| INT8 QAT | 部署端收益高 | 关键点回归精度风险更高 | 在 512+KD 收敛后另做硬件相关验证 |

依据：

1. Hinton、Vinyals、Dean 提出的知识蒸馏用教师软知识训练更易部署的学生模型：<https://arxiv.org/abs/1503.02531>。
2. FitNets 证明中间表示可以作为教师提示，指导更快的学生：<https://arxiv.org/abs/1412.6550>。
3. 检测任务存在前景/背景不均衡，直接等权模仿全部像素并非最优；Prediction-guided Feature Imitation 给出了检测蒸馏证据：<https://arxiv.org/abs/2112.04840>。
4. YOLOv6 官方报告和实现将自蒸馏用于工业检测器的精度—速度折中：<https://arxiv.org/abs/2209.02976>、<https://github.com/meituan/YOLOv6>。
5. PyTorch 官方教程明确说明蒸馏只影响训练权重，不增加学生推理路径的成本：<https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html>。

## 3. 当前实现

教师和学生都从已完成的 CarFusion `best.pt` 出发：

- 教师：冻结参数，输入 640；
- 学生：输入 512，正常接受框、类别、关键点坐标和可见性监督；
- 蒸馏位置：检测/关键点头之前的 P3、P4、P5 三层特征；
- 对齐：把教师特征双线性缩放到学生空间尺寸，按通道做 L2 归一化后计算均方距离；
- 总损失：`L = L_pose_supervised + lambda * L_feature_kd`，当前 `lambda=0.5`；
- 推理：只保存学生，教师不会注册为学生子模块，不进入优化器、EMA 或 checkpoint。

实现文件：

- `ultralytics/models/yolov10/distill.py`
- `scripts/train_pose_distillation.py`
- `configs/pose_distill_512.yaml`
- `tests/test_pose_distillation.py`
- `benchmarks/benchmark_resolution_pair.py`
- `benchmarks/benchmark_deploy_fp16.py`

## 4. 可复现命令

```bash
# 聚焦测试
.venv/bin/python -m pytest -q tests/test_pose_distillation.py tests/test_vehicle_pose.py

# 3 epoch 独立实验
WANDB_MODE=disabled .venv/bin/python scripts/train_pose_distillation.py \
  --config configs/pose_distill_512.yaml

# 网络前向 benchmark（学生 checkpoint 生成后）
.venv/bin/python benchmarks/benchmark_torch.py \
  --checkpoint vehicle_pose_runs_distill/carfusion_yolov10n_distill_512_3e/weights/best.pt \
  --imgsz 512 --batch 1 \
  --output reports/benchmark/distill_512

# 推荐部署路径：640 + one-to-one-only + Conv/BN fuse + FP16
.venv/bin/python benchmarks/benchmark_deploy_fp16.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  --imgsz 640 --warmup 100 --iterations 1000 --rounds 4 \
  --output reports/benchmark/network_distillation/deploy_fp16_640.json
```

## 5. 已获得的对照证据

原 640 baseline：PCK@0.05=0.6686，PCK@0.10=0.8525，NME=0.05960。

同一 `best.pt` 直接改为 512、尚未蒸馏微调：PCK@0.05=0.63235，PCK@0.10=0.83076，NME=0.06632。由此可见，直接降分辨率的精度损失不可忽略，必须通过目标分辨率微调/蒸馏做补偿。

## 6. 正式 3 epoch 结果

训练设置：完整 16,581 张训练图、3,606 张验证图，batch 32，学生 512、教师 640，教师与学生都由 baseline `best.pt` 初始化，SGD `lr0=1e-4`，无 warmup。训练链路、验证、EMA 和 checkpoint 均真实执行完成；best 对应 epoch 2。

| 模型/口径 | Box mAP50-95 | Pose mAP50-95 | PCK@0.05 | PCK@0.10 | NME |
|---|---:|---:|---:|---:|---:|
| baseline FP32 640 | 0.665 | 0.515 | 0.66863 | 0.85248 | 0.05960 |
| baseline 权重直接 512 | — | — | 0.63235 | 0.83076 | 0.06632 |
| 512 蒸馏 3 epoch best | 0.633 | 0.474 | 0.63550 | 0.83490 | 0.06634 |

结论：3 epoch 蒸馏相对“直接 512”只恢复约 0.315 个 PCK@0.05 百分点，仍比 640 baseline 低约 3.31 个百分点。它证明实现链路可用，但没有达到“精度几乎不损失”的验收目标。

## 7. 配对网络耗时复测：为什么 512 没有更快

最初先后运行的两组 benchmark 受 GPU 调度影响，离散度很大。随后使用同一进程、同一权重、同一 CUDA 环境，按 AB/BA 顺序交错执行 4 轮 × 500 次：

| PyTorch eager 路径 | 640 mean | 512 mean | 结果 |
|---|---:|---:|---:|
| dual-head 原路径 | 9.183 ms | 9.272 ms | 512 慢 0.97% |
| one-to-one-only | 6.905 ms | 7.001 ms | 512 慢 1.39% |

当前网络仅约 4M 参数，在 RTX 3090 Ti、batch 1 下主要受 kernel launch 和低 GPU 利用率限制，而不是纯 FLOPs 限制。因此减少输入像素没有转化成实际 latency 收益；蒸馏只改变权重，也不会自动缩短推理图。

## 8. 最终推荐：保留 640，融合 + FP16 + one-to-one-only

统一 640 输入、batch 1、100 次 warmup、4 轮 × 1000 次 CUDA Event 计时，并按 AB/BA 顺序交错 FP32 与 FP16：

| 路径 | mean 网络前向 | 相对 FP32 one-to-one |
|---|---:|---:|
| FP32 one-to-one-only | 7.147 ms | 1.000× |
| Conv/BN fused + FP16 one-to-one-only | 5.727 ms | **1.248×，延迟下降 19.88%** |

完整验证集 FP16 结果：Box mAP50-95=0.665，Pose mAP50-95=0.514，PCK@0.05=0.66763，PCK@0.10=0.85238，NME=0.05969。相对 FP32，PCK@0.05 仅下降 0.00100，PCK@0.10 仅下降 0.00010；这符合“明显优化耗时、精度几乎不损失”的目标。

建议落地顺序：

1. 生产默认保持 640，不替换当前 baseline 权重；启用已经验证无语义差异的 one-to-one-only 推理分支。
2. 部署模型执行 Conv/BN fuse 并使用 FP16；若部署环境支持 TensorRT，再导出静态 640、batch 1 的 FP16 engine，并用同一 3,606 图门禁复验。
3. 512 蒸馏 checkpoint 仅保留为研究产物，不设为生产默认。若未来硬件是算力受限的边缘设备而不是 3090 Ti，可在独立分支把训练延长到 20–30 epoch，并加入前景/关键点区域加权后重新判断。

证据文件：

- `reports/optimization/evidence/distilled_512_3e_results.csv`
- `reports/optimization/evidence/distilled_512_3e_vehicle_pose_metrics.json`
- `reports/optimization/evidence/baseline_best_at_512_vehicle_pose_metrics.json`
- `reports/optimization/evidence/baseline_best_fp16_640_vehicle_pose_metrics.json`
- `reports/benchmark/network_distillation/paired_640_512.json`
- `reports/benchmark/network_distillation/deploy_fp16_640.json`
