# 车辆姿态压缩实验归档（2026-07-24）

## 统一基线与下一轮门槛

所有结论均相对于同一 200-epoch Vehicle Pose baseline，并采用 RTX 3090 Ti、640、batch=1、one-to-one-only、Conv/BN fuse、FP16、100 次 warmup、3 x 1000 次前向的配对计时口径。

| 项目 | Baseline | 下一轮必须满足 |
|---|---:|---:|
| Box mAP50-95 | 0.66761 | >= 0.64761（绝对损失 <= 0.02） |
| Pose mAP50-95 | 0.51455 | >= 0.49455（绝对损失 <= 0.02） |
| PCK@0.05 | 0.66863 | 记录，不单独作为准入条件 |
| PCK@0.10 | 0.85248 | 记录，不单独作为准入条件 |
| NME | 0.05960 | 记录，越低越好 |
| 参数量 | 4,005,986 | 尽量下降 |
| Fused FP16 延迟 | 每次配对重测 | 相对该次 baseline 至少下降 10% |

不能将同一模型从 FP32 改成 FP16 的收益，或不同时间独立跑出的基准，记为结构压缩收益。

## 已完成实验

| 实验 | 结构与训练 | 参数量 | Box / Pose mAP50-95 | 配对 FP16 延迟 | 结论 |
|---|---|---:|---:|---:|---|
| FastHead 64x1 + Boxx1 | Head 缩窄、累计 50e、输出 KD | 3,018,722 (-24.64%) | 0.55303 / 0.37569 | 5.3314 vs 5.6955 ms (-6.39%) | 参数下降明显，但两项精度和速度均不达下一轮门槛；淘汰。 |
| LitePSA + P4 C2f 减深 | 保留到累计 60e、KD | 3,842,914 (-4.07%) | 0.65616 / 0.48395 | 5.2056 vs 5.5578 ms (-6.34%) | Box 接近门槛，Pose 损失 0.03060 且速度不足；淘汰。 |
| Joint RepC3 Backbone/Neck/Head | RepC3 + LitePSA + Head96、特征 KD、累计 30e | 3,297,122 (-17.70%) | best 0.48820 / 0.23798 | 训练前静态门禁约 -16.19% | 训练严重失配，远低于门槛；停止，不再续训。 |

FastHead 和 LitePSA+C2f 的最终数值来自加载真实训练 checkpoint 的报告；Joint RepC3 的延迟仅是训练前结构门禁，不得作为可交付模型的最终性能。

## 自适应训练决策

1. FastHead：10e 末段仍上升，因此在独立 checkpoint/run 中扩展到 30e 和最终 50e；50e 后 Pose/Box 仍远低于 baseline，按硬上限停止。
2. LitePSA+C2f：20e 和 40e 的末段仍有改善，故追加至 60e；60e 没有实质扩展 Box/Pose 联合精度包络，停止而非盲目扩展至 80e。
3. Joint RepC3：10e 至 30e 虽有上升，但最优值仍与准入线相差 Box 0.15941、Pose 0.25657；这不是“多训一点”可验证修复的差距，停止。

## 为什么压缩收益有限

该模型仅约 4.0M 参数，batch=1 的 GPU 延迟并不与参数量线性相关。FastHead 的模块剖析中 Backbone + Neck 已占 71.64% 的时间；压缩 Head 即使降低 24.64% 参数，真实端到端延迟也只下降 6.39%。小模型还受 kernel launch、访存、通道对齐和算子利用率限制，因此“少参数”不自动等于“少延迟”。

Joint RepC3 的失败也说明，速度目标不能通过一次替换多处特征块来换取：该替换同时改变 Backbone/Neck 的特征分布和下游姿态塔的输入语义，原始 checkpoint 的同形权重迁移与输出/特征 KD 都不足以恢复关键点几何表示。这次失败没有 OOM、NaN 或 GPU 异常，根因是结构容量和特征拓扑的过度扰动。

## 下一轮保守方案

候选 `yolov10n-vehicle-pose-joint-e4375-head96.yaml` 保留通道宽度、C2f/C2fCIB/PSA 类型和完整 96x2 pose tower，只把高成本 C2f/C2fCIB 的隐藏扩张比例调至 0.4375，并保留 box tower depth=1。

启动训练前必须重新执行配对 fused FP16 benchmark；仅当候选相对同次 baseline 至少快 10%，才从 200e baseline 权重初始化一个独立 10e run。10e 后仅在 Box 与 Pose 都有可信恢复趋势、且仍可能达到本报告顶部的两条精度下限时，再用独立 checkpoint 追加训练。否则立即停止。

## 原始证据

- `reports/fasthead/final_50e/final_report.md`
- `reports/backbone_neck/litepsa_c2f_final_60e_20260724.md`
- `reports/benchmark/fasthead_final_50e/*.json`
- `reports/benchmark/backbone_neck_final_60e/*.json`
- `vehicle_pose_runs_joint15/carfusion_yolov10n_joint_repc3_feature_kd_adaptive_30e_r2/results.csv`（运行目录和权重不入库）
