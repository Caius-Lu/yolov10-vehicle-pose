# LitePSA + P4 C2f 减深：60-epoch 最终结论

## 结论

LitePSA 与 P4 C2f 减深在 RTX 3090 Ti、640×640、batch=1、
one-to-one-only、Conv/BN fuse、FP16 口径下，将平均推理延迟从
`5.5578 ms` 降至 `5.2056 ms`，降幅 `6.34%`；参数量减少 `4.07%`。

速度目标成立，但精度代价仍不可忽略：相对 200-epoch baseline，
Box mAP50-95 下降 `1.145` 个百分点，Pose mAP50-95 下降 `3.060`
个百分点。因此该模型适合作为延迟受限候选，不替代精度优先的 baseline。

## 训练门禁

| 阶段 | 累计 epoch | 结果与决策 |
|---|---:|---|
| r1 | 20 | 末段 Box/Pose/PCK 上升且 NME 下降，进入 r2 |
| r2 | 40 | 最佳 Box `0.65776`、Pose `0.48275`，末段浅平台，允许一次有界探测 |
| r3 | 60 | Pose/PCK/NME 仅微幅刷新，Box 未超过 r2；停止，不运行 80 epoch |

r3 的训练选择器保留追加第 8 轮 `best.pt`。末轮虽然 PCK/NME 更好，
但综合验证包络没有实质扩展，不混用不同 epoch 的指标。

## 精度、参数量与模型大小

| 指标 | 200e baseline | LitePSA+C2f 60e best | 差值 |
|---|---:|---:|---:|
| Box mAP50-95 | 0.66761 | 0.65616 | -0.01145 |
| Pose mAP50-95 | 0.51455 | 0.48395 | -0.03060 |
| PCK@0.05 | 0.66863 | 0.64315 | -0.02548 |
| PCK@0.10 | 0.85248 | 0.83195 | -0.02053 |
| NME（越低越好） | 0.05960 | 0.06572 | +0.00612 |
| 参数量（未融合） | 4,005,986 | 3,842,914 | -4.07% |
| 参数量（融合后） | 3,992,210 | 3,829,778 | -4.07% |
| checkpoint | 8,422,070 B | 8,025,577 B | -4.71% |

## 同口径 FP16 Benchmark

每个模型执行 100 次 warmup、3 轮 × 1000 次计时；两者均使用同一台
RTX 3090 Ti，并加载各自真实训练 checkpoint。

| 指标 | 200e baseline | LitePSA+C2f 60e | 改善 |
|---|---:|---:|---:|
| 平均延迟 | 5.5578 ms | 5.2056 ms | -6.34% |
| 平均 p50 | 5.3986 ms | 5.0465 ms | -6.52% |
| 平均 p95 | 6.4551 ms | 6.1383 ms | -4.91% |
| FPS（由均值换算） | 179.93 | 192.10 | +6.77% |
| 相对加速比 | 1.000× | 1.068× | +6.77% |

原始证据：

- `reports/benchmark/backbone_neck_final_60e/baseline200_fused_fp16.json`
- `reports/benchmark/backbone_neck_final_60e/litepsa_c2f_60e_fused_fp16.json`
- `vehicle_pose_runs_backbone_neck/carfusion_yolov10n_litepsa_c2f_p4d1_kd_adaptive_60e_r3/results.csv`

## 工程判断

1. LitePSA 是主要延迟收益来源；单纯用 C2fCIB 替换虽然减少参数，
   但前置门禁实测延迟反而上升，因此未浪费训练预算。
2. P4 C2f 减深进一步降低计算量，但小模型已经接近并行与 kernel-launch
   开销下限，参数下降不会等比例转化为延迟下降。
3. 40→60 epoch 的收益已明显衰减，继续训练无法修复结构容量损失；
   按门禁停止比盲目扩展到 80 epoch 更合理。
4. 部署选择应按业务约束分流：精度优先使用 200e baseline；允许约
   3.1 个 Pose mAP 点损失、追求约 6.3% 延迟收益时使用本候选。
