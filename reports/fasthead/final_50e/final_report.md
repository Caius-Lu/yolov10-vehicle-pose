# FastHead 50e 最终验证与耗时瓶颈复盘

## 结论

FastHead 64×1 / Box×1 在 RTX 3090 Ti 上减少 24.64% 参数，并在统一 640、batch1、one-to-one-only、Conv/BN fuse、FP16 口径下减少 6.39% 延迟；但 Pose mAP50-95 相对 200e baseline 下降 0.13886。候选通过速度门禁，失败于精度门禁，不替换 baseline。

## 最终对比

| 指标 | 200e baseline | FastHead 50e | 变化 |
|---|---:|---:|---:|
| 参数量 | 4,005,986 | 3,018,722 | -24.64% |
| fused 参数量 | 3,992,210 | 3,006,098 | -24.70% |
| checkpoint | 8.42 MB | 6.10 MB | -27.55% |
| Box mAP50-95 | 0.66761 | 0.55303 | -0.11458 |
| Pose mAP50-95 | 0.51455 | 0.37569 | -0.13886 |
| PCK@0.05 | 0.66863 | 0.56017 | -0.10846 |
| PCK@0.10 | 0.85248 | 0.80219 | -0.05029 |
| NME | 0.05960 | 0.07635 | +0.01675 |
| fused FP16 latency | 5.695499 ms | 5.331401 ms | -6.39% |

## 自适应 epoch 轨迹

- r1：10e，末四轮 Pose/Box 指标仍上升；
- r2：首 batch 前因 checkpoint 缓存 `criterion` 位于 CPU 而安全失败，无训练结果；
- TDD：新增设备迁移回归测试，设备迁移前丢弃陈旧 `criterion`，21 项聚焦测试通过；
- r3：追加 20e，累计 30e，末五轮 Pose 0.35871 → 0.36314、Box 0.53662 → 0.54278、NME 0.07844 → 0.07807；
- r4：追加最后 20e，累计 50e 硬停止。

## 模块耗时剖析

模块剖析使用 CUDA Event 插桩，绝对值包含 Hook 开销，只解释比例；部署绝对延迟以上表无 Hook benchmark 为准。

| 阶段 | Baseline | FastHead | FastHead 占比 |
|---|---:|---:|---:|
| Backbone | 2.9275 ms | 2.6824 ms | 44.92% |
| Neck | 1.6575 ms | 1.5957 ms | 26.72% |
| Detect + Pose | 2.1431 ms | 1.6933 ms | 28.36% |

FastHead 的 Backbone + Neck 合计 71.64%，为主要耗时区域。Head 参数减少 45.73%，Head 时间只下降约 20.99%，说明 GPU 延迟受访存、Kernel 启动、通道对齐和算子利用率共同限制。

## 精度问题定位

1. 4.0M 级 nano 模型冗余有限，一次删除 24.64% 参数过于激进；
2. Pose/Detect Head 直接承担几何定位和可见性学习，比普通分类头更敏感；
3. 结构改变后仅能迁移同形权重，新增小塔不能完整继承 baseline 表示；
4. 输出级 KD 无法完全补偿中间空间特征与关键点关系；
5. 32 通道候选更小但更慢，证明继续缩 Head 同时触碰精度与硬件效率边界。

## 决策

- 生产部署继续使用 200e baseline + 640 + one-to-one-only + Conv/BN fuse + FP16；
- FastHead 保留为负证据和结构搜索样例；
- 下一轮首先针对 Backbone/Neck 的 PSA、C2f/C2fCIB 与 TensorRT Kernel 融合做训练前真实延迟门禁，再决定是否训练。
