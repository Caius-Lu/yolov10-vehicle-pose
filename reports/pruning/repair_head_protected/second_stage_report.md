# 第二阶段精度修复：Head 保护候选门禁报告

## 问题定位

第一阶段统一 15% 剪枝的验证指标在 epoch 14–31 达峰，继续训练到 100 epoch 没有恢复；因此“增加 epoch”被排除。53 个保守剪枝根中 36 个位于 Detect/Pose Head，Head 又占总参数的 53.9%。统一剪枝很可能直接损伤姿态隐藏塔和召回能力。

## 最小修复

- Backbone/Neck 根层剪枝率：20%
- Detect/Pose Head 隐藏根层：0%，完全保护
- 最终输出层：继续保护
- 通道对齐：8 改为 16
- 独立 checkpoint、结构报告和 benchmark 目录

实现通过 TDD：新增测试在实现前得到 3 个预期失败；实现后聚焦测试 `15 passed in 0.85s`。

## 训练前门禁

| 指标 | 200e baseline | Head 保护候选 | 变化 |
|---|---:|---:|---:|
| 参数量 | 4,005,986 | 3,817,154 | -4.71% |
| fused 参数量 | 3,991,106 | 3,803,714 | -4.70% |
| GFLOPs | 13.5 | 12.9 | -4.44% |
| fused FP16 latency | 5.7010 ms | 5.9229 ms | **+3.89%（更慢）** |

计时口径与第一阶段一致：RTX 3090 Ti、batch=1、640、one-to-one-only、Conv/BN fuse、FP16、100 warmup、3×1000 iterations。

## 决策

候选同时未达到“参数至少减少 10%”和“FP16 latency 至少降低 10%”两项训练前门禁。正式恢复训练无法改变执行图和通道形状，因此即使精度可能改善，也不能修复已经失败的 latency 目标。按预设停止规则，不启动小 epoch 或长 epoch 训练，不覆盖第一阶段和 baseline。

这次验证揭示的是结构性冲突：完全保护占参数 53.9% 的 Head 可以降低精度风险，但剩余可剪空间不足；在 RTX 3090 Ti 上，4.7% 的结构缩减还落入更慢的 kernel/调度组合。下一步若继续研究，应切换到目标部署后端的 latency-aware channel search、TensorRT engine profiling 或重新设计显式窄宽学生，而不是继续在当前规则上调整 epoch。
