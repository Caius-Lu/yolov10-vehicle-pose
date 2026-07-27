# Vehicle Pose 部署分支物理裁剪报告（2026-07-27）

## 最终结论

交付一个独立的 **deployment-only** checkpoint：物理删除推理
`one2one_only=True` 时不会执行的 one-to-many box/class/pose towers。
该方案的收益是模型参数、checkpoint 文件和加载内存压缩；不宣称显著加速。

| 指标 | Baseline | 分支裁剪 | 变化 |
|---|---:|---:|---:|
| 参数量 | 4,005,986 | 2,925,857 | -26.96% |
| checkpoint 大小 | 8,422,070 B | 6,182,717 B | -26.59% |
| fused 后参数量 | 3,992,210 | 2,914,065 | -27.01% |
| fused FP16 参数/缓冲存储 | 7.98 MB | 5.83 MB | -27.01% |
| CUDA 模型加载 allocated | 57.72 MB | 51.25 MB | -11.22% |
| 含 640 输入 allocated | 60.18 MB | 53.70 MB | -10.76% |
| 推理峰值 allocated | 72.24 MB | 65.74 MB | -8.99% |
| paired fused FP16 延迟 | 5.8359 ms | 5.8247 ms | -0.19% |

在 640×640 随机输入上，裁剪前后 one-to-one dense output：

- shape 均为 `[1, 47, 8400]`；
- 最大绝对误差：0；
- 平均绝对误差：0；
- `torch.equal`：true。

完整 3,606 图 CarFusion 验证：

| 指标 | 分支裁剪模型 |
|---|---:|
| Box mAP50-95 | 0.665 |
| Pose mAP50-95 | 0.514 |
| PCK@0.05 | 0.66769 |
| PCK@0.10 | 0.85236 |
| NME | 0.05970 |

结果与既有 fused-FP16 baseline 保持一致。分支裁剪没有改变实际执行的
one-to-one graph 或权重，因此无需恢复训练。

删除的模块为 `cv2`、`cv3`、`cv4`，保留 `one2one_cv2`、
`one2one_cv3`、`one2one_cv4`。模型被标记为 deployment-only；
若切到训练模式或尝试运行 one-to-many 分支，会显式抛出错误。

checkpoint 在远端生成于：

```text
outputs/pruning/deployment_branch_pruned/baseline200_one2one_only.pt
```

权重文件按项目约定不入 Git；生成脚本、结构报告和 paired benchmark 入库。

## 本轮被淘汰的 latency-aware 通道剪枝

| 候选 | 参数变化 | paired fused FP16 延迟变化 | 结论 |
|---|---:|---:|---|
| Backbone/Neck 15%、Head 10%、round32 | -29.80% | **慢 0.44%** | 不训练 |
| 定点 7 层 25%、Head 保护、round16 | -1.98% | **慢 0.27%** | 不训练 |
| 定点 7 层 35%、Head 保护、round16 | -3.03% | **慢 0.93%** | 不训练 |

`round_to=32` 将多组 10%/15%/20% 请求折叠成相同的 2,812,386 参数
结构；因此只 benchmark 一份等价 checkpoint。所有候选都在 RTX 3090 Ti、
640、batch=1、one-to-one-only、Conv/BN fuse、FP16、100 warmup、
3×1000 iterations 的交替 paired 口径下评估。

## TDD 与证据

- paired checkpoint benchmark：先因缺少 `summarize_paired_latency`
  产生预期导入失败，实现后转绿；
- 模型索引白名单：先得到 2 个预期失败，实现后转绿；
- 部署分支裁剪：先因缺少裁剪函数产生预期导入失败，实现后转绿；
- 最终相关回归：30/30 通过。

原始证据：

- `structure_report.json`
- `paired_fp16.json`
- `baseline_memory.json`
- `candidate_memory.json`
- `fullval_vehicle_pose_metrics.json`
- `../latency_search/*.json`
