# YOLOv10 Vehicle Pose 结构化剪枝与恢复蒸馏

本方案是一个**独立实验分支**：从已验收的 640 baseline 复制学生模型，做保守的通道结构化剪枝，再用原 baseline 作为冻结教师进行短周期恢复训练。它不会修改 baseline 代码、权重、数据集或既有报告，也不会把“蒸馏提升精度”误当成“部署一定加速”。最终是否采用，只由同一硬件上的精度与实测延迟门禁决定。

## 为什么是结构化剪枝

权重置零或非结构化稀疏通常不会直接减少普通 CUDA dense kernel 的张量尺寸，因此未必产生部署收益。本实现使用 [Torch-Pruning DepGraph](https://github.com/VainF/Torch-Pruning) 追踪耦合通道，并物理删除 Conv/BN 相关维度。剪枝后参数形状改变，checkpoint 必须保存完整模型对象；加载陌生来源的 whole-model checkpoint 存在反序列化风险，只加载本项目自己生成且来源可信的文件。

这里保护检测与姿态头的最终输出层，固定：

- DFL 框输出：`4 × reg_max = 64` 通道；
- 分类输出：`nc = 1` 通道；
- 关键点输出：`14 × 3 = 42` 通道；
- one-to-many 与 one-to-one 两套分支都受保护。

仅选择无静态 split 属性的保守 Conv 作为剪枝根节点，依赖图负责同步调整下游输入、BN 和 depthwise 耦合通道。默认只剪 15%，并将通道对齐到 8，兼顾 Tensor Core/部署后端的友好尺寸。

## 实现结构

```mermaid
flowchart LR
    A["640 baseline best.pt"] --> B["复制为学生"]
    A --> E["冻结全宽教师"]
    B --> C["DepGraph 结构化剪枝 10%~15%"]
    C --> D["物理参数减少 + 单次前向门禁"]
    D --> F["640 同分辨率输出蒸馏"]
    E --> F
    F --> G["精度 + 可视化门禁"]
    G --> H["one2one + Conv/BN fuse + FP16"]
    H --> I["同机 latency benchmark"]
    I -->|通过| J["导出部署模型"]
    I -->|不通过| K["止损，保留 baseline"]
```

相关文件：

- `ultralytics/models/yolov10/prune_distill.py`：剪枝、输出蒸馏及 whole-model checkpoint；
- `scripts/prune_vehicle_pose.py`：数据无关的结构剪枝入口；
- `scripts/train_pose_pruned_distillation.py`：加载已变形学生并恢复训练；
- `configs/pose_prune_distill.yaml`：20-epoch 研究配置；
- `configs/pose_prune_distill_100e.yaml`：已经远程验证的 100-epoch、batch=64 配置；
- `tests/test_pose_pruning_distillation.py`：结构与梯度边界；
- `tests/test_pose_pruning_pipeline.py`：路径和配置安全边界。

## 运行步骤

安装可选依赖：

```bash
.venv/bin/python -m pip install --no-deps -r requirements-pruning.txt
```

第一步只构建依赖图并生成初始剪枝模型，不读取数据集：

```bash
.venv/bin/python scripts/prune_vehicle_pose.py \
  --checkpoint outputs/baseline_weights/best.pt \
  --output outputs/pruning/yolov10n_pose_pruned15_init.pt \
  --report reports/pruning/prune15_structure.json \
  --graph-imgsz 64 \
  --pruning-ratio 0.15 \
  --round-to 8 \
  --device cuda
```

`graph-imgsz=64` 只用于追踪全卷积网络的通道依赖和做一次前向检查，不代表训练或验证分辨率。训练、教师和最终评估仍为 640。默认不做额外 MAC 统计，避免在构图阶段重复执行高内存 profile；如确有需要可显式加 `--measure-macs`。参数减少是结构门禁，真实速度必须用现有 benchmark 测量。

第二步做恢复蒸馏。短周期研究使用原配置；复现本次完整实验使用 100-epoch 配置：

```bash
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_distill_100e.yaml
```

学生继续接受真实 bbox、类别、14×3 关键点监督；教师只提供同分辨率 raw-output 软目标：DFL 分布 KL、分类/visibility 概率 MSE、关键点 XY SmoothL1。这样允许学生隐藏通道宽度变化，而最终输出语义保持一致。教师不注册为学生子模块，不进入 optimizer、EMA 或 checkpoint。

只做链路验证时，必须使用显式的小范围参数，并使用独立输出名：

```bash
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_distill.yaml \
  --epochs 1 --batch 2 --fraction 0.002 \
  --name carfusion_prune15_kd_smoke
```

该命令只是训练链路 smoke，不产生可用于决策的精度结论。

## 分阶段实验与止损门禁

不要一次批量跑 10%/15%/20% 三组。按下面顺序推进，每一组通过后再启动下一组：

1. **10% 剪枝**：验证结构、恢复能力和真实 latency 是否有方向性收益；
2. **15% 剪枝**：默认推荐候选，完整执行 20 epoch 恢复与验证；
3. **20% 剪枝**：仅当 15% 的精度余量和延迟收益都明显时考虑。

15% 候选的建议通过条件：

| 门禁 | 条件 |
|---|---:|
| 物理参数量 | 至少减少 10% |
| FP16 fused model-only latency | 相对现有 FP16 fused baseline 至少降低 10% |
| Pose mAP50-95 | 下降不超过 0.005 |
| PCK@0.05 | 下降不超过 0.005 |
| PCK@0.10 | 下降不超过 0.002 |
| NME | 增加不超过 0.001 |
| 可视化 | 固定 100 图无系统性错位、漏点或置信度退化 |

任何一项失败，都不覆盖 baseline，也不进入部署推荐。尤其是：即使蒸馏把精度恢复回来，如果同硬件、同 batch、同预处理和同后处理条件下 latency 没有改善，就判定该剪枝方案没有部署收益并停止。

## 评估与部署顺序

1. 对恢复后的 `best.pt` 跑完整 3,606 图验证，生成 Box/Pose mAP、PCK、NME；
2. 用与 baseline 完全相同的固定 100 图执行可视化门禁；
3. 先测 PyTorch FP32，再测 one-to-one-only + fuse + FP16；
4. 只有 PyTorch 门禁通过，才导出 ONNX/TensorRT，并重新核对输出和精度；
5. 把结构报告、训练配置、checkpoint SHA-256、指标和延迟结果存入新的 `reports/pruning/<experiment>/`，禁止覆盖已有结果。

## 已完成的 100-epoch 远程验证

2026-07-23 至 2026-07-24 在 `wsl-dev` 的 RTX 3090 Ti 上完成了 15% 结构化剪枝与 batch=64、100 epoch 恢复蒸馏，训练用时 3.81 小时。依赖图实际选取 53 个保守根层，73 个 Conv 只允许被依赖传播调整，18 个最终输出层受保护。训练未发生 OOM 或非有限 loss。

| 指标 | 200e baseline | 剪枝 + KD 100e | 变化 |
|---|---:|---:|---:|
| 参数量 | 4,005,986 | 3,350,666 | -16.36% |
| checkpoint | 8.42 MB | 7.06 MB | -16.20% |
| Box mAP50-95 | 0.66761 | 0.57041 | -0.09720 |
| Pose mAP50-95 | 0.51455 | 0.41532 | -0.09923 |
| PCK@0.05 | 0.66863 | 0.60902 | -0.05960 |
| PCK@0.10 | 0.85248 | 0.81319 | -0.03930 |
| NME | 0.05960 | 0.07291 | +0.01331 |
| fused FP16 latency | 5.7010 ms | 5.6819 ms | -0.34% |

计时严格使用同一设备、batch=1、640、one-to-one-only、Conv/BN fuse、FP16、100 次 warmup、1000 次计时和 3 轮平均。剪枝候选自身从 FP32 切换到 FP16 的加速不等于剪枝收益；部署决策看的是两个模型各自 fused FP16 的横向比较。

结果只通过参数量门禁，精度与 latency 门禁失败，因此不能替换 baseline。完整原始证据在：

- `reports/pruning/r2/prune15_structure_20260723.json`
- `reports/pruning/r2/results.csv`
- `reports/pruning/r2/vehicle_pose_metrics.json`
- `reports/pruning/r2/comparison.json`
- `reports/benchmark/prune15_kd100_r2/`

训练曲线显示 Box/Pose/PCK 分别在 epoch 14、24、30–31 左右达到最佳，之后训练 loss 仍下降而验证指标不再改善。因此“再增加 epoch”不是当前证据支持的修复。第二阶段应优先保护 Pose/Detect head 的敏感根层、使用分层剪枝率和硬件友好的通道对齐，并在独立配置和输出目录中先通过小规模精度与延迟门禁。

## 失败恢复与安全边界

1. 每个实验使用唯一 `name`、PID、日志和输出目录，存活时禁止重复启动；
2. OOM 时只降低 batch 并新建 run，不复用半写入目录；
3. whole-model checkpoint 只能加载本仓库自产且来源可信的文件；
4. baseline 权重、数据 YAML、转换后数据和既有报告全部只读；
5. 若参数减少但 latency 无改善，立即判定部署收益失败；若精度恢复但 latency 失败，也不得推荐上线。
