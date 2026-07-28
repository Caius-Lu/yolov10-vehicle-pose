# 通用结构化剪枝指南

这份文档只回答一个问题：如何给自己的 YOLOv10 Vehicle Pose
`n / s / l` checkpoint 做结构化通道剪枝。

历史实验、曲线和失败原因放在 [reports/](../reports/)；本页只保留可执行流程。

## 1. 输入和输出

你需要准备一个已经训练完成的 Pose baseline：

```text
<baseline>/weights/best.pt
```

脚本从 checkpoint 读取完整网络，所以 `n / s / l` 共用同一个入口：

```text
baseline best.pt
  → 结构化剪枝
  → pruned_init.pt
  → 同源教师恢复蒸馏
  → recovery/weights/best.pt
  → 精度、速度、显存验收
```

不要把检测模型 checkpoint 传给该流程。当前实现要求
`YOLOv10PoseModel`，并保护 Box、Class 和 14×3 Pose 的最终输出维度。

## 2. 推荐起点

| 网络 | 首次剪枝率 | Head 剪枝率 | 通道对齐 |
|---|---:|---:|---:|
| n | 10% | 0 | 16 |
| s | 10% | 0 | 16 |
| l | 10% | 0 | 16 |

这是安全的搜索起点，不是已经验证的最优值。先验证 10%；只有三项门禁通过，
才尝试 15%。Head 对关键点精度敏感，第一轮保持 `0`。

## 3. 一套可复制命令

先定义本次实验路径。下面以 `s` 为例：

```bash
export BASELINE_PT=vehicle_pose_runs/yolov10s_baseline/weights/best.pt
export PRUNED_PT=outputs/pruning/yolov10s_prune10_init.pt
export RUN_NAME=yolov10s_prune10_recovery
export DATA_YAML=ultralytics/cfg/datasets/CarFusion-pose.yaml
```

### 3.1 剪枝

```bash
.venv/bin/python scripts/prune_vehicle_pose.py \
  --checkpoint "$BASELINE_PT" \
  --output "$PRUNED_PT" \
  --report "reports/pruning/${RUN_NAME}_structure.json" \
  --pruning-ratio 0.10 \
  --head-pruning-ratio 0 \
  --round-to 16 \
  --graph-imgsz 64 \
  --device cuda
```

该步骤不读取数据。它会：

1. 加载 checkpoint 中的真实模型；
2. 用 Torch-Pruning 构建依赖图；
3. 物理删除通道，而不是仅把权重置零；
4. 检查参数确实减少；
5. 执行一次 finite forward；
6. 保存结构已改变的 whole-model checkpoint 和 JSON 报告。

`graph-imgsz` 只用于依赖图，不是训练分辨率。
若 `--output` 或 `--report` 已存在，脚本会直接报错，避免覆盖实验。

### 3.2 恢复蒸馏

```bash
WANDB_MODE=disabled \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_recovery.yaml \
  --pruned-weights "$PRUNED_PT" \
  --teacher-weights "$BASELINE_PT" \
  --data "$DATA_YAML" \
  --project vehicle_pose_runs_pruned \
  --name "$RUN_NAME"
```

同一个 baseline 同时承担两个角色：

- 剪枝前的结构来源；
- 冻结的恢复教师。

学生和教师必须来自同一网络大小、同一任务定义、同一数据语义。不要混用：

- `n` 教师 + `s` 学生；
- detection 教师 + pose 学生；
- 14 关键点教师 + 其他关键点定义学生。

只检查链路时：

```bash
WANDB_MODE=disabled \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_recovery.yaml \
  --pruned-weights "$PRUNED_PT" \
  --teacher-weights "$BASELINE_PT" \
  --data "$DATA_YAML" \
  --project vehicle_pose_runs_pruned \
  --name "${RUN_NAME}_smoke" \
  --epochs 1 --batch 2 --fraction 0.002
```

smoke 只能证明代码能运行，不能用于报告精度。

若正式训练被中断，从原 run 的 `last.pt` 恢复 optimizer、EMA 和 epoch：

```bash
WANDB_MODE=disabled \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_recovery.yaml \
  --teacher-weights "$BASELINE_PT" \
  --resume "vehicle_pose_runs_pruned/$RUN_NAME/weights/last.pt"
```

新运行若 `project/name` 已存在会报错，不会静默生成 `name2`。正常新实验请换
`RUN_NAME`；只有继续同一次训练时使用 `--resume`。

## 4. 怎样换成 l

不改 Python 代码，也不改剪枝算法。只换路径：

```bash
export BASELINE_PT=vehicle_pose_runs/yolov10l_baseline/weights/best.pt
export PRUNED_PT=outputs/pruning/yolov10l_prune10_init.pt
export RUN_NAME=yolov10l_prune10_recovery
```

然后重新执行第 3 节。显存不足只调整恢复训练的 `--batch`：

```bash
... --batch 4
```

不要因为 `l` 更大就直接提高剪枝率。更大的模型可能有更多冗余，但也必须由本机
精度和延迟结果证明。

## 5. 三项门禁

### 精度

使用 baseline 和 candidate 完全相同的数据 YAML、`imgsz` 和 validator。
两者都必须实际运行验证，不能只验证 candidate。
至少记录：

- Box mAP50-95；
- Pose mAP50-95；
- PCK@0.05、PCK@0.10；
- NME。

### 速度

必须成对比较：

```bash
.venv/bin/python -m benchmarks.benchmark_deploy_fp16 \
  --checkpoint "$BASELINE_PT" \
  --candidate-checkpoint \
    "vehicle_pose_runs_pruned/$RUN_NAME/weights/best.pt" \
  --device cuda:0 --imgsz 640 \
  --warmup 100 --iterations 1000 --rounds 3 \
  --output "reports/pruning/${RUN_NAME}_latency.json"
```

不要用参数量或 GFLOPs 代替真实延迟。

### 显存

baseline 和 candidate 分别运行
`python -m benchmarks.measure_deploy_memory`，每个模型使用一个独立进程。不要在
同一进程先后加载两个模型。

## 6. 决策

建议先为自己的 baseline 明确阈值。若沿用本项目发布口径：

- paired fused-FP16 latency 至少下降 10%；
- Box 和 Pose mAP50-95 的绝对下降分别不超过 0.02；
- candidate 独立进程推理峰值显存低于 baseline。

结果处理：

| 结果 | 动作 |
|---|---|
| 三项都通过 | 保存证据，再测试 15% 或进入交付 |
| 精度失败 | 降低剪枝率；不要只靠无限增加 epochs |
| 速度失败 | 停止该结构；训练不会改变执行图 |
| 显存失败 | 检查独立进程口径；确认后停止 |

每个实验使用不同的输出路径和 `name`，禁止覆盖 baseline 或已有报告。

## 7. 常用参数

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 决定实际网络，支持本项目的 n/s/l Pose checkpoint |
| `--pruning-ratio` | Backbone/Neck 剪枝率 |
| `--head-pruning-ratio` | Detect/Pose hidden tower 剪枝率；`0` 表示保护 |
| `--round-to` | 输出通道对齐倍数 |
| `--model-indices` | 只允许指定 backbone/neck 层作为剪枝根 |
| `--deployment-only` | 只用于 one-to-one 部署图的实验选项 |
| `--importance` | `magnitude` 或经过稀疏训练的 `bn_scale` |

不确定时只使用第 3 节给出的四个剪枝参数。

## 8. 当前证据边界

- 通用通道剪枝代码已经覆盖网络大小无关的 checkpoint 加载路径；
- `s/l` 模型 YAML 和复用式恢复 CLI 已加入测试；
- 已发布的最好结果 P5-Zero 来自 `n`，不是 `s/l` 实测结果；
- 过去 `n` 上的通道剪枝曾出现参数下降但速度/精度门禁失败，详见
  [实验归档](../reports/EXPERIMENT_SUMMARY_20260724.md)。

因此，对新的 `s/l` 网络必须重新跑完整三门禁，不能引用 `n` 的指标。
