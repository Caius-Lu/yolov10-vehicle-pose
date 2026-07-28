# YOLOv10 Vehicle Pose

车辆检测与 14 关键点姿态估计，支持 YOLOv10 `n / s / l`，并提供结构化通道
剪枝、恢复蒸馏、精度验证、速度和显存测试。

> 本仓库只提交代码、配置和实验记录。CarFusion 数据及训练生成的权重不进入 Git。

## 先选你要做的事

| 目标 | 从这里开始 |
|---|---|
| 直接训练 `n / s / l` | [训练基线](#训练基线) |
| 给自己的 `s / l` checkpoint 剪枝 | [通用剪枝：适用于-n--s--l](#通用剪枝适用于-n--s--l) |
| 复现已经通过门禁的 `n` 压缩方案 | [README_P5ZERO.md](README_P5ZERO.md) |
| 查看历史实验与失败方案 | [reports/](reports/) |

## 安装

```bash
git clone --branch feat/vehicle-pose \
  https://github.com/Caius-Lu/yolov10-vehicle-pose.git
cd yolov10-vehicle-pose

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-build-isolation -e . pytest
```

做结构化剪枝时再安装：

```bash
.venv/bin/python -m pip install --no-deps -r requirements-pruning.txt
```

## 准备数据

取得 CarFusion 授权数据后：

```bash
export CARFUSION_ARCHIVE=/absolute/path/to/CarFusion.merged.tar.gz
export CARFUSION_SHA256="$(sha256sum "$CARFUSION_ARCHIVE" | awk '{print $1}')"
export CARFUSION_EXTRACTED=/absolute/path/to/extracted
scripts/run_carfusion_pipeline.sh
```

默认数据配置：
[`ultralytics/cfg/datasets/CarFusion-pose.yaml`](ultralytics/cfg/datasets/CarFusion-pose.yaml)。
若数据位置不同，只修改数据 YAML 的 `path:`。

## 训练基线

模型配置：

| 网络 | 配置 |
|---|---|
| YOLOv10n | `ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml` |
| YOLOv10s | `ultralytics/cfg/models/v10/yolov10s-vehicle-pose.yaml` |
| YOLOv10l | `ultralytics/cfg/models/v10/yolov10l-vehicle-pose.yaml` |

以 `s` 为例：

```bash
.venv/bin/yolo pose train \
  model=ultralytics/cfg/models/v10/yolov10s-vehicle-pose.yaml \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 epochs=200 batch=16 device=0 \
  project=vehicle_pose_runs name=yolov10s_baseline
```

`l` 只需把模型 YAML 和 `name` 改成 `l`；显存不足时降低 `batch`。训练结束后，
后续所有剪枝命令都使用该次运行的 `weights/best.pt`。

## 通用剪枝：适用于 n / s / l

关键点只有一个：

> 网络大小由 `--checkpoint` 内保存的真实模型决定，不需要在剪枝脚本里再填写
> `n`、`s` 或 `l`。

下面以 YOLOv10s 为例。

### 1. 生成剪枝学生

第一次先用 10%，并保护 Detect/Pose head：

```bash
.venv/bin/python scripts/prune_vehicle_pose.py \
  --checkpoint vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --output outputs/pruning/yolov10s_prune10_init.pt \
  --report reports/pruning/yolov10s_prune10_structure.json \
  --pruning-ratio 0.10 \
  --head-pruning-ratio 0 \
  --round-to 16 \
  --graph-imgsz 64 \
  --device cuda
```

说明：

- `--pruning-ratio 0.10`：Backbone/Neck 剪 10%；
- `--head-pruning-ratio 0`：先不剪高敏感的检测和姿态头；
- `--round-to 16`：通道保持硬件友好对齐；
- `--graph-imgsz 64`：只用于构建依赖图，不改变 640 训练分辨率。

如果换成 `l`，只改三个路径中的 `yolov10s` 为自己的 `yolov10l` 路径。

### 2. 恢复精度

同一份配置可以用于任何大小：

```bash
WANDB_MODE=disabled \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_recovery.yaml \
  --pruned-weights outputs/pruning/yolov10s_prune10_init.pt \
  --teacher-weights vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --data ultralytics/cfg/datasets/CarFusion-pose.yaml \
  --project vehicle_pose_runs_pruned \
  --name yolov10s_prune10_recovery
```

教师必须是该学生剪枝前的同源 baseline，不能用 `n` 教师训练 `s/l` 学生。

### 3. 验证精度

先验证 baseline，再用完全相同的参数验证 candidate：

```bash
.venv/bin/yolo pose val \
  model=vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 batch=16 half=True device=0 workers=0 \
  project=vehicle_pose_runs_val name=yolov10s_baseline_val

.venv/bin/yolo pose val \
  model=vehicle_pose_runs_pruned/yolov10s_prune10_recovery/weights/best.pt \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 batch=16 half=True device=0 workers=0 \
  project=vehicle_pose_runs_val name=yolov10s_prune10_val
```

### 4. 比较速度

baseline 和 candidate 必须在同一台机器、同一进程口径下成对测试：

```bash
.venv/bin/python -m benchmarks.benchmark_deploy_fp16 \
  --checkpoint vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --candidate-checkpoint \
    vehicle_pose_runs_pruned/yolov10s_prune10_recovery/weights/best.pt \
  --device cuda:0 --imgsz 640 --warmup 100 --iterations 1000 --rounds 3 \
  --output reports/pruning/yolov10s_prune10_latency.json
```

### 5. 比较显存

两个模型必须分别启动独立 Python 进程：

```bash
.venv/bin/python -m benchmarks.measure_deploy_memory \
  --checkpoint vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --device cuda:0 --imgsz 640 \
  --output reports/pruning/yolov10s_baseline_memory.json

.venv/bin/python -m benchmarks.measure_deploy_memory \
  --checkpoint vehicle_pose_runs_pruned/yolov10s_prune10_recovery/weights/best.pt \
  --device cuda:0 --imgsz 640 \
  --output reports/pruning/yolov10s_prune10_memory.json
```

训练中断时，使用原教师和该 run 的 `last.pt` 真正续训：

```bash
WANDB_MODE=disabled \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_recovery.yaml \
  --teacher-weights vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --resume vehicle_pose_runs_pruned/yolov10s_prune10_recovery/weights/last.pt
```

只有精度、速度、显存都满足项目要求，才继续尝试 15%。不要一次启动多组长训练。
完整参数解释和决策规则见
[docs/pruning_distillation.md](docs/pruning_distillation.md)。

## 已验证结果

当前成功结果是 `n` 上的 P5-Zero：

| 指标 | 原始 baseline | P5-Zero |
|---|---:|---:|
| fused FP16 latency | 5.94537 ms | 5.06345 ms |
| Box/Pose mAP50-95 | 约 0.665/0.515 | 0.66254/0.49592 |
| fused 参数量 | 3,992,210 | 3,387,794 |
| 推理峰值 CUDA memory | 72,237,568 B | 68,609,024 B |

P5-Zero 的结构和指标只在当前 `n` checkpoint 上验证过，不能直接宣称在 `s/l`
上也有同样收益。`s/l` 应按上面的通用流程重新建立各自 baseline 和门禁。

## 测试

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_vehicle_pose.py \
  tests/test_pose_pruning_distillation.py \
  tests/test_pose_pruning_pipeline.py

scripts/validate_project.sh
```

## 文档

- [通用结构化剪枝与恢复蒸馏](docs/pruning_distillation.md)
- [P5-Zero 精确复现](README_P5ZERO.md)
- [CarFusion 与 Vehicle Pose 说明](README_VEHICLE_POSE.md)
- [实验执行日志](reports/execution_journal.md)
