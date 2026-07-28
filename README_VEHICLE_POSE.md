# Vehicle Pose 数据与模型说明

主流程请先看 [README.md](README.md)。本页只补充 CarFusion 数据和
Vehicle Pose 模型约定。

## 1. 任务定义

- 类别：`vehicle`，共 1 类；
- 关键点：14 个；
- 每个关键点格式：`x, y, visibility`；
- 训练同时保留 YOLOv10 one-to-many 和 one-to-one 分支；
- 推理使用 one-to-one NMS-free 分支。

模型配置：

```text
ultralytics/cfg/models/v10/
├── yolov10n-vehicle-pose.yaml
├── yolov10s-vehicle-pose.yaml
└── yolov10l-vehicle-pose.yaml
```

## 2. CarFusion 数据

原始数据需要自行申请授权：
<https://forms.gle/FCUcbt3jD1hB6ja57>。
不得提交 Token、私密下载链接、原始图片或生成标签。

```bash
export CARFUSION_ARCHIVE=/absolute/path/to/CarFusion.merged.tar.gz
export CARFUSION_SHA256="$(sha256sum "$CARFUSION_ARCHIVE" | awk '{print $1}')"
export CARFUSION_EXTRACTED=/absolute/path/to/extracted
scripts/run_carfusion_pipeline.sh
```

若收到 `.00/.01/...` 真正分卷，先按数据提供方说明合并；脚本只读取
`CARFUSION_ARCHIVE` 指向的单个文件。

转换完成后的目录：

```text
datasets/CarFusion/yolo_pose/
├── images/
├── labels/
├── train.txt
├── val.txt
└── test.txt
```

数据配置：
[`ultralytics/cfg/datasets/CarFusion-pose.yaml`](ultralytics/cfg/datasets/CarFusion-pose.yaml)。
数据放在其他位置时，复制该 YAML 并只修改 `path:`；不要改变类别、关键点顺序
或固定切分后再与已有实验比较。

单独质检：

```bash
.venv/bin/python scripts/check_carfusion.py \
  --root datasets/CarFusion/yolo_pose
```

## 3. 训练 n / s / l

```bash
# s 示例；n/l 只替换模型文件和运行名
.venv/bin/yolo pose train \
  model=ultralytics/cfg/models/v10/yolov10s-vehicle-pose.yaml \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 epochs=200 batch=16 device=0 \
  project=vehicle_pose_runs name=yolov10s_baseline
```

显存不足时降低 `batch`，不要降低验证分辨率后直接与 640 baseline 比较。

## 4. 验证

```bash
.venv/bin/yolo pose val \
  model=vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 batch=16 half=True device=0 workers=0
```

完整比较至少保留 Box/Pose mAP、PCK、NME、固定图片可视化和运行配置。

## 5. 剪枝入口

`n / s / l` 使用同一个剪枝脚本，网络由 checkpoint 自动识别：

```bash
.venv/bin/python scripts/prune_vehicle_pose.py \
  --checkpoint vehicle_pose_runs/yolov10s_baseline/weights/best.pt \
  --output outputs/pruning/yolov10s_prune10_init.pt \
  --report reports/pruning/yolov10s_prune10_structure.json \
  --pruning-ratio 0.10 --head-pruning-ratio 0 \
  --round-to 16 --graph-imgsz 64 --device cuda
```

后续恢复、测速和显存测试见
[通用结构化剪枝指南](docs/pruning_distillation.md)。

## 6. 已验证方案边界

P5-Zero 是目前 `n` 上通过精度、速度和显存门禁的成功方案，详见
[README_P5ZERO.md](README_P5ZERO.md)。它不是 `s/l` 的现成结论；
新的 `s/l` checkpoint 必须重新剪枝、恢复和验收。
