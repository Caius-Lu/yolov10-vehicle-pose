# YOLOv10 Vehicle Pose

车辆单类、14×3 关键点的 YOLOv10n 训练与 PyTorch 性能工程。模型保留 YOLOv10 one-to-many/one-to-one 训练语义，推理使用 one-to-one NMS-free 路径。

## 1. 环境

```bash
git clone https://github.com/THU-MIG/yolov10.git yolov10-vehicle-pose
cd yolov10-vehicle-pose
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-build-isolation -e . pytest
```

本项目已验证的远端兼容环境见 `reports/bootstrap/01_bootstrap_report.md`。

## 2. 权重

```bash
mkdir -p weights
curl -L --fail --retry 3 -o weights/yolov10n.pt \
  https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
sha256sum weights/yolov10n.pt
```

预期 SHA256：`61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e`。

## 3. CarFusion 授权数据

原始流程要求填写官方申请表 <https://forms.gle/FCUcbt3jD1hB6ja57>。本次使用用户明确提供的 OpenDataLab CarFusion 包；归档 SHA-256、两层解压结构和传输状态记录在 `reports/data/02_carfusion_download_report.md`。不得提交 token 或私密链接。

## 4. 数据转换与质检

远端首次完整执行可直接运行：

```bash
scripts/run_carfusion_pipeline.sh
```

等价的分步命令如下：

```bash
.venv/bin/python scripts/prepare_carfusion_coco.py \
  --train-json /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion/annotations/car_keypoints_train.json \
  --test-json /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion/annotations/car_keypoints_test.json \
  --output datasets/CarFusion/coco/car_keypoints_all.json
.venv/bin/python scripts/convert_carfusion.py \
  --coco-json datasets/CarFusion/coco/car_keypoints_all.json \
  --images-root /home/caius/projects/yolopose/data/CarFusion/extracted/carfusion \
  --output datasets/CarFusion/yolo_pose --seed 0 \
  --invalid-bbox-policy drop --fixed-test-group car_penn
.venv/bin/python scripts/check_carfusion.py --root datasets/CarFusion/yolo_pose
.venv/bin/python scripts/visualize_carfusion.py \
  --root datasets/CarFusion/yolo_pose --output reports/data/preview --limit 100 --seed 0
.venv/bin/python scripts/create_carfusion_lists.py \
  --root datasets/CarFusion/yolo_pose \
  --fixed-eval-output reports/training/fixed_eval_images.txt --seed 0
```

相机 1/2 按路口合并为同一 group；官方 `car_penn` 保留为 test，seed 0 使用 `car_fifth` 为 val。739 个零 bbox 仅在显式 `drop` 策略下丢弃并记录。`flip_idx` 人工确认前，所有训练命令保持 `fliplr=0.0`。

## 5. 训练

```bash
scripts/train_vehicle_pose_smoke.sh
scripts/train_vehicle_pose_overfit.sh
scripts/train_vehicle_pose_full.sh
scripts/resume_vehicle_pose.sh vehicle_pose_runs/baseline/weights/last.pt
```

必须依次通过 smoke、小集过拟合，再启动 full training。

## 6. 验证与关键点指标

```bash
.venv/bin/yolo pose val model=vehicle_pose_runs/baseline/weights/best.pt \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml device=0
.venv/bin/python scripts/evaluate_vehicle_pose.py --pairs-json reports/training/matched_pose_pairs.json
```

PCK/NME 按 GT bbox 长边归一化；输出可见/遮挡、每关键点及车辆尺度分组。

## 7. 推理

```python
from ultralytics import YOLOv10

model = YOLOv10("vehicle_pose_runs/baseline/weights/best.pt", task="pose")
results = model.predict("image.jpg", imgsz=640, device=0)
```

## 8. PyTorch benchmark 与 profiler

```bash
scripts/benchmark_torch.sh --checkpoint vehicle_pose_runs/baseline/weights/best.pt
scripts/benchmark_end_to_end.sh \
  --checkpoint vehicle_pose_runs/baseline/weights/best.pt \
  --source-list datasets/CarFusion/yolo_pose/test.txt
scripts/profile_torch.sh --checkpoint vehicle_pose_runs/baseline/weights/best.pt
```

正式结果采用 batch=1、warmup=100、iterations=1000、独立三轮，并分别报告 model-only 与 end-to-end。

## 9. 结构化剪枝与恢复蒸馏（独立实验）

安装可选依赖后，先生成物理改宽的学生模型，再启动恢复蒸馏：

```bash
.venv/bin/python -m pip install --no-deps -r requirements-pruning.txt
.venv/bin/python scripts/prune_vehicle_pose.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  --output outputs/pruning/yolov10n_pose_pruned15_init_20260723.pt \
  --report reports/pruning/prune15_structure_20260723.json \
  --graph-imgsz 64 --pruning-ratio 0.15 --round-to 8 --device cuda
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_prune_distill_100e.yaml
```

验证必须与 200-epoch baseline 使用相同数据、640 输入和指标定义；延迟必须用 batch=1、one-to-one-only、Conv/BN fuse、FP16 的同机口径横向比较。已经完成的 100-epoch 实验减少 16.36% 参数，但 Pose mAP50-95 下降 0.09923，FP16 latency 仅改善 0.34%，因此不进入部署推荐。详细结果、失败恢复和 whole-model checkpoint 安全边界见 [docs/pruning_distillation.md](docs/pruning_distillation.md)。

## 10. Fast / Balanced / Accurate

真实 baseline 训练、正式 benchmark 与消融完成前不会生成虚假版本。后续复现入口预留为 `scripts/reproduce_fast.sh`、`reproduce_balanced.sh`、`reproduce_accurate.sh`。

## 11. 总验收

```bash
DEVICE=cuda:0 IMGSZ=640 scripts/validate_project.sh
```

## 常见错误

- `BLOCKED_BY_DATASET_AUTHORIZATION`：历史状态；本次已由用户明确提供授权包，当前检查远端传输/哈希/解压门禁。
- PyTorch 2.6 checkpoint 报 `weights_only`：本分支已包含兼容修复，只加载可信官方/自训练 checkpoint。
- 关键点左右翻转错误：保持 `fliplr=0.0`，完成 100 张编号预览人工确认后再配置 `flip_idx`。
- latency 偏高：确认 `model.model[-1].one2one_only=True`，并区分原始/推理专用路径。
