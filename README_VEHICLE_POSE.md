# YOLOv10 Vehicle Pose

车辆单类、14×3 关键点的 YOLOv10n 训练与 PyTorch 性能工程。模型保留 YOLOv10 one-to-many/one-to-one 训练语义，推理使用 one-to-one NMS-free 路径。

## 1. 环境

```bash
git clone --branch p5zero-v1 --depth 1 \
  https://github.com/Caius-Lu/yolov10-vehicle-pose.git
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
export CARFUSION_ARCHIVE=/absolute/path/to/CarFusion.merged.tar.gz
export CARFUSION_SHA256="$(sha256sum "$CARFUSION_ARCHIVE" | awk '{print $1}')"
export CARFUSION_EXTRACTED=/absolute/path/to/extracted
scripts/run_carfusion_pipeline.sh
```

等价的分步命令如下：

```bash
.venv/bin/python scripts/prepare_carfusion_coco.py \
  --train-json "$CARFUSION_EXTRACTED/carfusion/annotations/car_keypoints_train.json" \
  --test-json "$CARFUSION_EXTRACTED/carfusion/annotations/car_keypoints_test.json" \
  --output datasets/CarFusion/coco/car_keypoints_all.json
.venv/bin/python scripts/convert_carfusion.py \
  --coco-json datasets/CarFusion/coco/car_keypoints_all.json \
  --images-root "$CARFUSION_EXTRACTED/carfusion" \
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

精度修复候选可用 `--head-pruning-ratio 0` 完全保护 Head。已验证的 Backbone/Neck 20%、Head 0%、`round-to=16` 候选只减少 4.71% 参数，FP16 latency 反而慢 3.89%，因此按训练前门禁停止，未启动正式恢复训练。复现参数见 `configs/pose_prune_repair_gate.yaml`。

## 10. 显式 FastHead 与训练前速度门禁

FastHead 不做不规则后剪枝，而是把 Pose Tower 的 `96×2` 显式缩为
`64×1`，并把 Box Tower 从两层 3×3 缩为一层。候选在读取训练数据前
必须通过参数量、GFLOPs、真实 fused FP16 latency 三个独立门禁：

```bash
.venv/bin/python -m benchmarks.pretrain_architecture_gate \
  --baseline ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml \
  --candidate fasthead=ultralytics/cfg/models/v10/yolov10n-vehicle-pose-fasthead-boxd1.yaml \
  --device cuda:0 --imgsz 640 --warmup 100 --iterations 1000 --rounds 3 \
  --output reports/benchmark/fasthead_gate/reproduce.json
```

RTX 3090 Ti 实测参数减少 24.64%、one-to-one GFLOPs 减少 26.24%、
fused FP16 延迟减少 6.51%，因此允许进入恢复训练。Pose 32×1 因只快
4.53% 被淘汰，证明更少参数不必然更快。

```bash
.venv/bin/python scripts/build_fasthead_checkpoint.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt
WANDB_MODE=disabled .venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_fasthead_kd_smoke.yaml
WANDB_MODE=disabled .venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_fasthead_kd_gate.yaml
```

smoke 只验证 AMP/KD/反向/保存链路，不用小样本 mAP 作精度决策。正式实验使用有界自适应 epoch：10e 和累计 30e 只有在末段 Box/Pose/PCK/NME 仍可信改善时才续训，并在累计 50e 硬停止。对应独立配置为：

```bash
configs/pose_fasthead_kd_gate.yaml
configs/pose_fasthead_kd_adaptive_30e_r3.yaml
configs/pose_fasthead_kd_adaptive_50e_r4.yaml
```

最终完整验证与同口径 640、batch1、one-to-one-only、Conv/BN fuse、FP16 benchmark：

| 指标 | 200e baseline | FastHead 50e |
|---|---:|---:|
| 参数量 | 4,005,986 | 3,018,722 |
| Box mAP50-95 | 0.66761 | 0.55303 |
| Pose mAP50-95 | 0.51455 | 0.37569 |
| PCK@0.05 / PCK@0.10 | 0.66863 / 0.85248 | 0.56017 / 0.80219 |
| NME | 0.05960 | 0.07635 |
| fused FP16 latency | 5.6955 ms | 5.3314 ms |

速度改善 6.39%，但 Pose mAP 损失 0.13886，因此不替换 baseline。模块剖析
显示 FastHead 的 Backbone + Neck 占 71.6% 顶层模块时间；后续因此转向
PSA、C2f/C2fCIB 和 kernel-friendly 串行深度，不继续盲目缩小 Pose Head。
详见 `reports/fasthead/final_50e/final_report.md`。

## 11. Fast / Balanced / Accurate

当前可复用的两类模型为：

- Accurate：原始 200-epoch baseline，Box/Pose 约 `0.665/0.515`；
- Balanced：P5-Zero，延迟下降 `14.83371%`，Box/Pose
  `0.66254/0.49592`，推理峰值显存下降 `5.02307%`。

Balanced 使用同源全模型 KD 和 Pose-only KD 精修，但部署只需最终学生模型。
完整权重准备、训练、验证、paired latency、独立 memory 和测试命令见
[README_P5ZERO.md](README_P5ZERO.md)。仓库不提供会意外启动长训练的一键
脚本，也不提交 P5-Zero 生成权重。

## 12. 总验收

```bash
DEVICE=cuda:0 IMGSZ=640 scripts/validate_project.sh
```

## 常见错误

- `BLOCKED_BY_DATASET_AUTHORIZATION`：历史状态；本次已由用户明确提供授权包，当前检查远端传输/哈希/解压门禁。
- PyTorch 2.6 checkpoint 报 `weights_only`：本分支已包含兼容修复，只加载可信官方/自训练 checkpoint。
- 关键点左右翻转错误：保持 `fliplr=0.0`，完成 100 张编号预览人工确认后再配置 `flip_idx`。
- latency 偏高：确认 `model.model[-1].one2one_only=True`，并区分原始/推理专用路径。
