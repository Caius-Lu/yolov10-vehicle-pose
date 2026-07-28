# P5-Zero：YOLOv10 Vehicle Pose 速度—精度平衡方案

P5-Zero 是本仓库当前通过发布门禁的 PyTorch 结构压缩方案。它不使用
TensorRT，只移除 backbone P5 与 neck P5 两个低分辨率 stage（C2f/C2fCIB）
的内部 bottleneck，完整保留 P3、P4 和 Box/Class/Pose 预测塔。

本文面向团队内需要精确复现、评估或改写该方案的算法工程师，覆盖环境准备、外部制品、
权重迁移、两阶段蒸馏、完整验证、延迟/显存测试、验收门禁和故障排查。
精确复现必须拥有授权数据、原始 baseline 和同源教师；Git checkout 本身只
闭环代码、配置、测试与证据，不承诺提供这些外部制品。

## 1. 结论与适用边界

验证环境：RTX 3090 Ti、PyTorch `2.6.0+cu124`、输入 `640×640`、
batch=1、one-to-one-only、Conv/BN fuse、FP16。

| 指标 | 原始 baseline | P5-Zero | 变化 |
|---|---:|---:|---:|
| paired fused-FP16 latency | 5.94537 ms | 5.06345 ms | **-14.83371%** |
| Box mAP50-95 | 约 0.665 | **0.66254** | 发布门槛 0.64761，通过 |
| Pose mAP50-95 | 约 0.515 | **0.49592** | 发布门槛 0.49455，通过 |
| fused 参数量 | 3,992,210 | 3,387,794 | **-15.13989%** |
| 模型驻留 CUDA memory | 57,721,856 B | 54,093,312 B | **-6.28626%** |
| 推理峰值 CUDA memory | 72,237,568 B | 68,609,024 B | **-5.02307%** |

最佳 checkpoint 的完整验证记录为 Box/Pose `0.66254/0.49592`；独立的最终
fused-FP16 全量复验为 `0.662/0.496`，PCK@0.05 为 `0.64366`，
PCK@0.10 为 `0.82799`，NME 为 `0.06729`。

这里的“通过”表示达到本项目发布门槛，不表示相对原始 baseline 完全无精度
下降。精度门槛来自架构搜索参考 checkpoint 的 Box/Pose
`0.66761/0.51455` 各减去绝对值 `0.02`，即
`0.64761/0.49455`；表中的约 `0.665/0.515` 是原始 baseline 的另一轮完整
部署验证显示值，不能反向用于重算门槛。两组数值引用的是同一个原始 baseline
checkpoint：前者来自建立门槛的标准精度验证，后者来自最终 fused/FP16
部署复验；paired speed baseline 也必须使用该 checkpoint。Pose 余量只有
`0.00137`，更换 GPU、
PyTorch/CUDA 版本、数据版本或验证实现后必须重新执行三门禁：

1. paired fused-FP16 latency 下降至少 10%；
2. Box/Pose mAP50-95 不低于 `0.64761/0.49455`；
3. 候选的独立进程 CUDA 推理峰值严格低于 baseline。

## 2. 方案结构

模型配置：
[`ultralytics/cfg/models/v10/yolov10n-vehicle-pose-litepsa-c2f-p4d1-p5zero-head96.yaml`](ultralytics/cfg/models/v10/yolov10n-vehicle-pose-litepsa-c2f-p4d1-p5zero-head96.yaml)

完整血缘为：

```text
原始 YOLOv10n Vehicle Pose baseline
  └─ LitePSA + C2f-P4D1 同源恢复模型
       └─ P5-Zero 初始化（本节的两处直接结构差异）
            └─ 全模型同源 KD
                 └─ Pose-only KD 精修（最终学生）
```

因此，“只改变两处”是相对直接父模型 LitePSA+C2f-P4D1 而言。相对最终测速
使用的原始 baseline，P5-Zero 还继承了父模型的 LitePSA 和 P4 C2f 减深，
不能把整个模型描述成只对原始 baseline 删除两个模块。

| 位置 | 层号 | 改动 |
|---|---:|---|
| backbone P5 | 8 | C2f 内部 bottleneck 数量降为 0 |
| neck P5 | 22 | C2fCIB 内部 bottleneck 数量降为 0 |

这里用“P5 模块”统称两个低分辨率 stage；layer 8 的具体类型是 C2f，
layer 22 是 C2fCIB。P3/P4 高分辨率路径、LitePSA、Box/Class/Pose 塔保持
完整。权重迁移不是简单
截断：605 个同形张量直接复制，layer 8/22 的两个输出投影使用 C2f 投影折叠，
missing/unexpected key 均为 0。

## 3. 仓库内容与权重边界

GitHub 中包含：

- 模型 YAML、训练配置和迁移实现；
- 单元测试、benchmark 与显存测量脚本；
- JSON 实测证据、实验复盘、PPT 和演讲稿。

GitHub **不包含** P5-Zero 生成权重、CarFusion 原始数据和运行目录。请使用
自己的授权数据与 checkpoint 制品存储。复现涉及以下权重：

| 名称 | 默认路径 | 获取方式 |
|---|---|---|
| 原始 baseline | `vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt` | 外部精确制品；用于教师血缘与最终配对测速 |
| 同源教师 | `vehicle_pose_runs_backbone_neck/carfusion_yolov10n_litepsa_c2f_p4d1_kd_adaptive_60e_r3/weights/best.pt` | 既有 LitePSA+C2f-P4D1 恢复制品 |
| P5-Zero 初始化 | `outputs/kernel_depth/litepsa_c2f_p4d1_p5zero_head96_folded_init.pt` | 第 6 节生成 |
| 全模型 KD best | `vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_kd_gate12e/weights/best.pt` | 第 7 节生成 |
| 最终 P5-Zero best | `vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt` | 第 8 节生成 |

发布实现快照为 Git commit `80282c31`。外部制品当前保存在团队受控制品存储，
没有公开下载 URL；向模型制品负责人申请时应按下列路径和 SHA256 精确核对，
不要用同名文件替换：

| 制品 | 大小 | SHA256 |
|---|---:|---|
| 原始 baseline | 8,422,070 B | `1338a583a930fe7e85c4ea7f4a9422eb28488a41435376a42ff154991f61f066` |
| 同源教师 | 8,025,577 B | `1b1cbc9fa13a80bbac711a95da4407f0350a93890486252b1688e738c7f8a65f` |
| 最终 P5-Zero best | 19,443,886 B | `cc0f9c934d0cd6b77058a1b8e9807fe9688cf7d3f291ba87ee976899299eb8a9` |

```bash
sha256sum \
  vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  vehicle_pose_runs_backbone_neck/carfusion_yolov10n_litepsa_c2f_p4d1_kd_adaptive_60e_r3/weights/best.pt \
  vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt
```

当前 checkout 可能含历史 `outputs/baseline_weights/best.pt`，但它不是本次
paired gate 所引用路径。除非哈希、结构和完整验证指标均与精确 baseline
一致，否则只能用于接口 smoke，不能替代发布门禁制品。

同源教师不能随意替换成原始 baseline：两者上游图和特征分布不同，直接替换会
降低迁移与特征蒸馏的可比性。若没有现成同源教师，可按第 6 节先重建它；该
过程仍需要原始 baseline 外部制品。

## 4. 环境准备

推荐 Linux 或 WSL2，并使用支持 CUDA FP16 的 NVIDIA GPU。

```bash
git clone --branch p5zero-v1 --depth 1 \
  https://github.com/Caius-Lu/yolov10-vehicle-pose.git
cd yolov10-vehicle-pose

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-build-isolation -e . pytest
```

`requirements.txt` 保留项目基础兼容版本；发布测速机实际使用
PyTorch `2.6.0+cu124`。要复现实测环境，在安装基础依赖后按目标 CUDA
重新安装匹配的 PyTorch/torchvision，例如：

```bash
.venv/bin/python -m pip install --upgrade \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

性能比较必须在相同 commit 和相同 PyTorch/CUDA 环境运行。记录：

```bash
git rev-parse HEAD
nvidia-smi
.venv/bin/python -m torch.utils.collect_env
```

文档中的 JSON 查看命令使用 `jq`；Ubuntu/WSL 可通过
`sudo apt-get install jq` 安装。`jq` 只用于查看和判定，不影响训练。

检查运行环境：

```bash
.venv/bin/python - <<'PY'
import torch
from ultralytics import YOLOv10

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("YOLOv10 import: PASS")
PY
```

## 5. 数据准备

CarFusion 需要自行取得授权，申请入口与原始目录说明见
[README_VEHICLE_POSE.md](README_VEHICLE_POSE.md#3-carfusion-授权数据)。流水线
接受两个环境变量：

```bash
export CARFUSION_ARCHIVE=/absolute/path/to/CarFusion.merged.tar.gz
export CARFUSION_SHA256="$(sha256sum "$CARFUSION_ARCHIVE" | awk '{print $1}')"
export CARFUSION_EXTRACTED=/absolute/path/to/extracted
scripts/run_carfusion_pipeline.sh
```

脚本只读取 `CARFUSION_ARCHIVE` 指向的单个文件，不会自动搜索或拼接相邻的
`.01/.02`。若数据提供方交付的确实是分卷压缩包，应先在输出目录之外按其说明
合并为一个归档，再设置 `CARFUSION_ARCHIVE`；非默认归档还需设置对应的
`CARFUSION_SHA256`，不能跳过完整性校验。

解压后预期至少存在：

```text
$CARFUSION_EXTRACTED/carfusion/
├── annotations/car_keypoints_train.json
├── annotations/car_keypoints_test.json
└── <图像目录>
```

转换后的默认结构为：

```text
datasets/CarFusion/yolo_pose/
├── images/
├── labels/
├── train.txt
├── val.txt
└── test.txt
```

流水线会依次解压、合并 COCO 标注、转换 YOLO Pose、质检、生成预览和固定
train/val/test 列表；之后可再次独立执行
`.venv/bin/python scripts/check_carfusion.py --root datasets/CarFusion/yolo_pose`。

训练配置默认引用版本化文件：
[`ultralytics/cfg/datasets/CarFusion-pose.yaml`](ultralytics/cfg/datasets/CarFusion-pose.yaml)。
如果数据不在仓库内默认目录：

```bash
mkdir -p configs/local
cp ultralytics/cfg/datasets/CarFusion-pose.yaml \
  configs/local/CarFusion-pose.yaml
```

编辑副本的 `path:`。只运行已有教师的 P5-Zero 两阶段时，修改两个 P5-Zero
配置；若还要从原始 baseline 重建同源教师，则以下五个配置的 `data:` 都必须
指向同一副本：

```text
configs/pose_litepsa_c2f_kd_20e.yaml
configs/pose_litepsa_c2f_kd_adaptive_40e_r2.yaml
configs/pose_litepsa_c2f_kd_adaptive_60e_r3.yaml
configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml
configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml
```

不要修改 train/val/test 列表、类别或
关键点定义，否则不能与已发布结果直接比较。`configs/local/` 应保持本地使用，
不要提交包含机器路径的文件。

WSL 中建议把数据放在 ext4，而不是 `/mnt/<drive>`，避免 `p9_client_rpc`
文件系统等待和 dataloader 抖动。

## 6. 准备同源教师并生成 P5-Zero 初始化

### 6.1 可选：从原始 baseline 重建同源教师

已有精确同源教师时跳过本节。否则先生成 LitePSA+C2f-P4D1 初始化：

```bash
PYTHONPATH=. .venv/bin/python scripts/build_fasthead_checkpoint.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  --model ultralytics/cfg/models/v10/yolov10n-vehicle-pose-litepsa-c2f-p4d1.yaml \
  --output outputs/backbone_neck/litepsa_c2f_p4d1_init.pt \
  --report reports/backbone_neck/litepsa_c2f_p4d1_transfer_reproduce.json \
  --device cpu
```

然后顺序执行三个有界恢复配置：

```bash
for cfg in \
  configs/pose_litepsa_c2f_kd_20e.yaml \
  configs/pose_litepsa_c2f_kd_adaptive_40e_r2.yaml \
  configs/pose_litepsa_c2f_kd_adaptive_60e_r3.yaml
do
  WANDB_MODE=disabled PYTHONPATH=. \
  .venv/bin/python scripts/train_pose_pruned_distillation.py --config "$cfg"
done
```

三段输出路径由各配置的 `project`/`name` 定义，后一段显式引用前一段 best。
最终同源教师应位于第 3 节表中的默认路径。该血缘共 60 个追加 epoch，属于
完整重建成本；如果只有最终 P5-Zero 推理需求，应从受控制品存储取得教师或
最终学生，而不是重新训练。

### 6.2 生成 P5-Zero 初始化 checkpoint

使用同源教师构建学生，并保存迁移报告：

```bash
PYTHONPATH=. .venv/bin/python scripts/build_fasthead_checkpoint.py \
  --checkpoint vehicle_pose_runs_backbone_neck/carfusion_yolov10n_litepsa_c2f_p4d1_kd_adaptive_60e_r3/weights/best.pt \
  --model ultralytics/cfg/models/v10/yolov10n-vehicle-pose-litepsa-c2f-p4d1-p5zero-head96.yaml \
  --output outputs/kernel_depth/litepsa_c2f_p4d1_p5zero_head96_folded_init.pt \
  --report reports/kernel_depth/p5zero_transfer_reproduce.json \
  --device cpu \
  --smoke-imgsz 64
```

检查报告：

```bash
jq '{
  direct_tensor_count,
  folded_tensor_count,
  missing_key_count,
  unexpected_key_count,
  student_parameters,
  smoke_finite
}' reports/kernel_depth/p5zero_transfer_reproduce.json
```

预期关键值：

```json
{
  "direct_tensor_count": 605,
  "folded_tensor_count": 2,
  "missing_key_count": 0,
  "unexpected_key_count": 0,
  "student_parameters": 3396962,
  "smoke_finite": true
}
```

若这些结构计数不一致，不要启动长训练；先核对教师 checkpoint、模型 YAML 和
当前代码提交。

## 7. 阶段一：全模型同源蒸馏

配置：
[`configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml`](configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml)

输入为第 6 节初始化 checkpoint 和同源教师；输出目录由配置固定为：

```text
vehicle_pose_runs_kernel_depth/
└── litepsa_c2f_p4d1_p5zero_kd_gate12e/
    ├── pruning_distillation.yaml
    ├── results.csv
    └── weights/{best,last}.pt
```

```bash
WANDB_MODE=disabled \
PYTHONPATH=. \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml
```

这一阶段训练全模型，使用：

- detection output KD：权重 1.0；
- pose output KD：权重 1.25；
- layer 8/22 feature KD：权重 0.10；
- 总 distillation weight：0.12，temperature：2.0。

记录实验计划为 12 epochs，但已发布运行完成第 4 epoch 验证后，在第 5
epoch 开始前手动停止，并保留第 2 epoch best：

| epoch | Box mAP50-95 | Pose mAP50-95 |
|---:|---:|---:|
| 1 | 0.60299 | 0.43531 |
| 2 | **0.66245** | **0.49369** |
| 3 | 0.65026 | 0.49116 |
| 4 | 0.64727 | 0.48279 |

不要通过修改配置中的总 epochs 来模拟这次停止，因为总 epochs 会影响学习率
调度。精确历史路径使用人工门禁，没有自动 early-stop 参数：确认第 4 epoch
验证和 `weights/best.pt` 已落盘后，可在下一轮开始前使用一次 `Ctrl-C`，不要
发送强制 kill。更稳妥的标准复现方式是跑满计划，并保留 `best.pt`、
`last.pt` 与周期 checkpoint 作为候选；它验证方案，不保证逐 epoch 轨迹与
历史人工停止完全一致。

## 8. 阶段二：Pose-only 蒸馏精修

配置：
[`configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml`](configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml)

输入路径在配置的 `pruned_weights:` 中显式指向阶段一 best；输出目录为：

```text
vehicle_pose_runs_kernel_depth/
└── litepsa_c2f_p4d1_p5zero_pose_refine_6e/
    ├── pruning_distillation.yaml
    ├── results.csv
    └── weights/{best,last}.pt
```

```bash
WANDB_MODE=disabled \
PYTHONPATH=. \
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_litepsa_c2f_p4d1_p5zero_pose_refine_6e.yaml
```

这一阶段从阶段一 best 启动，只训练：

```text
model.23.cv4.*
model.23.one2one_cv4.*
```

即 48 个 Pose tower 张量。检测输出 KD 和 feature KD 均关闭，保留 pose
output KD（pose weight 1.5，总 distillation weight 0.15）。冻结的
BatchNorm 强制保持 eval，避免 running mean/variance 漂移。

已发布运行第 1 epoch 达到：

- Box mAP50-95：`0.66254`
- Pose mAP50-95：`0.49592`
- PCK@0.05：`0.64399`
- NME：`0.06714`
- 训练显存：约 4.47 GB

门禁通过后在第 2 epoch 早期人工终止，最终使用第 1 epoch
`weights/best.pt`。标准复现同样可以跑满 6 epochs，但不要直接相信训练器的
聚合 fitness 或把 `last.pt` 当作发布模型。对 `best.pt`、`last.pt` 和
`epoch*.pt` 逐个执行第 9 节的完整验证：先过滤
Box/Pose 均不低于 `0.64761/0.49455` 的候选，再选 Pose 最高者；Pose 相同时
依次选择 Box 更高、paired latency 更低者。任何候选都未同时过双门槛则 HOLD。
已发布制品就是上表 SHA256 对应的第 1 epoch `best.pt`。推理部署只加载最终
学生 checkpoint，不需要教师模型或蒸馏代码。

## 9. 完整精度验证

```bash
.venv/bin/yolo pose val \
  model=vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt \
  data=ultralytics/cfg/datasets/CarFusion-pose.yaml \
  imgsz=640 \
  batch=64 \
  half=True \
  device=0 \
  workers=0 \
  project=vehicle_pose_runs_kernel_depth \
  name=p5zero_final_fullval_reproduce \
  exist_ok=True
```

必须使用完整验证集 3,606 图、11,569 instances。验证 batch=64 只提高吞吐，
不属于 batch=1 延迟口径；显存不足可降低验证 batch，不要改变数据列表和
`imgsz=640`。`workers=0` 更慢，但在 WSL/p9 环境中更稳；稳定的 Linux
ext4 环境可以提高 workers。

不要把训练日志中的 loss、部分数据集结果或最佳单轮输出替代完整验证。最终
half-precision 复验应接近 Box/Pose `0.662/0.496`。部署 benchmark 脚本会
显式执行 Conv/BN fuse；验证命令是否显示 fused 以本次运行的模型 summary
为准，不要仅凭 `half=True` 推断。

自定义 validator 会在验证输出目录同时生成：

```text
vehicle_pose_runs_kernel_depth/p5zero_final_fullval_reproduce/
├── matched_pose_pairs.json
└── vehicle_pose_metrics.json
```

因此 PCK/NME 可以直接读取：

```bash
jq '{
  pck_005: ."pck@0.05",
  pck_010: ."pck@0.10",
  nme
}' \
  vehicle_pose_runs_kernel_depth/p5zero_final_fullval_reproduce/vehicle_pose_metrics.json
```

也可从本次验证生成的 pairs 独立重算：

```bash
.venv/bin/python scripts/evaluate_vehicle_pose.py \
  --pairs-json vehicle_pose_runs_kernel_depth/p5zero_final_fullval_reproduce/matched_pose_pairs.json \
  --json-output vehicle_pose_runs_kernel_depth/p5zero_final_fullval_reproduce/vehicle_pose_metrics_recomputed.json \
  --csv-output vehicle_pose_runs_kernel_depth/p5zero_final_fullval_reproduce/per_keypoint_metrics.csv
```

## 10. Paired fused-FP16 延迟

延迟必须在同一进程中交替测 baseline/candidate，不得拼接两次独立运行的
绝对毫秒数：

```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_deploy_fp16.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  --candidate-checkpoint vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt \
  --imgsz 640 \
  --device cuda:0 \
  --warmup 100 \
  --iterations 1000 \
  --rounds 5 \
  --output reports/kernel_depth/p5zero_final_vs_original_paired_fp16_reproduce.json
```

读取结果：

```bash
jq '.summary' \
  reports/kernel_depth/p5zero_final_vs_original_paired_fp16_reproduce.json

jq -e '.environment.precision == "fused FP16"
  and .environment.batch == 1
  and .environment.path == "one-to-one-only"
  and .summary.latency_reduction_pct >= 10' \
  reports/kernel_depth/p5zero_final_vs_original_paired_fp16_reproduce.json
```

发布运行结果为 `5.94537→5.06345 ms`，下降 `14.83371%`。不同 GPU 时钟、
温度和系统负载会改变绝对毫秒数；验收只使用同一次 paired run 的相对收益。

## 11. 独立进程 CUDA memory

baseline 和 candidate 必须分别启动独立进程：

```bash
PYTHONPATH=. .venv/bin/python benchmarks/measure_deploy_memory.py \
  --checkpoint vehicle_pose_runs/carfusion_baseline_yolov10n_e_drive_20260718/weights/best.pt \
  --imgsz 640 \
  --device cuda:0 \
  --warmup 100 \
  --output reports/kernel_depth/original_baseline_memory_reproduce.json

PYTHONPATH=. .venv/bin/python benchmarks/measure_deploy_memory.py \
  --checkpoint vehicle_pose_runs_kernel_depth/litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt \
  --imgsz 640 \
  --device cuda:0 \
  --warmup 100 \
  --output reports/kernel_depth/p5zero_final_memory_reproduce.json
```

门禁唯一裁决字段是 `cuda_peak_during_inference_bytes`，要求 candidate 严格
小于 baseline；`cuda_model_delta_bytes` 与参数/Buffer 存储作为诊断项。
重点比较：

```bash
jq '{
  parameters_fused,
  parameter_and_buffer_storage_bytes,
  cuda_model_delta_bytes,
  cuda_peak_during_inference_bytes,
  dense_finite
}' reports/kernel_depth/*_memory_reproduce.json

jq -s -e \
  '.[1].cuda_peak_during_inference_bytes < .[0].cuda_peak_during_inference_bytes' \
  reports/kernel_depth/original_baseline_memory_reproduce.json \
  reports/kernel_depth/p5zero_final_memory_reproduce.json
```

不要用训练显存代替部署显存，也不要在同一 Python 进程连续加载两个模型后比较。
发布门禁采用上面的严格 `<` 比较；若重复测量，应让 baseline 和 candidate
使用相同的独立进程策略并保存每次原始 JSON，不能只挑选有利的一次。发布结果
下降 5.02307%，原始 JSON 均保留。

## 12. 测试

先运行 P5-Zero 相关测试：

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_fasthead_transfer.py \
  tests/test_lite_backbone_blocks.py \
  tests/test_pose_pruning_pipeline.py
```

发布运行结果为 `33 passed`。

随后运行项目综合验收：

```bash
DEVICE=cuda:0 IMGSZ=640 scripts/validate_project.sh
```

综合脚本包括 Vehicle Pose、CarFusion、指标、benchmark、蒸馏/冻结行为和官方
YOLOv10 Detect 回归。它需要转换后的 CarFusion 数据与官方
`weights/yolov10n.pt`；缺失时应报告环境阻塞，而不是把未执行步骤标记为通过。
最近一次 P5 聚焦集合为 33/33；综合脚本首段 31/31 与部署输出一致性通过，但
随后 CarFusion 数据质量步骤在 WSL p9 等待中达到 180 秒外部超时，因此该次
不能宣称“综合脚本全部通过”。

## 13. 推理

```python
from ultralytics import YOLOv10

model = YOLOv10(
    "vehicle_pose_runs_kernel_depth/"
    "litepsa_c2f_p4d1_p5zero_pose_refine_6e/weights/best.pt",
    task="pose",
)
results = model.predict(
    source="image.jpg",
    imgsz=640,
    device=0,
    half=True,
)
```

教师模型只用于训练。最终 checkpoint 是完整学生模型，可独立加载和推理。

## 14. 常见问题

### 找不到 `pruned_weights` 或 `teacher_weights`

所有配置路径相对仓库根目录解析。先按第 3 节放置教师权重，再执行第 6 节生成
初始化 checkpoint。不要把某个不同结构 checkpoint 仅重命名后使用。

### W&B 在训练开始前报 project 路径错误

复现实验不依赖 W&B，使用：

```bash
export WANDB_MODE=disabled
```

### 验证卡在字体下载

Ultralytics 可能首次尝试联网获取 Arial。无网络环境可预置兼容字体：

```bash
mkdir -p ~/.config/yolov10
cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  ~/.config/yolov10/Arial.ttf
```

不同 Ultralytics 配置目录可能不同；以启动日志显示的设置目录为准。

### WSL 长时间停在 `p9_client_rpc`

这是 `/mnt/<drive>` 文件访问等待，不是 CUDA OOM。将数据和运行目录放到 WSL
ext4，降低 workers，避免训练期间频繁跨 Windows 文件系统读取。

### CUDA OOM

先减小训练 `batch`，不要改变 `imgsz=640`。命令行可覆盖：

```bash
.venv/bin/python scripts/train_pose_pruned_distillation.py \
  --config configs/pose_litepsa_c2f_p4d1_p5zero_kd_gate12e.yaml \
  --batch 32
```

batch 改变后属于新的训练条件，必须重新跑完整三门禁。

### `torch.cuda.is_available()` 为 false 或 CUDA 版本不匹配

```bash
nvidia-smi
.venv/bin/python -m torch.utils.collect_env
.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

确认 NVIDIA 驱动支持目标 CUDA wheel；不要用 CPU 环境生成或比较 GPU
latency/memory 结论。

### 找不到 `yolo` 或 `jq`

```bash
.venv/bin/python -m pip install --no-build-isolation -e .
sudo apt-get update
sudo apt-get install jq
```

本文命令以 editable install 生成的 `.venv/bin/yolo` 为准。

### 数据 YAML 报 path 不存在或验证集为空

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("ultralytics/cfg/datasets/CarFusion-pose.yaml").read_text())
root = Path(cfg["path"])
print("dataset root:", root.resolve(), "exists:", root.exists())
for split in ("train", "val", "test"):
    item = root / cfg[split]
    print(split, item, "exists:", item.exists())
PY
```

不要让不同实验悄悄指向不同 split 文件。

### 迁移计数变化、输出非 finite 或 checkpoint 结构不匹配

先确认 commit、教师路径和模型 YAML；再运行第 12 节聚焦测试和第 6 节
`smoke_finite` 检查。预期学生任务为单类 vehicle、`kpt_shape=[14,3]`。
missing/unexpected key 非 0、迁移计数漂移或 NaN/Inf 都是训练前硬失败，不能
通过继续训练掩盖。

### Pose 指标在门槛附近波动

当前余量小，裁决规则是：固定同一 checkpoint、数据清单和验证实现重新完整
验证；任一次权威复验低于门槛都先标记 HOLD，排查环境/数据差异，不挑选多次
运行中的最高值。需要重新训练时应作为新实验记录 seed、配置和全部结果。

### 为什么不使用 TensorRT

本方案的目标是验证可复用的 PyTorch 结构压缩收益。TensorRT、量化或其他部署
后端可能带来额外收益，但不属于本次结果，不能与这里的 14.83371% 混算。

## 15. 证据与进一步阅读

- [最终结果报告](reports/kernel_depth/p5zero_balance_result_20260727.md)
- [全部压缩实验复盘](reports/kernel_depth/experiment_review_20260727.md)
- [最终 paired latency JSON](reports/kernel_depth/p5zero_final_vs_original_paired_fp16.json)
- [baseline memory JSON](reports/kernel_depth/original_baseline_memory_final.json)
- [P5-Zero memory JSON](reports/kernel_depth/p5zero_final_memory.json)
- [迁移报告](reports/kernel_depth/p5zero_transfer.json)
- [三页技术评审 PPT](slide-deck/vehicle-pose-compression-review/vehicle-pose-compression-review.pptx)
- [完整演讲稿](slide-deck/vehicle-pose-compression-review/speaker-script.md)

## 16. 发布检查表

- [ ] 使用正确的原始 baseline 和同源教师；
- [ ] 迁移报告 missing/unexpected key 为 0，smoke finite；
- [ ] 阶段一和阶段二保存了配置、CSV、best/last；
- [ ] 完整 3,606 图精度达到 Box/Pose 门槛；
- [ ] 同进程 paired latency 至少下降 10%；
- [ ] 两个独立进程的 CUDA memory 已记录且 candidate 不退化；
- [ ] P5-Zero 聚焦测试和项目综合验收已运行；
- [ ] JSON、日志、停止原因和环境版本已归档；
- [ ] 未提交 CarFusion 数据、生成权重、Token 或私密链接。
