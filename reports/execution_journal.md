# YOLOv10 Vehicle Pose 执行日志

本文件记录真实执行过程，供复现与部门分享使用。时间均为 Asia/Shanghai。

## 2026-07-16：项目启动

- 阅读 `yolov10_vehicle_pose_codex_prompts/00_README_使用顺序.md` 及 01–09 阶段说明。
- 在本地 `/Users/caius/Documents/yolopose/yolov10-vehicle-pose` 克隆官方仓库。
- 上游地址：`https://github.com/THU-MIG/yolov10.git`。
- 固定上游 commit：`453c6e38a51e9d1d5a2aa5fb7f1014a711913397`。
- 创建分支：`feat/vehicle-pose`。
- 创建 `external/`、`datasets/`、`weights/`、`reports/`、`scripts/`、`benchmarks/`、`docs/`。
- 将工程同步到远端 `/home/caius/projects/yolopose`。
- 远端：Ubuntu 22.04 / WSL2，RTX 3090 Ti 24 GB，驱动 560.94，磁盘可用约 728 GB。
- 远端全局 PyTorch：2.6.0+cu124，CUDA 可用。
- 官方 `requirements.txt` 固定 PyTorch 2.0.1；依赖源实测速率约 22 KB/s。为避免降级已验证 CUDA 栈和长时间重复下载，改为项目 `.venv` 通过 `--system-site-packages` 复用远端 PyTorch，再以 editable/no-deps 安装当前仓库。此偏差将在 bootstrap 报告保留。

### 已执行命令（脱敏）

```bash
git clone https://github.com/THU-MIG/yolov10.git yolov10-vehicle-pose
git switch -c feat/vehicle-pose
mkdir -p external datasets weights reports/{bootstrap,design,data,training,benchmark,optimization} scripts benchmarks docs
rsync -az --exclude .DS_Store /Users/caius/Documents/yolopose/ ubuntu-ts:/home/caius/projects/yolopose/
python3 -m venv --clear --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -e .
```

## 状态约定

- `PASS`：命令真实执行并通过。
- `FAIL`：命令真实执行但失败，报告保留错误。
- `BLOCKED`：缺少授权数据或外部前置条件，未伪造替代结果。

## 2026-07-17：用户提供 CarFusion 数据并恢复真实数据阶段

- 用户明确提供授权数据路径：`/Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion`。
- 按安全探查流程先读取元数据，未直接全量载入或解压。
- 盘点结果：503 个文件、34,451,319,446 字节；主体归档 34,123,960,720 字节，另有 500 张 sample JPEG。
- `file` 与解压后头部检查确认主体是 gzip 压缩 POSIX tar，顶层为 `CarFusion/`。
- 本地主体归档 SHA-256：`34ae638404d724a50b744dc7cba359b4f38cc336d9c45e5f0cea6a996fe90eb4`。
- 远端 `/home/caius` 所在磁盘解压前可用 727 GiB。
- 启动带 partial 的 rsync，目标为 `/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/`。
- Tailscale 显示本机与服务器为局域网 direct；额外测试 `192.168.68.79:2255` 未获得更高吞吐，因此保留原 rsync 会话，避免丢弃 partial 进度。
- 分享底稿重构为 Codex 工程实践主线，补充角色分工、故障纠正、证据链、提示词模板和现场演示建议。

### 已执行命令（脱敏）

```bash
python3 scripts/probe_carfusion_package.py /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion
file /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/CarFusion.tar.gz.00
shasum -a 256 /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/CarFusion.tar.gz.00
ssh ubuntu-ts 'df -h /home/caius'
rsync -a --partial --progress /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion/ ubuntu-ts:/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/
```

### 归档内标注的流式审计与 TDD 适配

- 识别出外层 `CarFusion.tar.gz.00` 包含内层 `CarFusion/carfusion.tar`，不能按单层归档处理。
- 不展开图片，仅流式读取内层开头的 train/test COCO JSON。
- 实际标注：32,948 张图、63,173 个车辆实例，所有实例都是 14×3；visibility 仅为 0/1/2。
- 发现 739 个 `[0,0,0,0]` bbox，增加显式 `invalid_bbox_policy=drop`，保留 source ID 和 drop 记录；默认严格模式仍报错。
- 将成对相机目录归并为路口 group，固定官方 `car_penn` 为 test，seed 0 选择 `car_fifth` 为 val，其余为 train。
- 增加官方 train/test COCO 合并器、完整质量统计与 100 图编号预览器。
- TDD 红灯包括缺少合并器、drop 后质量计数不一致、缺少分布字段、缺少预览器；修复后 10 项数据管线测试通过，四个脚本合计分支覆盖率 84%。
- 与 Vehicle Pose 核心、Detect 回归和指标测试合并执行：20 个聚焦测试全部通过（1.13 秒）。
- 补齐 PCK/NME 的 GT—预测框 IoU≥0.5 确定性一对一匹配；Validator 自动输出 `matched_pose_pairs.json` 与 `vehicle_pose_metrics.json`，并将可用标量写回验证结果。
- 新增配对纯函数测试后，聚焦测试提升到 21 项；重新运行 4+2 fixture 的完整 CUDA epoch，训练、两次验证、checkpoint 和新的指标文件导出均通过。随机 fixture 没有 IoU≥0.5 配对，因此 PCK/NME 为 `null`，未伪造数值。
- 补充 batch=1 的真实 JPEG 路径到 `Results` 的端到端 benchmark，wall time 与 preprocess/inference/postprocess 分量分开记录；新增 2 项工具测试与 CLI smoke。
- 一次单文件 rsync 把两个文件误放到远端工程根目录；发现后先同步到正确目录，再只删除这两个刚创建的精确副本。没有覆盖用户文件。
- 新增固定 smoke 128/32、overfit 32、fixed-eval 100 清单及 SHA-256；训练不再依赖 `fraction` 隐式抽样。
- 新增训练 manifest，在每个入口运行前记录 Git 状态、PyTorch/CUDA/GPU、模型/数据配置、权重、conversion/split/subset summary 的 SHA-256。
- 启动远端后台接力 PID 48704：只等待 rsync 将 partial 原子改名为最终归档，随后执行 SHA-256 门禁和 `run_carfusion_pipeline.sh`。不会自动越过真实 smoke/overfit 训练门禁。
- 尝试申请局域网只读 HTTP 测速时，安全审查指出会扩大私有数据暴露面并拒绝；未绕过，继续使用点对点加密 rsync。

## 2026-07-18：校验门禁拦截截断归档并恢复续传

- 状态复核发现真实 CarFusion 训练尚未启动；服务器上没有真实数据训练进程，也没有真实数据生成的 checkpoint 或指标文件。
- 首次传输在远程文件达到 14,903,312,384 / 34,123,960,720 字节时中断；原后台接力检测到 SHA-256 为 `0ff82abd2abbfdb6671e676ce5e94c0e498ccf673aedbd40b926e6267c4f0d45`，与期望值不符后停止。该门禁避免了截断归档被解压并进入训练。
- 远端现有的 `fixture_smoke/one_epoch/weights/{best,last}.pt`、`results.csv` 和 `vehicle_pose_metrics.json` 均来自 4+2 张生成 fixture，不作为 CarFusion 真实结果。
- 本机系统 rsync 不支持 `--append-verify`，改用兼容的 `--append` 从断点续传，并保留传输完成后的全文件 SHA-256 门禁。
- 续传恢复时进度约 14.94 / 34.12 GB（44%），吞吐约 3–5 MB/s。
- 初次重建远端接力进程 PID 50398；随后将其替换为 PID 50630。新接力等待文件精确达到 34,123,960,720 字节后，再运行校验、双层解压、转换、质量检查、预览与固定清单生成；所有门禁通过后自动启动 `carfusion_smoke_20260718`（3 epoch，固定 128/32 图）。Overfit 和 200 epoch baseline 不自动越过 smoke 门禁。

### 已执行命令（脱敏）

```bash
rsync -a --append --partial --progress /path/to/CarFusion.tar.gz.00 ubuntu-ts:/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/
ssh ubuntu-ts 'nohup bash -lc "等待精确文件大小后运行 run_carfusion_pipeline.sh" ... &'
```

### 上传完成后的远程可用性异常

- 受管 rsync 会话确认主体归档达到 34,123,960,720 / 34,123,960,720 字节（100%），上传进程正常结束。
- 随后通过 Tailscale 地址和局域网地址分别检查远程状态，两条路径均在 SSH banner 阶段超时；Tailscale ping 仍为 direct，约 32 ms，说明主机网络在线但 SSH 服务暂时不可用或严重阻塞。
- 在恢复 SSH 可观测性前，不把“上传完成”记作“SHA-256 通过、解压完成或真实训练完成”；真实 CarFusion checkpoint 与指标仍标记为未确认。

## 2026-07-19：baseline、视觉门禁、耗时优化与分享材料收口

- 后续统一使用 `wsl-dev` 连接；数据在 Windows E 盘完成解压、转换和真实质量检查。
- CarFusion smoke 3/3、32 图 overfit 100/100、baseline 200/200 均完成。baseline 在 epoch 176 遭遇 WSL 重启和 E 盘临时不可见，核对挂载与 checkpoint 后从 `last.pt` 续跑，没有启动重复任务。
- 最终验证集 3,606 图、11,569 实例：Box mAP50-95 0.665，Pose mAP50-95 0.515，PCK@0.05 0.6686，PCK@0.10 0.8525，NME 0.05960。
- 指标后执行 100 图视觉门禁。Validator 图正常而 raw Predictor 宽图框异常；先写失败测试，再确认 `YOLOv10PosePredictor.postprocess` 缺少 `xywh2xyxy`。最小修复后 Vehicle Pose 回归 9/9 通过并重做 100 图检查。
- 数据质检确认索引 13 在 62,434 个转换实例中 visibility 全为 0；系统接口保持 14×3，但只对 13 个有效点声明监督和精度，`flip_idx` 继续为空。
- 固定 checkpoint、固定 100 JPEG、3×1000 次端到端 benchmark：`/mnt/e` 平均 60.10 ms / P95 109.25 ms；相同字节放入 WSL ext4 后为 16.63 ms / 18.62 ms，平均延迟下降 72.3%，吞吐提升 3.61×。
- 新增独立 `stage_carfusion_to_wsl.sh` 与逐文件 SHA-256 校验器；微型符号链接夹具在 Mac 和 wsl-dev 均通过，不修改原 baseline 路径和输出。
- 完成 Codex 主线的图文分享底稿与 14 页 PPT；PPT 使用真实训练曲线、修复前后同图和优化图表，逐页渲染检查并通过画布溢出检测。

### 网络耗时优化与蒸馏小实验

- 按 TDD 新增 7 项蒸馏测试：先记录模块不存在和 checkpoint 会复制教师的失败，再实现并修复；与车辆姿态回归合计 16/16 通过。
- 新增完全 opt-in 的分辨率感知特征蒸馏：冻结 640 教师，512 学生继续接受真实框/关键点监督，并对齐 head 前 P3/P4/P5 特征；教师不进入学生参数、优化器、EMA 或 checkpoint。
- 使用 baseline `best.pt` 同时初始化教师和学生，完整训练集运行 3 epoch、batch 32。best 为 epoch 2：Box mAP50-95 0.633、Pose mAP50-95 0.474、PCK@0.05 0.63550、PCK@0.10 0.83490、NME 0.06634。
- 同一 baseline 权重直接在 512 验证为 PCK@0.05 0.63235；短蒸馏有小幅恢复，但相对 640 baseline 仍低约 3.31 个百分点，未宣称满足低损精度目标。
- 同进程 AB/BA 配对 benchmark 发现 512 在 RTX 3090 Ti、batch 1 的 PyTorch eager 下没有 latency 收益：one-to-one-only 为 7.001 ms，对应 640 的 6.905 ms。当前 4M 小模型受 kernel launch/低利用率限制。
- 转而验证可落地路径：保留 640，one-to-one-only + Conv/BN fuse + FP16。最终采用 AB/BA 交错的 4×1000 次纯网络前向，由 7.147 ms 降至 5.727 ms，延迟下降 19.88%，加速 1.248×。
- 完整 3,606 图 FP16 门禁：Box mAP50-95 0.665、Pose mAP50-95 0.514、PCK@0.05 0.66763、PCK@0.10 0.85238、NME 0.05969；相对 FP32 的 PCK@0.05 仅下降 0.00100。

### 结构化剪枝 + 恢复蒸馏代码交付（未运行完整训练）

- 用户要求在现有 FP16/fuse 方案上补充可落地的剪枝蒸馏代码，但先不批量运行远程实验；随后允许远程静默执行轻量验证。
- 按 TDD 先增加剪枝配置、最终头保护、物理参数减少、教师梯度隔离、whole-model 序列化、输出路径与同分辨率配置测试。首次远程执行分别真实失败于缺少模块和缺少脚本，保留为红灯证据；实现后新增测试 11/11 通过。
- 新增依赖图结构化通道剪枝、640 同分辨率 raw-output 蒸馏、独立 YAML 与训练入口。baseline checkpoint、既有训练代码和数据均未修改。
- 首次远程结构 smoke 同时残留了依赖安装解析和两个 CPU 构图进程，实际 CPU/内存占用超出轻量边界，随即停止；未启动数据集训练。由此将依赖图输入默认改为 64，并把额外 MAC 统计改为显式选项，真实速度继续由独立 benchmark 判断。
- 为该资源修复新增测试，先观察到 `PruningConfig` 缺少 `measure_macs` 的红灯，再实施最小修改；最终远程聚焦测试 11/11 通过（0.82 秒），并确认没有遗留剪枝/安装进程。
- 随后仅用单张 64×64 虚拟输入、CUDA、`/tmp` 输出和 90 秒硬上限执行一次真实 DepGraph smoke。构图未在上限内完成，命令以 timeout 124 终止且无结果文件；没有读取 CarFusion、没有训练、没有宣称已生成可用剪枝模型。完整剪枝、恢复训练、精度和 latency 对比均尚未运行。
- 最终将 Vehicle Pose 核心回归与新增剪枝测试合并执行，20/20 通过（1.43 秒）；再次确认远程无剪枝/安装残留进程，`/tmp` 下不存在剪枝 checkpoint 或报告。

## 2026-07-27：保守结构恢复、替代门禁与最终回退

- `e=0.4375 + Head96` 在 RTX 3090 Ti、640、batch=1、one-to-one-only、
  fuse、FP16、3×1000 次配对门禁中从 6.0370 ms 降至 5.3843 ms，
  加速 10.81%，参数/GFLOPs 分别下降 12.65%/13.92%，允许进入 10 epoch 恢复。
- 10 epoch 最终 Box/Pose mAP50-95 为 0.47036/0.25717，远低于
  0.64761/0.49455 准入线。追加段启动时曾阻塞在 WSL `p9_client_rpc`，
  未报 CUDA/OOM；约 3 分钟后自行恢复并正常扫描数据缓存。
- 独立追加段前两轮为 Box 0.47616→0.47320、Pose 0.25831→0.26233，
  PCK@0.05 0.45983→0.45893，NME 0.09306→0.09349。联合指标进入平台且
  剩余预算不可能接近准入线，因此发送 TERM 停止；best/last、CSV 和日志保留。
- 聚焦架构/迁移/门禁测试 27/27 通过（2.72 秒）。
- 最终执行 `scripts/validate_project.sh`：核心回归 29/29 通过（1.00 秒），
  checkpoint 迁移输出 shape 相同、max/mean absolute error 均为 0；
  随后的真实 CarFusion 树检查持续 6 分钟阻塞在 WSL `p9_client_rpc`，
  无测试断言、CUDA、OOM 或数据质量错误。为避免无期限占用，精确终止
  validation 父进程及 `check_carfusion.py` 子进程，并保留为受环境 I/O 阻塞的
  部分验证，不将 `validate_project.sh` 记作完整 PASS。
- 替代门禁第一次运行因缺少 `PYTHONPATH` 失败并保留错误；设置
  `PYTHONPATH=.` 后成功。w=0.875/w=0.75 仅加速 8.08%/6.82%，速度门禁失败；
  e=0.375 Head96/Head88 加速 10.33%/10.55%，但结构比已无法恢复的 e=0.4375
  更激进，按证据停止、不启动训练。
- 最终回退到既有完整精度门禁通过的 baseline one-to-one-only + fuse + FP16
  部署方案，不提交或宣称结构压缩 checkpoint 可交付。

### 已执行命令（脱敏）

```bash
.venv/bin/python -m pytest -q tests/test_joint_latency_candidates.py tests/test_fasthead_transfer.py tests/test_pretrain_architecture_gate.py
.venv/bin/python scripts/train_pose_pruned_distillation.py --config configs/pose_conservative_e4375_head96_kd_adaptive_30e_r3.yaml
PYTHONPATH=. .venv/bin/python benchmarks/pretrain_architecture_gate.py --baseline ... --candidate ... --min-latency-reduction-pct 10
```

## 2026-07-27：PyTorch 结构化剪枝搜索与部署分支裁剪

- 用户明确要求继续压缩剪枝且不涉及 TensorRT；本轮没有执行或实现 TensorRT。
- 启动搜索前发现上次 TERM 主训练进程后遗留 4 个 PPID=1 的孤儿子进程，
  占用约 15.5 GB 显存。结果 CSV 仍只有 2 个完整 epoch；精确终止 4 个子进程
  和监控进程后显存回落到 999 MiB。
- 新增 paired whole-checkpoint fused-FP16 benchmark，避免把单模型
  FP32→FP16 收益误算为剪枝收益。TDD 红灯为缺少 helper 的导入失败；
  实现后剪枝与 benchmark 回归 22/22 通过。
- `round_to=32` 的三组 Backbone/Head 比例请求被量化成同一个 29.80% 参数
  缩减结构；paired latency 从 6.0396 ms 变为 6.0663 ms，慢 0.44%，不训练。
- 按模块剖析新增 `model_indices` 剪枝根白名单。TDD 先得到 2 个预期失败，
  实现后相关回归 23/23 通过。只剪索引 4/6/10/13/16/19/22、保护 Head 的
  25%/35% 候选分别减少 1.98%/3.03% 参数，但慢 0.27%/0.93%，不训练。
- 转向部署分支物理裁剪：删除 one-to-many `cv2/cv3/cv4`，保留完整
  one-to-one 分支。TDD 红灯后实现，Vehicle Pose/剪枝/benchmark 回归
  30/30 通过。
- 真实 baseline checkpoint 裁剪后参数从 4,005,986 降到 2,925,857
  （-26.96%），文件从 8,422,070 B 降到 6,182,717 B（-26.59%）；
  640 dense output 逐元素完全相同，max/mean error 均为 0。
- paired fused-FP16 latency 为 5.8359→5.8247 ms（快 0.19%）。
  因此只把它认定为模型体积/内存压缩，不宣称显著 latency 优化。
- 用两个独立进程测量 fused-FP16 CUDA 内存：模型加载 allocated
  57,721,856→51,245,056 B（-11.22%），含输入 60,179,456→53,702,656 B
  （-10.76%），推理峰值 72,237,568→65,742,336 B（-8.99%）；
  参数/缓冲 tensor 存储 7,984,420→5,828,130 B（-27.01%）。
- 完整 3,606 图验证通过：Box/Pose mAP50-95 0.665/0.514，
  PCK@0.05 0.66769、PCK@0.10 0.85236、NME 0.05970，与既有 fused-FP16
  baseline 保持一致。
- checkpoint 为 deployment-only，误用于训练或 one-to-many 执行会显式报错；
  权重不入库，脚本和 JSON 证据入库。
- 最终 `scripts/validate_project.sh` 的代码/模型阶段为 31/31 通过（1.19 秒），
  checkpoint 迁移输出 shape 相同且 max/mean error 为 0。随后重复的
  CarFusion 原始树检查再次阻塞在 WSL `p9_client_rpc`，在无断言、CUDA、
  OOM 或数据质量错误的情况下精确终止；不把整条脚本标为完整 PASS。

### 已执行命令（脱敏）

```bash
.venv/bin/python scripts/prune_vehicle_pose.py --model-indices 4,6,10,13,16,19,22 ...
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_deploy_fp16.py --checkpoint BASELINE --candidate-checkpoint CANDIDATE ...
PYTHONPATH=. .venv/bin/python scripts/prune_deployment_branches.py --checkpoint BASELINE --output DEPLOYMENT_ONLY ...
```

## 2026-07-27：BN-scale 稀疏训练与 active-head 物理剪枝

- 用户批准继续稀疏训练→物理剪枝路线，并再次明确只做压缩/剪枝、
  不涉及 TensorRT；本轮没有执行或实现 TensorRT。
- 按 TDD 新增 opt-in BN-scale 稀疏 Pose 模型、训练入口、分布统计和
  Torch-Pruning `BNScaleImportance`。首次红灯为模块/模型/生命周期 helper
  不存在，实现后相关测试转绿。
- 首次真实 1 epoch 冒烟在训练前发现 trainer.model 仍是 checkpoint 字符串，
  产生 `AttributeError`；新增生命周期回归后修复。第二次正式启动在训练前被
  W&B 把本地绝对 project 路径判为非法，设置 `WANDB_MODE=disabled` 后继续，
  两次失败均未产生训练结果。
- 第一版稀疏项没有按现有 Pose loss 的 batch size 缩放，1 epoch BN mean
  仅从 1.1773639 降至 1.1773475。新增 batch=4 的红灯测试后修复，重跑相同
  权重后单轮降至 1.1763493，收缩幅度约提高 62 倍；Box/Pose 为
  0.65179/0.50477，允许进入 10 epoch。
- 10 epoch 完整训练结束。BN mean 为 1.1667584（-0.90%），零通道比例仍为
  0；best 完整验证 Box/Pose 约 0.661/0.512，PCK@0.05 0.67301、
  PCK@0.10 0.85768、NME 0.05914，稀疏训练本身保持精度。
- coverage/pytest-cov 两次尝试均被远端环境阻断：首次触发 NumPy 重载及用户
  OpenCV 包的 `cv2.dnn.DictValue` 错误；`PYTHONNOUSERSITE=1` 后又缺少
  `pygments`。不伪造覆盖率数字；最终真实相关回归为 37/37。
- 全头 BN-scale r10/r15、round16 实际参数下降 15.85%/17.03%，paired
  fused-FP16 分别慢 0.81%/0.92%，未启动 KD。
- 初版 deployment-only 在物理删除 one-to-many 后再构建 DepGraph，
  Torch-Pruning 因输出中含 `None` 失败；没有生成 checkpoint。改为用完整图
  追踪依赖、只允许 active one-to-one head 成为剪枝根，结构剪完再删除
  one-to-many，相关测试转绿。
- active-head r10/r15 round16、r25/r35 round32 最终部署参数分别下降
  36.27%/37.45%/44.36%/52.12%，但 paired 延迟依次为慢 0.69%、
  快 0.04%、快 0.08%、慢 1.23%，全部未过 10% 门槛。
- 对 r10/r25 执行完整 3,606 图 FP16 验证：Box mAP50-95 仅
  0.000379/0.000244，Pose mAP50-95 约 2.67e-7/2.32e-7，
  PCK@0.05 0.01611/0.01466，精度门禁明确失败。
- 显存测量先发现稀疏训练的普通 Python BN 引用仍保留已删除 tower；
  TDD 清理后又发现 deployment-only 参数仍可求梯度，fuse 会保留 autograd
  图。新增冻结测试并修复后，r10 的 fused 参数/缓冲存储下降 36.34%，
  CUDA 模型加载下降 15.17%，峰值下降 13.74%。该候选因精度失败不交付。
- 一次 scp 把 `scripts/prune_vehicle_pose.py` 误放到远端
  `ultralytics/models/yolov10/`；发现后只删除这个刚创建的精确副本。
- 所有 BN-scale 候选均未通过延迟门槛，KD 不能改善延迟，因此按门禁不启动
  恢复训练。最终继续采用既有 branch-only checkpoint 作为精度不变的
  模型体积/内存压缩方案。
- 最终相关 Vehicle Pose/稀疏/剪枝回归 38/38 通过（2.13 秒）。
  `scripts/validate_project.sh` 的代码/模型阶段 31/31 通过（1.09 秒），
  checkpoint 迁移 shape 相同、max/mean absolute error 均为 0；
  随后 CarFusion 原始树检查连续 180 秒无输出并由 timeout 以 124 结束，
  与本日已记录的 WSL p9 I/O 阻塞一致。整条脚本不记作完整 PASS。

### 已执行命令（脱敏）

```bash
.venv/bin/python scripts/train_pose_sparsity.py --config configs/pose_sparsity_gate10e.yaml
.venv/bin/python scripts/prune_vehicle_pose.py --importance bn_scale --global-pruning --deployment-only ...
.venv/bin/python benchmarks/benchmark_deploy_fp16.py --checkpoint BASELINE --candidate-checkpoint CANDIDATE ...
.venv/bin/yolo pose val model=CANDIDATE data=CarFusion-pose.yaml imgsz=640 half=True
.venv/bin/python benchmarks/measure_deploy_memory.py --checkpoint CANDIDATE ...
```

## 2026-07-27：串行深度压缩（进行中）

- 继续遵守“只做压缩/剪枝、不涉及 TensorRT”。本轮目标为减少真正影响
  GPU latency 的串行卷积层，而不是继续缩窄通道。
- 审计既有 e=0.4375+Head96 权重迁移：原方案有 123 个缺失 tensor。
  新增多维 prefix slicing 后，直接/切片/重映射 tensor 为 495/106/24，
  缺失降至 17；但完整验证 Box/Pose mAP50-95 仅
  0.000519/9.72e-8。结论：切片破坏通道语义，不能作为恢复初始化。
- 改用已完成 60 epoch 恢复的 LitePSA+C2f-P4D1+Head96 模型，只把检测塔
  深度从 2 降到 1，保持骨干、颈部和姿态塔结构不变。候选参数
  3,620,962，较源模型 3,842,914 下降 5.78%。
- paired fused-FP16（独立交替采样）基线/候选为
  6.20268/5.46363 ms，候选加速 11.915%，通过 10% 延迟门禁。
- 未训练初始化的完整 3,606 图验证：Pose mAP50-95 为 0.484，和源模型
  一致；Box mAP50-95 为 0.00916，证明姿态分支权重保持成功，但缩短后的
  检测塔不能靠简单搬运输出层恢复。
- 已按 TDD 增加“只解冻匹配参数名前缀”的恢复训练能力。下一阶段只训练
  cv2/cv3 与 one2one_cv2/one2one_cv3 检测分支，冻结骨干、颈部和 cv4
  姿态分支，再根据门禁决定是否进入全模型低学习率 KD。
- 第一次定向恢复在第 1 epoch 将 Box 从 0.00916 恢复至 0.544，但 Pose
  从 0.484 降至 0.418。定位为冻结参数的 BatchNorm 仍在训练模式更新
  running mean/variance；第 2 epoch 45% 时主动终止，不把该 checkpoint
  作为候选。新增冻结 BN 必须使用保存统计的回归测试后，从原始干净 init
  重新训练。
- 一次批量 scp 把本轮 7 个文件平铺到远端仓库根目录；逐一 `ls` 核对后只
  删除这 7 个刚创建的精确副本，并分别同步到正确目录。未删除或覆盖其他
  根目录文件。
- 修复冻结 BN 后，检测塔 depth=1 定向恢复 10 epoch 的最好结果约为
  Box/Pose 0.633/0.474；联合 KD、检测续训和加强 one-to-one KD 均未越过
  精度门槛，因此拒绝该结构。
- 姿态塔单独 depth=1 的配对 fused-FP16 加速仅 3.48%，未过 10% 延迟
  门槛，按预设 gate 未浪费恢复训练。
- 额外删除 P3 C2f bottleneck 的候选相对原始基线加速 10.56%。简单切片
  初始化只有 Box/Pose 0.131/0.053；新增残差投影折叠后提升至
  0.316/0.181。完整 KD 在第 3 epoch 达到最好约 0.634/0.460，之后至
  第 7 epoch 停滞或回退，主动终止并拒绝。
- 同时压缩姿态塔和分类塔为 depth=1 可加速 15.47%，但分支恢复从第 1 至
  第 3 epoch 仅由 0.544/0.227 提升至 0.580/0.366，增幅快速放缓，主动
  终止。
- 只压缩分类塔 depth=1 可加速 12.26%。64 通道版本训练至第 7 epoch
  最高约 0.610/0.452；加宽到 96 通道仍有 12.01% 加速，但最好只有
  0.61831/0.45890，均因平台期终止。
- 新增 ClassLite 结构：保留分类塔第一组 depthwise+pointwise，只把第二组
  替换为 pointwise。64→64 版本加速 9.84%，未过门槛；进一步把第二层输出
  调为 48 通道后，paired fused-FP16 原始/候选为
  7.2552/6.3155 ms，加速 12.952%，通过延迟门槛。fused 参数从
  3,992,210 降至 3,819,602，计算量从 13.6 降至 13.2 GFLOPs。
- 为结构迁移新增兼容 tensor padding、C2f 输出投影折叠和 ClassLite
  pointwise 定向搬运；未恢复初始化完整验证约为 0.0368/0.012。
  使用原始完整模型作为教师、只训练分类分支时，第 1 至第 3 epoch 为
  0.59436/0.44496、0.60863/0.45501、0.60417/0.45246，继续至第 4 epoch
  仍无上升后终止。
- 随后改用具有相同 LitePSA+C2f-P4D1 上游特征的已恢复模型作为教师，
  避免不同特征分布干扰分类蒸馏。该试验正在执行；第 1 epoch 为
  Box/Pose 0.60922/0.45343，尚未超过上一试验。
- 本轮曾误把远端 `ultralytics/models/yolov10/head.py` 作为同步目标创建；
  用 `ls` 精确核对后只移除这个刚创建的错误文件，正确文件
  `ultralytics/nn/modules/head.py` 已同步。
- 新增/更新的结构迁移与轻量模块回归最近一次为 21/21 通过（1.76 秒）；
  此前选择性训练和冻结 BN 相关集合为 24/24，通过的更大相关集合为
  36/36。一次命令误引用不存在的 `tests/test_vehicle_pose_head.py` 导致
  pytest collection 报 file not found，随后改为实际测试文件并通过；该
  失败不隐藏。
- 同源教师 ClassLite48 精修第 1–5 epoch 的 Box/Pose 分别为
  0.60922/0.45343、0.61830/0.46186、0.60278/0.45134、
  0.61932/0.46163、0.61400/0.45884；第 6 epoch 早期因平台期终止，
  best 保留第 4 epoch。
- 为避免把 9.84% 的 64 通道候选算作通过，新增可独立配置
  `class_lite_channels`。先写 56 通道红灯测试，因配置不存在按预期失败；
  实现后 ClassLite 相关 2/2 回归通过。ClassLite56 paired fused-FP16
  基线/候选为 5.93824/5.35292 ms，只快 9.85677%，距门槛 0.14323 个
  百分点，严格拒绝。
- ClassLite52 paired fused-FP16 基线/候选为 6.02302/5.38792 ms，
  加速 10.54459%，通过延迟门槛；fused 参数为 3,821,186。将
  ClassLite48 best 通过零填充扩宽到 52 通道，36 个 tensor 完整 padding，
  无 missing/unexpected key；完整 3,606 图热启动验证约为
  Box/Pose 0.619/0.461。
- ClassLite52 仅标注监督、无 KD 的恢复首次启动连续约 2 分钟卡在 WSL
  `p9_client_rpc`；精确终止后重启，第二次也短暂进入相同内核等待但随后
  自行恢复。该基础设施等待不是 CUDA、OOM 或模型异常。
- 无 KD 恢复第 1–4 epoch 为 0.61734/0.46079、0.62507/0.46795、
  0.61133/0.45626、0.61709/0.45866；第 5 epoch 早期终止并保留第 2
  epoch best。
- 从该 best 做全模型低学习率、输出+特征 KD 联合精修，第 1–3 epoch 为
  0.61535/0.45831、0.61750/0.45997、0.61406/0.45722，且训练显存约
  14.5–14.7 GB；确认骨干/颈部漂移后于第 4 epoch 早期终止。
- 当前改为只训练分类塔与姿态塔，冻结骨干、颈部及 box tower，并移除
  feature KD；结构与 10.54459% 延迟结果保持不变。
- 分类+姿态头、低权重输出 KD 的第 1–2 epoch 为
  0.61707/0.45844、0.61834/0.46138，仍低于无 KD 分类恢复 best，已在
  第 3 epoch 早期终止。随后保持相同冻结范围并完全关闭 KD；当前前两轮为
  训练最终完成 12 轮，最后一轮为 0.62505/0.46712，训练显存约
  5.8 GB；仍略低于总最佳 0.62507/0.46795，确认该结构在当前监督方式下
  进入平台。
- 新增 `reports/kernel_depth/experiment_review_20260727.md`，统一汇总本轮
  所有主要压缩/剪枝路线、延迟、显存、精度、停止原因与复盘结论。

### 串行深度压缩已执行命令（脱敏）

```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_deploy_fp16.py --checkpoint BASELINE --candidate-checkpoint CANDIDATE ...
.venv/bin/python scripts/train_pose_pruned_distillation.py --config configs/POSE_COMPRESSION_EXPERIMENT.yaml
.venv/bin/yolo pose val model=CANDIDATE data=CarFusion-pose.yaml imgsz=640 half=True
PYTHONPATH=. .venv/bin/pytest -q tests/test_fasthead_transfer.py tests/test_lite_backbone_blocks.py
```

## 2026-07-27：三页算法工程复盘 PPT

- 将全部压缩/剪枝路线压缩为 3 页中文技术评审材料，统一采用 blueprint
  工程网格风格，面向算法工程师。
- 第 1 页覆盖实验地图与双门禁；第 2 页突出 deployment-only 分支裁剪和
  ClassLite52 两组最佳方案；第 3 页给出失败机制、可靠性边界与后续决策。
- PPT 明确区分“精度等价的体积/显存压缩”和“速度过线但精度未过线的研究
  候选”，没有把两者合并宣称为双门槛成功。
- 三张 1600×900 幻灯片均完成视觉检查；修复第 2 页底部文字贴边后重新渲染。
  最终 PPTX 压缩包完整性检查无错误，PDF 为 3 页、页面尺寸 1600×900。
- 输出目录：
  `slide-deck/vehicle-pose-compression-review/`，同时保留 source、analysis、
  outline、逐页 prompts、PNG、PPTX 和 PDF。

### PPT 工程化整理与复用

- 将指标、门禁、候选方案和复盘结论集中到
  `slide-deck/vehicle-pose-compression-review/src/data.mjs`，后续更新实验
  数字时无需在三页绘图代码中重复查找和替换。
- 将渲染与交付物组装拆分为 `src/render.mjs` 和 `src/build.mjs`；顶层
  `render-slides.mjs`、`merge-output.mjs` 保留为兼容入口，避免既有命令
  失效。关键约束处已添加注释，包括 paired latency、门禁一致性、连续页码
  检查和依赖延迟加载。
- 新增 `README.md`、`docs/metrics-and-gates.md` 和 `package.json`，记录
  快速重建命令、目录职责、指标口径、复用步骤和更新检查表。
- 按 TDD 先建立失败测试，再完成模块化实现。`tests/deck.test.mjs` 覆盖：
  已发布门禁/最佳两组不漂移、拒绝与指标矛盾的 PASS 标签、固定三页
  1600×900 输出，以及 PNG 页码连续和排序。
- 单元测试最终 4/4 通过；Node 实验性覆盖率统计为：line 88.52%、
  branch 86.67%、function 92.59%。完整重新渲染及合并成功，PPTX
  `unzip -t` 无错误，PDF 保持 3 页且页面为 1600×900；三页均重新完成
  视觉回归。
