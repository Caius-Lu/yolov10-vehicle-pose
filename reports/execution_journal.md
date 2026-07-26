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
