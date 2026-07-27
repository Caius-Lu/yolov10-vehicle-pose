# 指标与门禁口径

## 评估环境

- GPU：RTX 3090 Ti
- 输入：640×640，batch=1
- 路径：one-to-one-only
- 精度：Conv/BN fuse + FP16
- 延迟：同进程 AB/BA 交错的 paired benchmark
- 验证集：CarFusion 3,606 图、11,569 实例

## 准入线

| 门禁 | 条件 |
|---|---:|
| 延迟 | 相对原始完整 baseline 下降 ≥10% |
| Box | mAP50-95 ≥0.64761 |
| Pose | mAP50-95 ≥0.49455 |

只有三项同时通过才可称为结构压缩最终候选。

## 两组重点结果

### Deployment-only 分支裁剪

- 删除 one-to-many 分支，保留 one-to-one 推理图。
- 参数量下降 26.96%，checkpoint 文件下降 26.59%。
- 推理峰值 CUDA 内存下降 8.99%。
- dense output max/mean error 均为 0。
- Box/Pose 为 0.665/0.514。
- paired latency 只下降 0.19%，因此只宣称体积/显存收益。

### ClassLite52

- paired fused-FP16：6.02302→5.38792 ms，下降 10.54459%。
- 参数量下降约 4.28%。
- 最佳 Box/Pose 为 0.62507/0.46795。
- 距离精度线分别差 0.02254/0.02660。
- 速度通过、精度失败，只能标记为研究候选。

## 为什么绝对延迟会变化

不同实验中的 baseline 绝对毫秒数会受 GPU 时钟、温度和系统负载影响。因此：

1. 只使用同一次运行内的 baseline/candidate 配对结果计算比例。
2. 不跨实验直接比较候选的绝对毫秒数。
3. 最终候选必须重复 paired benchmark，不能只取一次最优结果。

## 数据更新检查表

- [ ] 指标已写入 `src/data.mjs`
- [ ] `speedPassed`/`accuracyPassed` 与数值门槛一致
- [ ] source、outline、prompts 中没有残留旧指标
- [ ] `npm run test:coverage` 通过
- [ ] PNG 完成人工视觉检查
- [ ] PPTX `unzip -t` 无错误
- [ ] PDF 为 3 页、16:9
