使用 outline.md 的完整 STYLE_INSTRUCTIONS。

生成 16:9 中文技术评审幻灯片。标题“26+ 组实验后的硬结论：参数更少 ≠ GPU 更快”。
副标题包含“RTX 3090 Ti · 640 · batch=1 · fused FP16 · paired AB/BA · 3,606 图验证”。
采用 dashboard 布局：左侧实验地图按 512 分辨率、通道剪枝、BN-scale、串行深度、
P5 串行深度和部署分支等路线排列；每类标注代表性结果和 PASS/FAIL。右侧显示
三道硬门禁：延迟下降 ≥10%，Box/Pose ≥0.64761/0.49455，以及独立进程
CUDA 推理峰值下降。突出唯一三门通过的 P5-Zero：latency -14.83%、
peak memory -5.02%。底部标出 baseline 0.665/0.515。
所有数字必须精确，不使用照片，不添加页码、logo 或脚注。
