# Slide Deck Outline

**Topic**: Vehicle Pose 压缩/剪枝实验总结与工程决策  
**Style**: blueprint  
**Dimensions**: grid + cool + technical + dense  
**Audience**: 算法工程师  
**Language**: zh-CN  
**Slide Count**: 3 slides  
**Generated**: 2026-07-27

---

<STYLE_INSTRUCTIONS>
Design Aesthetic: 精确的工程蓝图风格，以网格、门禁、对比矩阵和数据仪表盘组织信息。
高密度但严格分区，所有数字使用统一口径，结论与证据紧邻。

Background:
  Texture: 极浅灰工程网格
  Base Color: Blueprint Paper #FAF8F5

Typography:
  Headlines: 粗体几何无衬线，紧凑、权威、清晰
  Body: 现代中文无衬线，数字使用等宽风格

Color Palette:
  Primary Text: Deep Slate #334155
  Background: Blueprint Paper #FAF8F5
  Accent 1: Engineering Blue #2563EB
  Accent 2: Navy Blue #1E3A5F
  Pass: Green #16A34A
  Warning: Amber #F59E0B
  Fail: Red #DC2626

Visual Elements:
  - 精确直线、90 度连接、门禁标记、技术标签
  - 小型条形图、散点矩阵、指标卡、结构示意
  - 不使用照片、渐变、装饰性插画

Density Guidelines:
  - 每页一个主结论，允许多个工程数据点
  - 信息按 12 列网格排列，留出明确区块间距

Style Rules:
  Do: 强调数据口径、PASS/FAIL、可交付边界；使用对齐和一致线宽
  Don't: 不夸大未达标结果；不把 FP16 收益算作剪枝收益；不使用模糊形容词
</STYLE_INSTRUCTIONS>

---

## Slide 1 of 3

**Type**: Content
**Filename**: 01-slide-experiment-map.png

// NARRATIVE GOAL
用一页展示所有主要路线和统一门禁，建立“参数下降不等于延迟下降”的核心认识。

// KEY CONTENT
Headline: 26+ 组实验后的硬结论：参数更少 ≠ GPU 更快
Sub-headline: RTX 3090 Ti · 640 · batch=1 · fused FP16 · paired AB/BA · 3,606 图验证
Body:
- 结构化通道剪枝：参数最多 -52%，延迟基本无收益，部分精度崩溃
- 串行深度压缩：速度可过线，但精度恢复成为主瓶颈
- 唯一可靠的完整精度压缩：删除 one-to-many 部署分支
- 当前最接近双门槛的延迟结构：ClassLite52

// VISUAL
左侧实验路线矩阵，按分辨率、通道剪枝、稀疏、串行深度、部署分支五类分组；
右侧两道门禁和通过数量；底部标出 baseline 与准入线。

// LAYOUT
Layout: dashboard
顶部标题，中部 7:5 分栏，底部横向基准条。

---

## Slide 2 of 3

**Type**: Content
**Filename**: 02-slide-best-two.png

// NARRATIVE GOAL
突出两组最佳成果，并明确它们解决的是不同工程目标。

// KEY CONTENT
Headline: 两组最佳方案：一组可交付，一组仍是研究候选
Sub-headline: 不把体积/显存收益与延迟收益混为同一个成功
Body:
- Deployment-only 分支裁剪：参数 -26.96%，文件 -26.59%，峰值显存 -8.99%，
  Box/Pose 0.665/0.514，输出逐元素一致；延迟仅快 0.19%
- ClassLite52：6.02302→5.38792 ms，快 10.54459%，参数约 -4.28%；
  最佳 0.62507/0.46795，距离精度线仍差 0.02254/0.02660

// VISUAL
左右二元对比。左侧绿色“SHIP”证据链，右侧琥珀色“RESEARCH”门禁图；
中央小型二维坐标展示二者在精度保持与延迟收益上的位置。

// LAYOUT
Layout: binary-comparison
两张大卡片等宽，中央以工程决策轴连接。

---

## Slide 3 of 3

**Type**: Back Cover
**Filename**: 03-slide-engineering-review.png

// NARRATIVE GOAL
给出面向算法工程师的机制复盘、可靠性边界和下一步决策。

// KEY CONTENT
Headline: 工程决策：先交付内存压缩，停止重复微调 ClassLite52
Body:
- 机制：GPU 延迟由串行 kernel 主导；分类置信度同时牵动 Box/Pose mAP
- 训练：冻结参数必须冻结 BN 统计；纯监督恢复优于当前异构教师 KD
- 迁移：拒绝简单 slicing，采用投影折叠、定向搬运和 padding 热启动
- 下一步：保留完整分类塔，从骨干/颈部寻找可近恒等初始化、可减少串行 kernel
  的小改动；任何候选必须重新通过 paired latency + 完整精度 + 显存三门禁

// VISUAL
三段证据链：失败机制 → 已证实规则 → 下一步门禁；右下角给出 SHIP/HOLD 决策框。

// LAYOUT
Layout: linear-progression
横向三阶段流程，下方决策结论条。
