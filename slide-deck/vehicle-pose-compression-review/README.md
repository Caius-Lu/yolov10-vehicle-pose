# Vehicle Pose 压缩实验复盘 Deck

这是一个可重复生成的 3 页算法工程评审材料。PNG、PPTX 和 PDF 都由代码生成，
所有公开指标集中保存在一个数据文件中，便于复用、改写和审计。

## 快速开始

```bash
cd slide-deck/vehicle-pose-compression-review
npm install
npm run all
```

仅修改指标后重新生成：

```bash
npm run test
npm run render
npm run build
```

当前 Codex 工作区已内置依赖时，也可以不安装本地 `node_modules`：

```bash
export NODE_PATH=/Users/caius/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules
node render-slides.mjs
node merge-output.mjs
```

## 从哪里修改

| 需求 | 修改文件 | 说明 |
|---|---|---|
| 更新指标、门槛、实验结论 | `src/data.mjs` | 单一数据源；内置 PASS/FAIL 一致性校验 |
| 修改颜色、字体、布局 | `src/render.mjs` | 1600×900 SVG 组件和三页布局 |
| 修改 PPTX/PDF 合并方式 | `src/build.mjs` | 连续页检查、speaker notes、文档元数据 |
| 修改现场演讲稿 | `speaker-notes/*.md` | 每页一份；构建时写入 PPT 备注并汇总 |
| 修改页面叙事 | `outline.md` | 幻灯片目标、文案和视觉说明 |
| 修改生成提示 | `prompts/*.md` | 每页可追溯的图像生成上下文 |
| 修改原始长文 | `source-vehicle-pose-compression-review.md` | 完整实验摘要 |

日常更新通常只需要改 `src/data.mjs`。不要在 `src/render.mjs` 中硬编码新指标，
否则图表、门禁标签和文档容易出现不一致。

## 目录结构

```text
vehicle-pose-compression-review/
├── README.md                         # 使用与改写指南
├── package.json                      # test/render/build/all 命令
├── src/
│   ├── data.mjs                      # 指标、门槛、结论：单一数据源
│   ├── render.mjs                    # 纯 SVG 渲染 + PNG 输出
│   └── build.mjs                     # PPTX/PDF 合并与输入校验
├── tests/
│   └── deck.test.mjs                 # 数据、渲染、构建契约
├── docs/
│   └── metrics-and-gates.md          # 口径、门禁与可靠性说明
├── render-slides.mjs                 # 兼容入口：生成 PNG
├── merge-output.mjs                  # 兼容入口：生成 PPTX/PDF
├── analysis.md                       # 内容与受众分析
├── outline.md                        # 三页提纲
├── prompts/                          # 逐页图像生成提示与视觉溯源
├── speaker-notes/                    # 逐页演讲稿：PPT 备注的单一来源
├── speaker-script.md                 # 构建时自动汇总的完整演讲稿
├── 01-slide-*.png                    # 生成结果
├── vehicle-pose-compression-review.pptx
└── vehicle-pose-compression-review.pdf
```

## 复用为新项目

1. 复制整个目录并重命名。
2. 修改 `src/data.mjs` 的 `metadata`、`methodology`、`gates`、
   `experimentRows` 和 `bestCandidates`。
3. 根据新项目调整 `outline.md` 与 `prompts/`。
4. 修改 `speaker-notes/*.md`；保留每页的时长、讲述目标、逐字稿和转场。
5. 运行 `npm run test`。如果手工标注的 PASS/FAIL 与数值门槛冲突，测试会失败。
6. 运行 `npm run render && npm run build`。
7. 人工检查三张 PNG 和 PPT 演讲者备注后再分发。

## 测试策略

- 数据契约：固定准入线和两组重点候选，防止误改。
- 结论校验：`speedPassed`、`accuracyPassed` 必须与数值门槛一致。
- 渲染契约：必须生成 3 张 1600×900 幻灯片。
- 构建契约：文件必须从 `01` 连续编号，避免漏页或混入旧图。
- 端到端：重新生成 PNG、PPTX、PDF；PPTX 用 `unzip -t` 检查，PDF 用
  `pdfinfo` 验证页数和尺寸。

## 可靠性原则

- 只比较同一次 paired AB/BA 运行里的 baseline/candidate。
- FP16/fuse 收益与结构剪枝收益分开表述。
- 完整精度来自 3,606 图验证，不用训练 loss 代替 mAP。
- 未同时通过速度、精度和独立 CUDA memory 门槛的结构只能标记为研究候选。
- 权重迁移、BN 冻结、停止原因和基础设施异常必须保留记录。
