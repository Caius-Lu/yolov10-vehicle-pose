/**
 * Single source of truth for every number and label shown in the deck.
 *
 * Reuse rule:
 * - Update metrics here; do not edit SVG coordinates or text in render.mjs.
 * - Keep latency values from the same paired run. Absolute values from
 *   different GPU clock states must not be compared across rows.
 */
export const DECK_DATA = {
  metadata: {
    title: "Vehicle Pose 压缩/剪枝实验总结",
    subject: "PyTorch model compression and pruning experiment review",
    author: "Vehicle Pose Compression Review",
    language: "zh-CN",
  },
  methodology: {
    device: "RTX 3090 Ti",
    imageSize: 640,
    batch: 1,
    precision: "fused FP16",
    latencyProtocol: "paired AB/BA",
    validationImages: 3606,
    validationInstances: 11569,
  },
  gates: {
    minLatencyReductionPct: 10,
    minBoxMap: 0.64761,
    minPoseMap: 0.49455,
  },
  baseline: {
    boxMap: 0.665,
    poseMap: 0.515,
    deployedPoseMap: 0.514,
    fp32Ms: 7.147,
    fusedFp16Ms: 5.727,
    latencyReductionPct: 19.88,
  },
  experimentRows: [
    {
      category: "分辨率",
      variant: "512 KD",
      latency: "7.001 > 6.905 ms",
      accuracy: "0.633 / 0.474",
      status: "FAIL",
      statusType: "fail",
    },
    {
      category: "通道剪枝",
      variant: "参数最多 −52%",
      latency: "延迟 −1.23% ~ +0.08%",
      accuracy: "部分精度≈0",
      status: "FAIL",
      statusType: "fail",
    },
    {
      category: "BN-scale",
      variant: "稀疏精度保持",
      latency: "物理剪枝不加速",
      accuracy: "0.661 / 0.512 → 崩溃",
      status: "FAIL",
      statusType: "fail",
    },
    {
      category: "串行深度",
      variant: "Box d1 / P3 C2f",
      latency: "快 11.92% / 10.56%",
      accuracy: "0.633/0.474 · 0.634/0.460",
      status: "ACC FAIL",
      statusType: "fail",
    },
    {
      category: "ClassLite",
      variant: "48 / 52 / 56 ch",
      latency: "快 12.95% / 10.54% / 9.86%",
      accuracy: "52ch 最佳 0.625/0.468",
      status: "NEAREST",
      statusType: "warn",
    },
    {
      category: "部署分支",
      variant: "删除 one-to-many",
      latency: "参数 −26.96% · 延迟 −0.19%",
      accuracy: "0.665 / 0.514 等价",
      status: "SHIP",
      statusType: "pass",
    },
  ],
  bestCandidates: [
    {
      id: "deployment-only",
      label: "Deployment-only 分支裁剪",
      decision: "SHIP",
      decisionType: "pass",
      description: "删除 one-to-many，保留完整 one-to-one 推理图",
      parameterReductionPct: 26.96,
      fileReductionPct: 26.59,
      peakMemoryReductionPct: 8.99,
      latencyReductionPct: 0.19,
      boxMap: 0.665,
      poseMap: 0.514,
      outputEquivalent: true,
      speedPassed: false,
      accuracyPassed: true,
    },
    {
      id: "classlite52",
      label: "ClassLite52 串行分类塔",
      decision: "RESEARCH",
      decisionType: "warn",
      description: "保留第一组 DS block；第二组替换为 52ch pointwise",
      baselineMs: 6.02302,
      candidateMs: 5.38792,
      latencyReductionPct: 10.54459,
      parameterReductionPct: 4.28,
      boxMap: 0.62507,
      poseMap: 0.46795,
      speedPassed: true,
      accuracyPassed: false,
    },
  ],
  review: {
    mechanisms: [
      ["参数量不是 latency 代理", "−52% 参数仍可能不加速"],
      ["分类分数牵动双任务 mAP", "Box / Pose 同时受置信度影响"],
      ["异构 KD 可能限制恢复", "纯监督最好：0.62507 / 0.46795"],
    ],
    rules: [
      ["冻结参数 ≠ 冻结行为", "必须固定 BN running statistics"],
      ["拒绝简单 tensor slicing", "使用投影折叠 / 定向搬运 / padding"],
      ["速度先 gate，再投入训练", "paired AB/BA，避免时钟偏差"],
    ],
    nextSteps: [
      ["保留完整分类塔", "避免当前最主要精度瓶颈"],
      ["从骨干 / 颈部减少串行 kernel", "优先可近恒等初始化的小改动"],
      ["重新执行三重门禁", "延迟 + 完整精度 + CUDA 内存"],
    ],
  },
};

function requireFinite(value, label) {
  if (!Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
}

/**
 * Fail fast when a manually edited label contradicts the numeric gates.
 * This prevents a polished slide from silently overstating an experiment.
 */
export function validateDeckData(data) {
  requireFinite(data?.gates?.minLatencyReductionPct, "minLatencyReductionPct");
  requireFinite(data?.gates?.minBoxMap, "minBoxMap");
  requireFinite(data?.gates?.minPoseMap, "minPoseMap");
  if (!Array.isArray(data.experimentRows) || data.experimentRows.length === 0) {
    throw new TypeError("experimentRows must not be empty");
  }
  if (!Array.isArray(data.bestCandidates) || data.bestCandidates.length !== 2) {
    throw new TypeError("bestCandidates must contain exactly two entries");
  }

  for (const candidate of data.bestCandidates) {
    const expectedSpeed = candidate.latencyReductionPct >= data.gates.minLatencyReductionPct;
    const expectedAccuracy =
      candidate.boxMap >= data.gates.minBoxMap && candidate.poseMap >= data.gates.minPoseMap;
    if (candidate.speedPassed !== expectedSpeed) {
      throw new Error(`${candidate.id}.speedPassed contradicts numeric gates`);
    }
    if (candidate.accuracyPassed !== expectedAccuracy) {
      throw new Error(`${candidate.id}.accuracyPassed contradicts numeric gates`);
    }
  }
  return true;
}
