import path from "node:path";
import { createRequire } from "node:module";
import { DECK_DATA, validateDeckData } from "./data.mjs";

// Canvas and palette are intentionally centralized. Change these constants to
// restyle the entire deck without touching experiment data.
const W = 1600;
const H = 900;
const C = {
  bg: "#FAF8F5",
  grid: "#E5E7EB",
  text: "#334155",
  muted: "#64748B",
  blue: "#2563EB",
  navy: "#1E3A5F",
  pale: "#DBEAFE",
  green: "#16A34A",
  greenBg: "#DCFCE7",
  amber: "#F59E0B",
  amberBg: "#FEF3C7",
  red: "#DC2626",
  redBg: "#FEE2E2",
  white: "#FFFFFF",
  line: "#CBD5E1",
};

const font = "PingFang SC, Noto Sans CJK SC, Microsoft YaHei, Arial, sans-serif";
const mono = "SFMono-Regular, Menlo, Consolas, monospace";

function esc(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function text(x, y, value, size = 24, weight = 400, fill = C.text, anchor = "start", family = font) {
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${esc(value)}</text>`;
}

function lines(x, y, values, size = 22, gap = 32, weight = 400, fill = C.text) {
  return values.map((v, i) => text(x, y + i * gap, v, size, weight, fill)).join("");
}

function rect(x, y, w, h, fill = C.white, stroke = C.line, r = 12, sw = 1.5) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function chip(x, y, label, type = "blue", w = 110) {
  const palette = {
    blue: [C.pale, C.blue],
    pass: [C.greenBg, C.green],
    warn: [C.amberBg, "#B45309"],
    fail: [C.redBg, C.red],
    neutral: ["#F1F5F9", C.muted],
  }[type];
  return `${rect(x, y, w, 32, palette[0], "none", 16, 0)}${text(x + w / 2, y + 22, label, 16, 700, palette[1], "middle")}`;
}

function metric(x, y, value, label, color = C.blue, valueSize = 34) {
  return `${text(x, y, value, valueSize, 750, color, "start", mono)}${text(x, y + 28, label, 17, 500, C.muted)}`;
}

function header(title, subtitle, kicker = "MODEL COMPRESSION REVIEW") {
  return `
    ${text(80, 56, kicker, 15, 750, C.blue, "start", mono)}
    ${text(80, 108, title, 38, 750, C.navy)}
    ${text(80, 142, subtitle, 19, 450, C.muted)}
    <line x1="80" y1="164" x2="1520" y2="164" stroke="${C.line}" stroke-width="2"/>
  `;
}

function grid() {
  let out = `<rect width="${W}" height="${H}" fill="${C.bg}"/>`;
  for (let x = 0; x <= W; x += 40) out += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="${C.grid}" stroke-width="0.7" opacity="0.55"/>`;
  for (let y = 0; y <= H; y += 40) out += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="${C.grid}" stroke-width="0.7" opacity="0.55"/>`;
  return out;
}

function shell(body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    ${grid()}
    ${body}
  </svg>`;
}

function fixed(value, digits = 5) {
  return Number(value).toFixed(digits);
}

function slide1(data) {
  const rows = data.experimentRows;
  const { methodology, gates, baseline } = data;
  let rowSvg = "";
  rows.forEach((r, i) => {
    const y = 206 + i * 72;
    rowSvg += `
      ${rect(82, y, 955, 60, i % 2 ? "#FFFFFF" : "#F8FAFC", C.line, 8)}
      ${text(104, y + 25, r.category, 18, 750, C.navy)}
      ${text(104, y + 47, r.variant, 15, 500, C.muted)}
      ${text(312, y + 25, r.latency, 17, 650, C.text, "start", mono)}
      ${text(645, y + 39, r.accuracy, 17, 600, C.text)}
      ${chip(905, y + 14, r.status, r.statusType, 108)}
    `;
  });
  return shell(`
    ${header("26+ 组实验后的硬结论：参数更少 ≠ GPU 更快", `统一口径：${methodology.device} · ${methodology.imageSize} · batch=${methodology.batch} · ${methodology.precision} · ${methodology.latencyProtocol} · ${methodology.validationImages.toLocaleString()} 图完整验证`)}
    ${text(82, 198, "实验路线地图", 18, 750, C.navy)}
    ${rowSvg}
    ${rect(1070, 218, 450, 310, C.white, C.line, 12)}
    ${text(1098, 252, "三门禁", 19, 750, C.navy)}
    ${chip(1098, 275, "GATE 1 · SPEED", "blue", 170)}
    ${metric(1098, 330, `≥ ${gates.minLatencyReductionPct}%`, "paired fused-FP16 延迟下降", C.blue, 32)}
    <line x1="1098" y1="366" x2="1492" y2="366" stroke="${C.line}" stroke-width="1.5"/>
    ${chip(1098, 382, "GATE 2 · ACCURACY", "blue", 190)}
    ${text(1310, 407, `Box ≥ ${fixed(gates.minBoxMap)}`, 19, 700, C.text)}
    ${text(1310, 432, `Pose ≥ ${fixed(gates.minPoseMap)}`, 19, 700, C.text)}
    <line x1="1098" y1="451" x2="1492" y2="451" stroke="${C.line}" stroke-width="1.5"/>
    ${chip(1098, 467, "GATE 3 · MEMORY", "blue", 184)}
    ${text(1310, 492, "独立进程 CUDA 峰值下降", 18, 700, C.text)}
    ${rect(1070, 546, 450, 156, C.navy, C.navy, 12)}
    ${text(1098, 578, "三门禁同时通过", 18, 600, "#BFDBFE")}
    ${text(1098, 650, "1", 68, 800, C.white, "start", mono)}
    ${text(1160, 644, "个结构候选", 23, 600, C.white)}
    ${text(1098, 682, "P5-Zero · latency −14.83% · peak −5.02%", 17, 750, "#FDE68A")}
    ${rect(80, 742, 1440, 96, "#EFF6FF", "#93C5FD", 10)}
    ${text(104, 775, "BASELINE", 16, 800, C.blue, "start", mono)}
    ${metric(250, 786, `${baseline.boxMap.toFixed(3)} / ${baseline.poseMap.toFixed(3)}`, "Box / Pose mAP50-95", C.navy, 29)}
    ${metric(600, 786, `${baseline.fp32Ms.toFixed(3)} → ${baseline.fusedFp16Ms.toFixed(3)} ms`, "one-to-one + fuse + FP16（部署优化，不计入剪枝收益）", C.blue, 29)}
    ${text(1475, 798, `−${baseline.latencyReductionPct.toFixed(2)}%`, 34, 800, C.green, "end", mono)}
  `);
}

function cardTitle(x, y, tag, titleValue, tagType) {
  return `${chip(x, y, tag, tagType, 120)}${text(x, y + 74, titleValue, 28, 750, C.navy)}`;
}

function slide2(data) {
  const [deployment, balanced] = data.bestCandidates;
  const boxMargin = balanced.boxMap - data.gates.minBoxMap;
  const poseMargin = balanced.poseMap - data.gates.minPoseMap;
  return shell(`
    ${header("两组最佳方案：体积优先与速度—精度平衡", "两组均可交付；P5-Zero 首次同时通过精度、速度与显存门禁")}
    ${rect(80, 205, 686, 565, "#F0FDF4", "#86EFAC", 16, 2)}
    ${cardTitle(112, 232, deployment.decision, deployment.label, deployment.decisionType)}
    ${text(112, 336, deployment.description, 18, 500, C.muted)}
    ${metric(112, 410, `−${deployment.parameterReductionPct.toFixed(2)}%`, "参数量", C.green, 38)}
    ${metric(330, 410, `−${deployment.fileReductionPct.toFixed(2)}%`, "checkpoint 文件", C.green, 38)}
    ${metric(570, 410, `−${deployment.peakMemoryReductionPct.toFixed(2)}%`, "推理峰值显存", C.green, 38)}
    <line x1="112" y1="467" x2="734" y2="467" stroke="#86EFAC" stroke-width="1.5"/>
    ${metric(112, 535, `${deployment.boxMap.toFixed(3)} / ${deployment.poseMap.toFixed(3)}`, `Box / Pose · 完整 ${data.methodology.validationImages.toLocaleString()} 图`, C.navy, 34)}
    ${metric(450, 535, `−${deployment.latencyReductionPct.toFixed(2)}%`, "paired latency（基本不变）", C.muted, 34)}
    ${chip(112, 600, "BIT-EXACT", "pass", 138)}
    ${text(270, 622, "dense output max/mean error = 0", 19, 650, C.text, "start", mono)}
    ${text(112, 661, "工程定位", 17, 750, C.green)}
    ${lines(112, 693, ["✓ 立即交付：模型体积 / CUDA 内存优化", "✓ 精度与输出等价，风险边界清晰", "× 不宣称显著 latency 收益"], 20, 30, 550)}

    ${rect(834, 205, 686, 565, "#EFF6FF", "#60A5FA", 16, 2)}
    ${cardTitle(866, 232, balanced.decision, balanced.label, balanced.decisionType)}
    ${text(866, 336, balanced.description, 18, 500, C.muted)}
    ${metric(866, 410, fixed(balanced.baselineMs), "baseline ms", C.muted, 34)}
    ${text(1070, 410, "→", 34, 700, C.green)}
    ${metric(1132, 410, fixed(balanced.candidateMs), "candidate ms", C.blue, 34)}
    ${metric(1375, 410, `−${balanced.latencyReductionPct.toFixed(2)}%`, "paired latency", C.green, 34)}
    <line x1="866" y1="467" x2="1488" y2="467" stroke="#93C5FD" stroke-width="1.5"/>
    ${metric(866, 535, `${fixed(balanced.boxMap)} / ${fixed(balanced.poseMap)}`, "最佳 Box / Pose", C.navy, 32)}
    ${metric(1245, 535, `−${balanced.parameterReductionPct.toFixed(2)}%`, "参数量", C.blue, 32)}
    ${chip(866, 600, "SPEED PASS", "pass", 142)}
    ${chip(1024, 600, "ACCURACY PASS", "pass", 178)}
    ${chip(1218, 600, "MEMORY PASS", "pass", 164)}
    ${text(866, 666, "精度余量 / 显存收益", 17, 750, C.green)}
    ${text(866, 704, `Box +${fixed(boxMargin)}`, 22, 700, C.green, "start", mono)}
    ${text(1080, 704, `Pose +${fixed(poseMargin)}`, 22, 700, C.green, "start", mono)}
    ${text(1300, 704, `Peak −${balanced.peakMemoryReductionPct.toFixed(2)}%`, 22, 700, C.green, "start", mono)}
    ${text(866, 744, "完整 3,606 图验证；paired AB/BA；无 TensorRT", 19, 650, C.text)}

    ${rect(246, 804, 1108, 52, C.navy, C.navy, 10)}
    ${text(800, 837, "决策：SHIP deployment-only  ·  SHIP P5-Zero 平衡结构", 23, 750, C.white, "middle")}
  `);
}

function stage(x, titleValue, number, bullets, accent) {
  let out = `${rect(x, 236, 420, 362, C.white, C.line, 14, 1.5)}
    ${text(x + 28, 278, number, 17, 800, accent, "start", mono)}
    ${text(x + 28, 320, titleValue, 25, 750, C.navy)}
    <line x1="${x + 28}" y1="342" x2="${x + 392}" y2="342" stroke="${accent}" stroke-width="3"/>`;
  bullets.forEach((b, i) => {
    out += `<circle cx="${x + 38}" cy="${382 + i * 62}" r="5" fill="${accent}"/>`;
    out += text(x + 56, 390 + i * 62, b[0], 19, 650, C.text);
    if (b[1]) out += text(x + 56, 416 + i * 62, b[1], 16, 450, C.muted);
  });
  return out;
}

function slide3(data) {
  const { methodology, review } = data;
  return shell(`
    ${header("工程决策：P5-Zero 通过三门禁，进入可复用交付", "从失败机制到成功结构，固化可复用的模型压缩搜索与验证规则")}
    ${text(80, 208, "EVIDENCE → RULES → DELIVERY", 16, 800, C.blue, "start", mono)}
    ${stage(80, "失败机制", "01", review.mechanisms, C.red)}
    ${stage(590, "已证实的工程规则", "02", review.rules, C.blue)}
    ${stage(1100, "交付与复验", "03", review.nextSteps, C.green)}
    <path d="M500 417 H570" stroke="${C.blue}" stroke-width="3"/><path d="M560 407 L575 417 L560 427" fill="none" stroke="${C.blue}" stroke-width="3"/>
    <path d="M1010 417 H1080" stroke="${C.blue}" stroke-width="3"/><path d="M1070 407 L1085 417 L1070 427" fill="none" stroke="${C.blue}" stroke-width="3"/>
    ${rect(80, 632, 1440, 170, "#EFF6FF", "#93C5FD", 14, 2)}
    ${text(110, 672, "可靠性边界", 20, 750, C.navy)}
    ${chip(110, 692, "LATENCY", "blue", 112)}
    ${text(238, 714, `同进程 ${methodology.latencyProtocol} · ${methodology.precision} · batch=${methodology.batch} · ${methodology.imageSize}`, 18, 600, C.text)}
    ${chip(790, 692, "ACCURACY", "blue", 120)}
    ${text(926, 714, `${methodology.validationImages.toLocaleString()} 图 · ${methodology.validationInstances.toLocaleString()} 实例 · Box/Pose/PCK/NME`, 18, 600, C.text)}
    ${chip(110, 742, "MEMORY", "blue", 112)}
    ${text(238, 764, "独立进程测模型加载 / 输入后 / 推理峰值", 18, 600, C.text)}
    ${chip(790, 742, "RECORD", "blue", 120)}
    ${text(926, 764, "CSV / JSON / 日志 / 停止原因全部留存", 18, 600, C.text)}
    ${rect(80, 826, 700, 42, C.green, C.green, 8)}
    ${text(430, 854, "SHIP · deployment-only branch pruning", 19, 750, C.white, "middle", mono)}
    ${rect(800, 826, 720, 42, C.green, C.green, 8)}
    ${text(1160, 854, "SHIP · P5-Zero balanced pruning", 19, 750, C.white, "middle", mono)}
  `);
}

/**
 * Render pure SVG strings. Keeping this function side-effect free makes slide
 * content testable without Sharp, PowerPoint, PDF, or filesystem access.
 */
export function renderDeckSvgs(data = DECK_DATA) {
  validateDeckData(data);
  return [
    { filename: "01-slide-experiment-map.png", svg: slide1(data) },
    { filename: "02-slide-best-two.png", svg: slide2(data) },
    { filename: "03-slide-engineering-review.png", svg: slide3(data) },
  ];
}

/**
 * Rasterize every slide into a target directory.
 * Sharp is loaded lazily so data/geometry unit tests stay dependency-light.
 */
export async function renderDeckPngs(outputDirectory, data = DECK_DATA) {
  const require = createRequire(import.meta.url);
  const sharp = require("sharp");
  const slides = renderDeckSvgs(data);
  for (const slide of slides) {
    await sharp(Buffer.from(slide.svg)).png().toFile(path.join(outputDirectory, slide.filename));
  }
  return slides.map((slide) => path.join(outputDirectory, slide.filename));
}
