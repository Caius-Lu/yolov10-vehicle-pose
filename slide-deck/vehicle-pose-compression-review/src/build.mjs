import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

/**
 * Discover rendered slides using the skill-compatible NN-slide-*.png naming
 * convention. A gap is rejected because it usually means a stale or missing
 * page would be silently shipped.
 */
export function findSlideImages(directory) {
  if (!existsSync(directory)) throw new Error(`Deck directory not found: ${directory}`);
  const items = readdirSync(directory)
    .filter((filename) => /^\d{2}-slide-.*\.png$/i.test(filename))
    .map((filename) => ({
      filename,
      path: path.join(directory, filename),
      index: Number.parseInt(filename.slice(0, 2), 10),
      promptPath: path.join(directory, "prompts", filename.replace(/\.png$/i, ".md")),
      speakerNotePath: path.join(directory, "speaker-notes", filename.replace(/\.png$/i, ".md")),
    }))
    .sort((a, b) => a.index - b.index);

  if (items.length === 0) throw new Error(`No slide images found in: ${directory}`);
  for (let index = 0; index < items.length; index += 1) {
    if (items[index].index !== index + 1) {
      throw new Error("Slide image indices must be continuous and start at 01");
    }
  }
  return items;
}

/**
 * Merge PNG slides into PPTX and PDF. Dependencies are loaded lazily so unit
 * tests for discovery/validation do not require presentation libraries.
 */
export async function buildDeckArtifacts({
  directory,
  metadata,
  basePromptPath,
  basename = "vehicle-pose-compression-review",
}) {
  const require = createRequire(import.meta.url);
  const PptxGenJS = require("pptxgenjs");
  const { PDFDocument } = require("pdf-lib");
  const slides = findSlideImages(directory);
  const basePrompt = basePromptPath && existsSync(basePromptPath) ? readFileSync(basePromptPath, "utf8") : "";

  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = metadata.author;
  pptx.subject = metadata.subject;
  pptx.title = metadata.title;
  pptx.lang = metadata.language;
  const speakerSections = [];
  for (const item of slides) {
    const slide = pptx.addSlide();
    slide.background = { color: "FAF8F5" };
    slide.addImage({ path: item.path, x: 0, y: 0, w: 13.333333, h: 7.5 });

    // Speaker notes take precedence over image-generation prompts. The prompt
    // fallback keeps this builder reusable for older decks without a script.
    if (existsSync(item.speakerNotePath)) {
      const speakerNote = readFileSync(item.speakerNotePath, "utf8").trim();
      slide.addNotes(speakerNote);
      speakerSections.push(speakerNote);
    } else if (existsSync(item.promptPath)) {
      const prompt = readFileSync(item.promptPath, "utf8");
      slide.addNotes(basePrompt ? `${basePrompt}\n\n---\n\n${prompt}` : prompt);
    }
  }
  const pptxPath = path.join(directory, `${basename}.pptx`);
  await pptx.writeFile({ fileName: pptxPath });

  const pdf = await PDFDocument.create();
  pdf.setAuthor(metadata.author);
  pdf.setSubject(metadata.subject);
  pdf.setTitle(metadata.title);
  for (const item of slides) {
    const image = await pdf.embedPng(readFileSync(item.path));
    const page = pdf.addPage([1600, 900]);
    page.drawImage(image, { x: 0, y: 0, width: 1600, height: 900 });
  }
  const pdfPath = path.join(directory, `${basename}.pdf`);
  writeFileSync(pdfPath, await pdf.save());

  // Keep a printable/editable script next to the deck. Its content is generated
  // from the same per-slide notes embedded in PowerPoint to prevent drift.
  const speakerScriptPath = path.join(directory, "speaker-script.md");
  if (speakerSections.length > 0) {
    writeFileSync(
      speakerScriptPath,
      `# Vehicle Pose 压缩实验复盘：演讲稿\n\n` +
        `> 面向算法工程师；建议总时长 8–10 分钟。PPT 演讲者备注与本文同源。\n\n` +
        `${speakerSections.join("\n\n---\n\n")}\n`,
    );
  }
  return { slides: slides.length, pptxPath, pdfPath, speakerScriptPath };
}
