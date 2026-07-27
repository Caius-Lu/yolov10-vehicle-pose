#!/usr/bin/env node
/**
 * Backward-compatible PPTX/PDF build entry point.
 *
 * Run render-slides.mjs first when data or layout changes. Speaker notes are
 * populated from prompts/*.md to preserve the slide-generation context.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildDeckArtifacts } from "./src/build.mjs";
import { DECK_DATA } from "./src/data.mjs";

const directory = path.dirname(fileURLToPath(import.meta.url));
const result = await buildDeckArtifacts({
  directory,
  metadata: DECK_DATA.metadata,
  basePromptPath: "/Users/caius/.agents/skills/baoyu-slide-deck/references/base-prompt.md",
});
console.log(`created ${result.slides}-slide PPTX and PDF`);
