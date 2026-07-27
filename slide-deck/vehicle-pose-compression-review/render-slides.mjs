#!/usr/bin/env node
/**
 * Backward-compatible rendering entry point.
 *
 * Most edits belong in:
 * - src/data.mjs   metrics, labels, gates, conclusions
 * - src/render.mjs geometry and visual components
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DECK_DATA } from "./src/data.mjs";
import { renderDeckPngs } from "./src/render.mjs";

const directory = path.dirname(fileURLToPath(import.meta.url));
const outputs = await renderDeckPngs(directory, DECK_DATA);
for (const output of outputs) console.log(`rendered ${path.basename(output)}`);
