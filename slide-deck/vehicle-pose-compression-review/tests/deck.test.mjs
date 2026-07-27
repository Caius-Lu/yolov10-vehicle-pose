import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { DECK_DATA, validateDeckData } from "../src/data.mjs";
import { renderDeckSvgs } from "../src/render.mjs";
import { findSlideImages } from "../src/build.mjs";

test("deck data keeps the published gates and the two highlighted candidates", () => {
  assert.doesNotThrow(() => validateDeckData(DECK_DATA));
  assert.equal(DECK_DATA.gates.minLatencyReductionPct, 10);
  assert.equal(DECK_DATA.gates.minBoxMap, 0.64761);
  assert.equal(DECK_DATA.gates.minPoseMap, 0.49455);
  assert.deepEqual(
    DECK_DATA.bestCandidates.map((candidate) => candidate.id),
    ["deployment-only", "p5zero"],
  );
  assert.equal(DECK_DATA.bestCandidates[1].speedPassed, true);
  assert.equal(DECK_DATA.bestCandidates[1].accuracyPassed, true);
});

test("deck data rejects contradictory pass labels", () => {
  const invalid = structuredClone(DECK_DATA);
  invalid.bestCandidates[1].accuracyPassed = false;

  assert.throws(() => validateDeckData(invalid), /accuracyPassed/);
});

test("renderer produces exactly three 1600x900 slide SVGs", () => {
  const slides = renderDeckSvgs(DECK_DATA);

  assert.equal(slides.length, 3);
  for (const slide of slides) {
    assert.match(slide.filename, /^\d{2}-slide-.*\.png$/);
    assert.match(slide.svg, /width="1600" height="900"/);
    assert.match(slide.svg, /Vehicle Pose|P5-Zero|参数更少/);
  }
});

test("build input discovery sorts slides and rejects gaps", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "vehicle-pose-deck-"));
  writeFileSync(path.join(directory, "02-slide-b.png"), "b");
  writeFileSync(path.join(directory, "01-slide-a.png"), "a");
  writeFileSync(path.join(directory, "notes.md"), "ignored");

  assert.deepEqual(
    findSlideImages(directory).map((item) => item.filename),
    ["01-slide-a.png", "02-slide-b.png"],
  );
  assert.match(findSlideImages(directory)[0].speakerNotePath, /speaker-notes\/01-slide-a\.md$/);

  writeFileSync(path.join(directory, "04-slide-gap.png"), "gap");
  assert.throws(() => findSlideImages(directory), /continuous/);
});
