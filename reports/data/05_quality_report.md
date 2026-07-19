# 05 Quality Report

Status: **CHECKER READY / NOT YET RUN ON EXTRACTED DATA**

The quality command checks missing/corrupt images, empty labels, field count, finite values, normalized bbox/keypoint ranges, visibility values, class IDs, image/label/converted-annotation counts, exact duplicate instances, and group leakage. It reports cars per image; normalized and pixel bbox distributions; COCO small/medium/large and tiny-vehicle counts; per-keypoint visibility, coordinate distributions, and outside-bbox ratios.

Streaming annotation audit found:

- train: 20,187 images, 40,642 annotations, 484 zero bboxes;
- test: 12,761 images, 22,531 annotations, 255 zero bboxes;
- bbox outside image: 0 in both splits;
- keypoint value length: 42 for every annotation;
- visibility values are limited to 0/1/2.

The full image integrity pass and 100 rendered previews wait for remote extraction.
