# 05 Split Report

Status: **REAL GROUP POLICY CONFIRMED / CONVERSION NOT YET RUN**

The converter derives groups from the image directory hierarchy, falling back only to a stable non-numeric filename prefix. Whole groups are deterministically shuffled with a fixed seed and assigned to train/val/test. If neither source provides a reliable group, conversion stops instead of creating a random per-image split.

The real file names expose paired cameras at five sites. Camera suffixes `1/2` are collapsed to the same leakage group:

- official train: `car_butler` 2,524 images, `car_craig` 7,148, `car_fifth` 3,606, `car_morewood` 6,909;
- official test: `car_penn` 12,761 images.

With seed 0 and fixed official test group `car_penn`, the deterministic full split is:

- train: `car_butler`, `car_craig`, `car_morewood` — 16,581 images;
- val: `car_fifth` — 3,606 images;
- test: `car_penn` — 12,761 images.

This prevents paired cameras from the same site crossing splits. Final counts and manifest hashes will be recorded after conversion.
