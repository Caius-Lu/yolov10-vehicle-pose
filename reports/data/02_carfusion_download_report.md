# 02 CarFusion Download Report

Status: **AUTHORIZED USER-SUPPLIED PACKAGE / TRANSFER IN PROGRESS**

## Official sources

- Project: <https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/>
- Access form: <https://forms.gle/FCUcbt3jD1hB6ja57>
- Converter: `https://github.com/dineshreddy91/carfusion_to_coco.git`
- Converter commit: `ac241a9caec16d45ceb8130b21a541c72cb09c33`

## Completed

- Cloned and audited the public converter into `external/carfusion_to_coco`.
- Confirmed its README requires the separately authorized `download_carfusion.py`.
- Inspected `carfusion_coco_setup.sh`; it contains fixed relative paths and invokes Python conversion for train/test. It contains no recursive deletion, but will not be run until source paths are verified.
- Created `datasets/CarFusion/raw`, `datasets/CarFusion/coco`, and report directories on the server.
- Implemented and fixture-tested the COCO-to-YOLO pose converter with strict 14-point validation, group-level deterministic splits, visibility checks, image validation, stable ID mapping, and leakage reporting.
- On 2026-07-17 the user explicitly supplied `/Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion` as the authorized dataset source.
- Safe metadata probe found 503 files and 34,451,319,446 total bytes: one 34,123,960,720-byte gzip/tar payload, 500 sample JPEGs, README and metadata.
- The archive header is valid and begins with `CarFusion/`; local SHA-256 is `34ae638404d724a50b744dc7cba359b4f38cc336d9c45e5f0cea6a996fe90eb4`.
- Remote free space was 727 GiB before transfer.
- Started resumable rsync to `/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion`.

## Current gate

Do not extract or convert until rsync completes and the remote archive SHA-256 exactly matches the local value. After extraction, inventory the actual annotation format before selecting or adapting a converter.

## Commands

```bash
git clone https://github.com/dineshreddy91/carfusion_to_coco.git external/carfusion_to_coco
git -C external/carfusion_to_coco rev-parse HEAD
.venv/bin/python -m pytest tests/test_carfusion_pipeline.py -q
python3 scripts/probe_carfusion_package.py /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion
shasum -a 256 /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion/raw/CarFusion.tar.gz.00
rsync -a --partial --progress /Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion/ ubuntu-ts:/home/caius/projects/yolopose/data/CarFusion/OpenDataLab___CarFusion/
```
