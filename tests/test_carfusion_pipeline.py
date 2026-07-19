"""Tests for deterministic, leakage-safe CarFusion conversion."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.convert_carfusion import convert_coco_dataset, derive_group, normalize_annotation, split_groups
from scripts.prepare_carfusion_coco import combine_official_splits
from scripts.visualize_carfusion import render_previews
from scripts.check_carfusion import check_dataset
from scripts.create_carfusion_lists import create_lists
from scripts.materialize_dataset_config import materialize_dataset_config


def _keypoints(visibility=2):
    values = []
    for index in range(14):
        values.extend((10 + index, 20 + index, visibility))
    return values


def test_normalize_annotation_preserves_visibility_and_invalidates_out_of_bounds_points():
    annotation = {"id": 7, "image_id": 3, "bbox": [10, 20, 40, 20], "keypoints": _keypoints()}
    annotation["keypoints"][0:3] = [-1, 10, 2]
    label, issues = normalize_annotation(annotation, width=100, height=80)
    fields = label.split()
    assert len(fields) == 5 + 14 * 3
    assert fields[:5] == ["0", "0.300000", "0.375000", "0.400000", "0.250000"]
    assert fields[5:8] == ["0.000000", "0.000000", "0"]
    assert issues == ["annotation 7: keypoint 0 outside image; marked invalid"]


@pytest.mark.parametrize(
    "annotation,error",
    [
        ({"id": 1, "bbox": [0, 0, 0, 10], "keypoints": _keypoints()}, "invalid bbox"),
        ({"id": 2, "bbox": [0, 0, 10, 10], "keypoints": _keypoints()[:-3]}, "expected 14 keypoints"),
        ({"id": 3, "bbox": [0, 0, 10, 10], "keypoints": _keypoints(9)}, "invalid visibility"),
    ],
)
def test_invalid_annotations_are_rejected(annotation, error):
    with pytest.raises(ValueError, match=error):
        normalize_annotation(annotation, width=100, height=80)


def test_group_derivation_and_split_are_deterministic_without_leakage():
    assert derive_group("camera_01/sequence_02/frame_0001.jpg") == "camera_01/sequence_02"
    assert derive_group("cam01_000123.jpg") == "cam01"
    assert derive_group("000123.jpg") is None
    assert derive_group("train/car_fifth1/images_jpg/6_01361.jpg") == "car_fifth"
    assert derive_group("train/car_fifth2/images_jpg/9_03761.jpg") == "car_fifth"
    groups = {f"camera_{index:02d}/sequence_00": [index] for index in range(20)}
    first = split_groups(groups, seed=17)
    second = split_groups(groups, seed=17)
    assert first == second
    memberships = [set(first[name]) for name in ("train", "val", "test")]
    assert not (memberships[0] & memberships[1] or memberships[0] & memberships[2] or memberships[1] & memberships[2])


def test_five_group_split_keeps_every_split_nonempty_and_can_reserve_official_test():
    groups = {name: list(range(size)) for name, size in {
        "car_butler": 2524,
        "car_craig": 7148,
        "car_fifth": 3606,
        "car_morewood": 6909,
        "car_penn": 12761,
    }.items()}
    split = split_groups(groups, seed=0, fixed_test_groups={"car_penn"})
    assert split == {
        "train": ["car_butler", "car_craig", "car_morewood"],
        "val": ["car_fifth"],
        "test": ["car_penn"],
    }


def test_fixture_conversion_writes_yolo_labels_and_reproducible_summaries(tmp_path):
    images_root = tmp_path / "images"
    coco_images = []
    annotations = []
    for index in range(6):
        relative = Path(f"camera_{index:02d}") / "sequence_00" / f"frame_{index:04d}.jpg"
        path = images_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), color=(index, 0, 0)).save(path)
        coco_images.append({"id": index, "file_name": relative.as_posix(), "width": 100, "height": 80})
        annotations.append({"id": index, "image_id": index, "bbox": [10, 20, 40, 20], "keypoints": _keypoints()})
    coco = {"images": coco_images, "annotations": annotations, "categories": [{"id": 1, "name": "car"}]}
    json_path = tmp_path / "carfusion.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    output = tmp_path / "converted"
    summary = convert_coco_dataset(json_path, images_root, output, seed=4)
    assert summary["status"] == "READY"
    assert summary["images"] == 6
    assert summary["annotations"] == 6
    assert sum(summary["split_images"].values()) == 6
    assert (output / "conversion_summary.json").is_file()
    assert (output / "split_summary.json").is_file()
    assert len(list(output.glob("labels/*/*.txt"))) == 6
    assert len(list(output.glob("images/*/*.jpg"))) == 6
    for split in ("train", "val", "test"):
        assert all(line.startswith(f"./images/{split}/") for line in (output / f"{split}.txt").read_text().splitlines())
    quality = check_dataset(output)
    assert quality["status"] == "PASS"
    assert quality["instances"] == 6
    assert quality["group_leakage"] is False
    assert quality["cars_per_image"]["mean"] == 1.0
    assert quality["bbox"]["area_normalized"]["count"] == 6
    assert quality["keypoints"]["0"]["visibility"] == {"0": 0, "1": 0, "2": 6}
    preview = tmp_path / "preview"
    manifest = render_previews(output, preview, limit=3, seed=0)
    assert len(manifest) == 3
    assert len(list(preview.glob("*.jpg"))) == 3
    assert (preview / "manifest.json").is_file()
    fixed_eval = tmp_path / "fixed_eval_images.txt"
    subsets = create_lists(output, fixed_eval, smoke_train=2, smoke_val=1, overfit=2, fixed_eval_count=1, seed=7)
    assert subsets["counts"] == {"smoke_train": 2, "smoke_val": 1, "overfit": 2, "fixed_eval": 1}
    assert (output / "smoke_train.txt").is_file()
    assert (output / "smoke_val.txt").is_file()
    assert (output / "overfit.txt").is_file()
    assert fixed_eval.is_file()


def test_explicit_drop_policy_records_zero_bbox_annotation(tmp_path):
    images_root = tmp_path / "images"
    images = []
    annotations = []
    for image_id, site in enumerate(("fifth", "butler", "craig"), start=1):
        relative = Path(f"car_{site}1") / "images_jpg" / f"{image_id}.jpg"
        image_path = images_root / relative
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 80)).save(image_path)
        images.append({"id": image_id, "file_name": relative.as_posix(), "width": 100, "height": 80})
        annotations.append(
            {"id": image_id, "image_id": image_id, "bbox": [10, 20, 40, 20], "keypoints": _keypoints()}
        )
    annotations.append({"id": 4, "image_id": 1, "bbox": [0, 0, 0, 0], "keypoints": _keypoints()})
    coco = {
        "images": images,
        "annotations": annotations,
    }
    json_path = tmp_path / "carfusion.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")
    summary = convert_coco_dataset(
        json_path,
        images_root,
        tmp_path / "converted",
        invalid_bbox_policy="drop",
        fixed_test_groups=set(),
    )
    assert summary["annotations"] == 4
    assert summary["converted_annotations"] == 3
    assert summary["dropped_invalid_bboxes"] == 1
    quality = check_dataset(tmp_path / "converted")
    assert quality["status"] == "PASS"
    assert quality["instances"] == 3


def test_combine_official_splits_prefixes_paths_and_remaps_annotation_ids(tmp_path):
    train = {
        "images": [{"id": 2001, "file_name": "car_fifth1/images_jpg/1.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 0, "image_id": 2001, "bbox": [1, 1, 2, 2], "keypoints": _keypoints()}],
        "categories": [{"id": 1, "name": "car", "keypoints": [str(index) for index in range(14)]}],
    }
    test = {
        "images": [{"id": 1001, "file_name": "car_penn1/images_jpg/2.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 0, "image_id": 1001, "bbox": [1, 1, 2, 2], "keypoints": _keypoints()}],
        "categories": train["categories"],
    }
    train_path, test_path = tmp_path / "train.json", tmp_path / "test.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    test_path.write_text(json.dumps(test), encoding="utf-8")
    output = tmp_path / "combined.json"
    summary = combine_official_splits(train_path, test_path, output)
    combined = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {"images": 2, "annotations": 2, "train_images": 1, "test_images": 1}
    assert [image["file_name"] for image in combined["images"]] == [
        "train/car_fifth1/images_jpg/1.jpg",
        "test/car_penn1/images_jpg/2.jpg",
    ]
    assert len({annotation["id"] for annotation in combined["annotations"]}) == 2
    assert combined["annotations"][1]["source_annotation_id"] == 0


def test_quality_checker_reports_missing_conversion_metadata(tmp_path):
    result = check_dataset(tmp_path)
    assert result["status"] == "ERROR"
    assert result["errors"] == ["conversion or split summary is missing"]


def test_quality_checker_allows_six_decimal_bbox_rounding_at_image_boundary(tmp_path):
    images_root = tmp_path / "images"
    images = []
    annotations = []
    for image_id, site in enumerate(("butler", "fifth", "penn"), start=1):
        relative = Path(f"car_{site}1") / "images_jpg" / f"{image_id}.jpg"
        image_path = images_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080)).save(image_path)
        images.append({"id": image_id, "file_name": relative.as_posix(), "width": 1920, "height": 1080})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "bbox": [1643, 222, 277, 126],
                "keypoints": _keypoints(),
            }
        )
    json_path = tmp_path / "carfusion.json"
    json_path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
    output = tmp_path / "converted"
    convert_coco_dataset(json_path, images_root, output, fixed_test_groups={"car_penn"})

    quality = check_dataset(output)

    assert quality["status"] == "PASS"
    assert quality["errors"] == []


def test_runtime_dataset_config_uses_absolute_project_data_root(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text("path: datasets/CarFusion/yolo_pose\ntrain: smoke_train.txt\n", encoding="utf-8")
    output = tmp_path / "runtime" / "smoke.yaml"
    dataset_root = tmp_path / "project" / "datasets" / "CarFusion" / "yolo_pose"

    materialize_dataset_config(template, output, dataset_root)

    assert output.read_text(encoding="utf-8").splitlines() == [
        f"path: {dataset_root.resolve()}",
        "train: smoke_train.txt",
    ]


def test_conversion_can_defer_image_verification_to_the_quality_gate(tmp_path):
    images_root = tmp_path / "images"
    images = []
    annotations = []
    for image_id, site in enumerate(("butler", "fifth", "penn"), start=1):
        relative = Path(f"car_{site}1") / "images_jpg" / f"{image_id}.jpg"
        image_path = images_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"not-a-real-jpeg")
        images.append({"id": image_id, "file_name": relative.as_posix(), "width": 100, "height": 80})
        annotations.append(
            {"id": image_id, "image_id": image_id, "bbox": [10, 20, 40, 20], "keypoints": _keypoints()}
        )
    json_path = tmp_path / "carfusion.json"
    json_path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")

    summary = convert_coco_dataset(
        json_path,
        images_root,
        tmp_path / "converted",
        fixed_test_groups={"car_penn"},
        verify_images=False,
    )

    assert summary["status"] == "READY"
    assert summary["image_verification"] == "deferred_to_quality_gate"
