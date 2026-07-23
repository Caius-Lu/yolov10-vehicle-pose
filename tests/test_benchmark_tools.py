from pathlib import Path

import pytest
import torch

from benchmarks.benchmark_deploy_fp16 import build_model
from benchmarks.benchmark_end_to_end import load_sources
from benchmarks.benchmark_torch import describe
from scripts.build_training_manifest import build_manifest


def test_load_sources_resolves_dataset_list_and_rejects_missing_files(tmp_path):
    image = tmp_path / "images" / "test" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    source_list = tmp_path / "test.txt"
    source_list.write_text("images/test/one.jpg\n", encoding="utf-8")
    assert load_sources(source_list) == [image.resolve()]
    source_list.write_text("images/test/missing.jpg\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing benchmark source"):
        load_sources(source_list)


def test_benchmark_describe_reports_tail_latency():
    summary = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["mean_ms"] == 3.0
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == 4.8


def test_deploy_benchmark_loads_whole_model_checkpoint(monkeypatch):
    class PrunedCheckpointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.fused = False

        def fuse(self):
            self.fused = True
            return self

    checkpoint_model = PrunedCheckpointModel()
    calls = []

    def fake_attempt_load_one_weight(checkpoint, device=None, fuse=False):
        calls.append((checkpoint, device, fuse))
        return checkpoint_model, {"pruning": {"parameters_after": 1}}

    monkeypatch.setattr(
        "benchmarks.benchmark_deploy_fp16.attempt_load_one_weight",
        fake_attempt_load_one_weight,
    )

    loaded = build_model("pruned.pt", torch.device("cpu"), half=True, fuse=True)

    assert loaded is checkpoint_model
    assert calls == [("pruned.pt", torch.device("cpu"), False)]
    assert loaded.fused is True
    assert loaded.weight.dtype == torch.float16


def test_training_manifest_hashes_configs_and_dataset_summaries(tmp_path):
    model = tmp_path / "model.yaml"
    data = tmp_path / "data.yaml"
    model.write_text("model: test\n", encoding="utf-8")
    data.write_text("data: test\n", encoding="utf-8")
    dataset = tmp_path / "datasets" / "CarFusion" / "yolo_pose"
    dataset.mkdir(parents=True)
    (dataset / "split_summary.json").write_text('{"seed": 0}', encoding="utf-8")
    output = tmp_path / "manifest.json"
    manifest = build_manifest(tmp_path, output, model, data)
    assert output.is_file()
    assert manifest["files"]["model_config"]["sha256"]
    assert manifest["files"]["data_config"]["sha256"]
    assert manifest["files"]["split_summary"]["sha256"]
