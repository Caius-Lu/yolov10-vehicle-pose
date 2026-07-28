from pathlib import Path

import pytest
import torch

from benchmarks.benchmark_deploy_fp16 import (
    build_model,
    summarize_paired_latency,
    validate_output_path as validate_latency_output,
)
from benchmarks.benchmark_end_to_end import load_sources
from benchmarks.benchmark_torch import describe
from benchmarks.measure_deploy_memory import (
    parameter_storage_bytes,
    validate_output_path as validate_memory_output,
)
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


def test_paired_deploy_summary_requires_matching_rounds_and_reports_reduction():
    summary = summarize_paired_latency([6.0, 6.2, 5.8], [5.4, 5.6, 5.2])

    assert summary["baseline_fused_fp16_ms"] == pytest.approx(6.0)
    assert summary["candidate_fused_fp16_ms"] == pytest.approx(5.4)
    assert summary["latency_reduction_pct"] == pytest.approx(10.0)
    assert summary["speedup"] == pytest.approx(6.0 / 5.4)

    with pytest.raises(ValueError, match="same non-zero"):
        summarize_paired_latency([6.0], [])
    with pytest.raises(ValueError, match="positive"):
        summarize_paired_latency([6.0], [0.0])


def test_deploy_memory_counts_parameter_and_buffer_storage():
    module = torch.nn.BatchNorm2d(4).half()

    expected = sum(tensor.numel() * tensor.element_size() for tensor in [*module.parameters(), *module.buffers()])

    assert parameter_storage_bytes(module) == expected


@pytest.mark.parametrize("validator", [validate_latency_output, validate_memory_output])
def test_benchmark_reports_refuse_to_overwrite_existing_evidence(tmp_path, validator):
    output = tmp_path / "result.json"
    validator(output)
    output.write_text("existing evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        validator(output)


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
