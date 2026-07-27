from pathlib import Path

import pytest
import torch

from ultralytics.models.yolov10.sparsity import (
    SparsityConfig,
    YOLOv10PoseSparsityModel,
    add_bn_scale_sparsity_gradient,
    bn_scale_sparsity_loss,
    collect_sparsity_bn_layers,
    summarize_bn_scales,
)
from scripts.train_pose_sparsity import load_experiment_config, setup_sparsity_model
from ultralytics.nn.tasks import YOLOv10PoseModel


ROOT = Path(__file__).resolve().parents[1]
POSE_CFG = ROOT / "ultralytics/cfg/models/v10/yolov10n-vehicle-pose.yaml"


def test_sparsity_config_validates_weight_and_indices():
    config = SparsityConfig(weight=1e-4, model_indices=(4, 6), include_head_hidden=False)
    assert config.weight == pytest.approx(1e-4)
    assert config.model_indices == (4, 6)
    with pytest.raises(ValueError, match="weight"):
        SparsityConfig(weight=0)
    with pytest.raises(ValueError, match="weight"):
        SparsityConfig(weight=-1e-4)
    with pytest.raises(ValueError, match="model_indices"):
        SparsityConfig(weight=1e-4, model_indices=(-1,))
    with pytest.raises(ValueError, match="model_indices"):
        SparsityConfig(weight=1e-4, model_indices=(4, 4))


def test_collect_sparsity_bn_layers_matches_profiled_pruning_roots():
    model = YOLOv10PoseModel(POSE_CFG, verbose=False)

    layers = collect_sparsity_bn_layers(
        model,
        SparsityConfig(
            weight=1e-4,
            model_indices=(4, 6, 10, 13, 16, 19, 22),
            include_head_hidden=False,
        ),
    )

    assert len(layers) == 7
    assert layers == [
        model.model[4].cv2.bn,
        model.model[6].cv2.bn,
        model.model[10].cv2.bn,
        model.model[13].cv2.bn,
        model.model[16].cv2.bn,
        model.model[19].cv2.bn,
        model.model[22].cv2.bn,
    ]


def test_bn_scale_sparsity_gradient_only_updates_selected_scales():
    selected = torch.nn.BatchNorm2d(3)
    untouched = torch.nn.BatchNorm2d(3)
    selected.weight.data.copy_(torch.tensor([-2.0, 0.0, 3.0]))
    selected.weight.grad = torch.tensor([1.0, 1.0, 1.0])
    untouched.weight.grad = torch.ones(3)

    add_bn_scale_sparsity_gradient([selected], weight=0.25)

    assert torch.equal(selected.weight.grad, torch.tensor([0.75, 1.0, 1.25]))
    assert torch.equal(untouched.weight.grad, torch.ones(3))


def test_bn_scale_sparsity_loss_matches_batch_scaled_pose_loss():
    layer = torch.nn.BatchNorm2d(3)
    layer.weight.data.copy_(torch.tensor([-2.0, 0.0, 3.0]))

    loss = bn_scale_sparsity_loss([layer], weight=0.25, batch_size=4)

    assert loss.item() == pytest.approx(5.0)


def test_bn_scale_summary_reports_quantiles_and_zero_fraction():
    layer = torch.nn.BatchNorm2d(4)
    layer.weight.data.copy_(torch.tensor([0.0, 0.01, 0.1, 1.0]))

    summary = summarize_bn_scales([layer], zero_threshold=0.02)

    assert summary["channels"] == 4
    assert summary["zero_fraction"] == pytest.approx(0.5)
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(1.0)
    assert summary["p50"] == pytest.approx(0.055)


def test_sparsity_model_adds_finite_training_loss_and_bn_gradients():
    model = YOLOv10PoseSparsityModel(POSE_CFG, verbose=False)
    model.configure_sparsity(
        SparsityConfig(weight=1e-4, model_indices=(4, 6), include_head_hidden=False)
    )
    keypoints = torch.zeros(1, 14, 3)
    keypoints[..., :2] = 0.5
    keypoints[..., 2] = 2
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "keypoints": keypoints,
    }

    loss, items = model.train().loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert items.shape == (10,)
    assert model.last_sparsity_loss.item() > 0
    assert all(layer.weight.grad is not None for layer in model.sparsity_bn_layers)


def test_sparse_experiment_config_requires_inputs_and_positive_weight(tmp_path):
    config = tmp_path / "sparse.yaml"
    config.write_text("student_weights: baseline.pt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_experiment_config(config)

    config.write_text(
        "\n".join(
            [
                "student_weights: baseline.pt",
                "data: data.yaml",
                "imgsz: 640",
                "sparsity_weight: 0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sparsity_weight"):
        load_experiment_config(config)


def test_sparse_trainer_sets_up_checkpoint_model_before_bn_summary():
    class FakeModel:
        sparsity_bn_layers = ["bn"]

    class FakeTrainer:
        model = "checkpoint.pt"
        setup_calls = 0

        def setup_model(self):
            self.setup_calls += 1
            self.model = FakeModel()

    trainer = FakeTrainer()

    layers = setup_sparsity_model(trainer)

    assert trainer.setup_calls == 1
    assert layers == ["bn"]
