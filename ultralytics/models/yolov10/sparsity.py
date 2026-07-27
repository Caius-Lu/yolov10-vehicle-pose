"""Opt-in BN-scale sparsity helpers for latency-aware Vehicle Pose pruning."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ultralytics.models.yolov10.prune_distill import collect_prunable_root_layers
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import YOLOv10PoseModel


@dataclass(frozen=True)
class SparsityConfig:
    """Select BN scales and the L1 gradient applied during sparse training."""

    weight: float = 1e-4
    model_indices: tuple[int, ...] = ()
    include_head_hidden: bool = True

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("sparsity weight must be positive")
        if any(int(index) < 0 for index in self.model_indices) or len(set(self.model_indices)) != len(
            self.model_indices
        ):
            raise ValueError("model_indices must contain unique non-negative model indices")


def collect_sparsity_bn_layers(model, config):
    """Return BatchNorm scales paired with the configured pruning-root convolutions."""
    if not isinstance(model, YOLOv10PoseModel):
        raise TypeError("model must be a YOLOv10PoseModel")
    roots = collect_prunable_root_layers(
        model,
        include_head_hidden=config.include_head_hidden,
        model_indices=config.model_indices,
    )
    root_ids = {id(layer) for layer in roots}
    layers = [
        module.bn
        for module in model.modules()
        if isinstance(module, Conv) and id(module.conv) in root_ids and isinstance(module.bn, nn.BatchNorm2d)
    ]
    if len(layers) != len(roots):
        raise RuntimeError("every sparsity pruning root must have a BatchNorm scale")
    return layers


def add_bn_scale_sparsity_gradient(layers, weight):
    """Add an L1 subgradient to selected BN scales after AMP gradient unscaling."""
    if weight <= 0:
        raise ValueError("sparsity weight must be positive")
    for layer in layers:
        if layer.weight.grad is not None:
            layer.weight.grad.add_(layer.weight.detach().sign(), alpha=weight)


def bn_scale_sparsity_loss(layers, weight, batch_size):
    """Return an L1 BN-scale penalty on the same batch scale as the pose loss."""
    if weight <= 0:
        raise ValueError("sparsity weight must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return weight * batch_size * torch.stack([layer.weight.abs().sum() for layer in layers]).sum()


def summarize_bn_scales(layers, zero_threshold=1e-3):
    """Summarize absolute BN scales for experiment gates and manifests."""
    if zero_threshold < 0:
        raise ValueError("zero_threshold must be non-negative")
    scales = [layer.weight.detach().float().abs().cpu() for layer in layers]
    if not scales:
        raise ValueError("at least one BN layer is required")
    values = torch.cat(scales)
    return {
        "channels": int(values.numel()),
        "zero_threshold": float(zero_threshold),
        "zero_fraction": float((values <= zero_threshold).float().mean()),
        "min": float(values.min()),
        "p25": float(torch.quantile(values, 0.25)),
        "p50": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


class YOLOv10PoseSparsityModel(YOLOv10PoseModel):
    """Vehicle Pose model with an opt-in BN-scale L1 training objective."""

    def configure_sparsity(self, config):
        self.sparsity_config = config
        self.sparsity_bn_layers = collect_sparsity_bn_layers(self, config)
        self.last_sparsity_loss = torch.zeros(())
        return self

    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        config = getattr(self, "sparsity_config", None)
        if self.training and config is not None:
            sparsity_loss = bn_scale_sparsity_loss(
                self.sparsity_bn_layers,
                weight=config.weight,
                batch_size=int(batch["img"].shape[0]),
            )
            loss = loss + sparsity_loss
            self.last_sparsity_loss = sparsity_loss.detach()
        else:
            self.last_sparsity_loss = loss.detach().new_zeros(())
        return loss, items
