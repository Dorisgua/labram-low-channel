"""Generic reconstruction steps with no scope or composition policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .losses import reconstruction_mse
from .modeling import ReconstructionSpec


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    input_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            raise TypeError(f"batch[{key!r}] must be a tensor")
        if key.startswith("x_"):
            moved[key] = value.float().to(device, non_blocking=True) * float(input_scale)
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def train_reconstruction_step(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    spec: ReconstructionSpec,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run one policy-agnostic reconstruction optimizer step."""

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_reconstruction(batch, spec)
    loss = reconstruction_mse(output["pred"], output["target"])
    loss.backward()
    optimizer.step()
    return output, {"loss": float(loss.detach().item())}


@torch.no_grad()
def evaluate_reconstruction_step(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    spec: ReconstructionSpec,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    output = model.forward_reconstruction(batch, spec)
    loss = reconstruction_mse(output["pred"], output["target"])
    return output, {"loss": float(loss.item())}
