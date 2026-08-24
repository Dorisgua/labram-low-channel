"""Model-agnostic reconstruction and contrastive losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def reconstruction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over already-composed prediction and target tensors."""

    if pred.shape != target.shape:
        raise ValueError(
            f"prediction/target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("prediction or target contains NaN or Inf")
    return F.mse_loss(pred, target)


def _paired_reconstruction_mse(
    pred_left: torch.Tensor,
    target_left: torch.Tensor,
    pred_right: torch.Tensor,
    target_right: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        reconstruction_mse(pred_left, target_left)
        + reconstruction_mse(pred_right, target_right)
    )


def swap_sub_reconstruction(
    pred_left: torch.Tensor,
    target_left: torch.Tensor,
    pred_right: torch.Tensor,
    target_right: torch.Tensor,
) -> torch.Tensor:
    """Symmetric MSE for predictions built with exchanged subject components."""

    return _paired_reconstruction_mse(
        pred_left,
        target_left,
        pred_right,
        target_right,
    )


def swap_task_reconstruction(
    pred_left: torch.Tensor,
    target_left: torch.Tensor,
    pred_right: torch.Tensor,
    target_right: torch.Tensor,
) -> torch.Tensor:
    """Symmetric MSE for predictions built with exchanged task components."""

    return _paired_reconstruction_mse(
        pred_left,
        target_left,
        pred_right,
        target_right,
    )


def symmetric_info_nce(
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Diagonal-positive InfoNCE averaged over both matching directions."""

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"temperature must be a positive finite number, got {temperature}"
        )
    if z_left.shape != z_right.shape:
        raise ValueError(
            f"pair shapes differ: {tuple(z_left.shape)} vs {tuple(z_right.shape)}"
        )
    if z_left.ndim not in {2, 3}:
        raise ValueError(
            f"pair latents must be [B,D] or [G,M,D], got {tuple(z_left.shape)}"
        )
    if not torch.isfinite(z_left).all() or not torch.isfinite(z_right).all():
        raise ValueError("pair latents contain NaN or Inf")

    left = F.normalize(z_left, dim=-1)
    right = F.normalize(z_right, dim=-1)
    if left.ndim == 2:
        logits = left @ right.transpose(0, 1) / temperature
        labels = torch.arange(left.shape[0], device=left.device)
    else:
        logits = torch.einsum("gic,gjc->gij", left, right) / temperature
        labels = torch.arange(left.shape[1], device=left.device)
        labels = labels.unsqueeze(0).expand(left.shape[0], -1)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(-1, -2), labels)
    )
