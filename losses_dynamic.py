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


def compute_stage1_losses(
    outputs,
    missing_weight: float = 1.0,
    reg_weight: float = 0.01,
    subject_summary_contra_weight: float = 0.0,
    task_summary_contra_weight: float = 0.0,
    subject_correction_contra_weight: float = 0.0,
    task_correction_contra_weight: float = 0.0,
    permute_sub_weight: float = 1.0,
    permute_task_weight: float = 1.0,
    sub_pair_outputs=None,
    task_pair_outputs=None,
):
    """Dynamic Stage 1：重建、正则和 CSLP loss。"""
    required = {
        "h_pred_miss", "h_miss_target", "p_miss",
        "z_sub", "z_task", "d_sub", "d_task",
    }
    missing = required.difference(outputs)
    if missing:
        raise ValueError(f"forward_stage1 output is missing keys: {sorted(missing)}")

    loss_missing = reconstruction_mse(
        outputs["h_pred_miss"],
        outputs["h_miss_target"],
    )
    loss_reg = outputs["d_sub"].square().mean() + outputs["d_task"].square().mean()

    zero = outputs["h_pred_miss"].sum() * 0.0

    if sub_pair_outputs is not None:
        sub_left, sub_right, sub_groups, sub_samples = sub_pair_outputs
        loss_subject_summary_contra = (
            symmetric_info_nce(
                sub_left["z_sub"].reshape(sub_groups, sub_samples, -1),
                sub_right["z_sub"].reshape(sub_groups, sub_samples, -1),
            )
            if subject_summary_contra_weight > 0.0 else zero
        )
        loss_subject_correction_contra = (
            symmetric_info_nce(
                sub_left["d_sub"].flatten(1).reshape(sub_groups, sub_samples, -1),
                sub_right["d_sub"].flatten(1).reshape(sub_groups, sub_samples, -1),
            )
            if subject_correction_contra_weight > 0.0 else zero
        )
        loss_permute_sub = (
            swap_sub_reconstruction(
                sub_left["p_miss"] + sub_right["d_sub"] + sub_left["d_task"],
                sub_left["h_miss_target"],
                sub_right["p_miss"] + sub_left["d_sub"] + sub_right["d_task"],
                sub_right["h_miss_target"],
            )
            if permute_sub_weight > 0.0 else zero
        )
    else:
        loss_subject_summary_contra = zero
        loss_subject_correction_contra = zero
        loss_permute_sub = zero

    if task_pair_outputs is not None:
        task_left, task_right, task_groups, task_samples = task_pair_outputs
        loss_task_summary_contra = (
            symmetric_info_nce(
                task_left["z_task"].reshape(task_groups, task_samples, -1),
                task_right["z_task"].reshape(task_groups, task_samples, -1),
            )
            if task_summary_contra_weight > 0.0 else zero
        )
        loss_task_correction_contra = (
            symmetric_info_nce(
                task_left["d_task"].flatten(1).reshape(task_groups, task_samples, -1),
                task_right["d_task"].flatten(1).reshape(task_groups, task_samples, -1),
            )
            if task_correction_contra_weight > 0.0 else zero
        )
        loss_permute_task = (
            swap_task_reconstruction(
                task_left["p_miss"] + task_left["d_sub"] + task_right["d_task"],
                task_left["h_miss_target"],
                task_right["p_miss"] + task_right["d_sub"] + task_left["d_task"],
                task_right["h_miss_target"],
            )
            if permute_task_weight > 0.0 else zero
        )
    else:
        loss_task_summary_contra = zero
        loss_task_correction_contra = zero
        loss_permute_task = zero

    total_loss = (
        missing_weight * loss_missing
        + reg_weight * loss_reg
        + subject_summary_contra_weight * loss_subject_summary_contra
        + task_summary_contra_weight * loss_task_summary_contra
        + subject_correction_contra_weight * loss_subject_correction_contra
        + task_correction_contra_weight * loss_task_correction_contra
        + permute_sub_weight * loss_permute_sub
        + permute_task_weight * loss_permute_task
    )
    return {
        "missing": loss_missing,
        "reg": loss_reg,
        "subject_summary_contra": loss_subject_summary_contra,
        "task_summary_contra": loss_task_summary_contra,
        "subject_correction_contra": loss_subject_correction_contra,
        "task_correction_contra": loss_task_correction_contra,
        "permute_sub": loss_permute_sub,
        "permute_task": loss_permute_task,
        "total_loss": total_loss,
    }
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
