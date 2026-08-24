"""Build, inspect, or train the clean ERP-Core disentanglement model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .engine import move_batch
from .losses import (
    reconstruction_mse,
    swap_sub_reconstruction,
    swap_task_reconstruction,
    symmetric_info_nce,
)
from .modeling import (
    FULL_DIRECT_SPEC,
    FULL_PROTOTYPE_SPEC,
    MISSING_DIRECT_SPEC,
    MISSING_PROTOTYPE_SPEC,
    ComponentMode,
    CompositionMode,
    MissingFillMode,
    OutputBaseMode,
    ReconstructionModel,
    ReconstructionScope,
    ReconstructionSpec,
    StableCore,
)
from .prototype import PrototypeProvider


SPEC_PRESETS = {
    "full_direct": FULL_DIRECT_SPEC,
    "full_prototype": FULL_PROTOTYPE_SPEC,
    "missing_direct": MISSING_DIRECT_SPEC,
    "missing_prototype": MISSING_PROTOTYPE_SPEC,
    "missing_prototype_input_only": ReconstructionSpec(
        scope=ReconstructionScope.MISSING,
        missing_fill=MissingFillMode.PROTOTYPE,
        output_base=OutputBaseMode.NONE,
        component_mode=ComponentMode.IDENTITY,
        composition_mode=CompositionMode.SUM,
    ),
    "missing_prototype_output_only": ReconstructionSpec(
        scope=ReconstructionScope.MISSING,
        missing_fill=MissingFillMode.ZERO,
        output_base=OutputBaseMode.PROTOTYPE,
        component_mode=ComponentMode.IDENTITY,
        composition_mode=CompositionMode.SUM,
    ),
}


def parse_reconstruction_spec(name: str) -> ReconstructionSpec:
    try:
        return SPEC_PRESETS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown reconstruction spec {name!r}; choices={sorted(SPEC_PRESETS)}"
        ) from error


def reconstruction_spec_from_args(args: argparse.Namespace) -> ReconstructionSpec:
    if args.spec is not None:
        return parse_reconstruction_spec(args.spec)
    return ReconstructionSpec(
        scope=ReconstructionScope(args.scope),
        missing_fill=MissingFillMode(args.missing_fill),
        output_base=OutputBaseMode(args.output_base),
        component_mode=ComponentMode(args.component_mode),
        composition_mode=CompositionMode(args.composition_mode),
    )


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_legacy_root(legacy_root: Path) -> None:
    legacy_root = legacy_root.expanduser().resolve()
    if not (legacy_root / "modeling_finetune.py").is_file():
        raise FileNotFoundError(f"modeling_finetune.py not found under {legacy_root}")
    root_string = str(legacy_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)


def _load_base_patch_embed(patch_embed: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "module"):
            if isinstance(checkpoint.get(key), dict):
                source = checkpoint[key]
                break
    if not isinstance(source, dict):
        raise TypeError(f"checkpoint does not contain a state_dict: {checkpoint_path}")

    student = {
        key[len("student.") :]: value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("student.")
    }
    if student:
        source = student
    selected = {
        key[len("patch_embed.") :]: value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("patch_embed.")
    }
    expected = patch_embed.state_dict()
    missing = sorted(set(expected) - set(selected))
    unexpected = sorted(set(selected) - set(expected))
    mismatched = sorted(
        (key, tuple(selected[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(selected)
        if selected[key].shape != expected[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "patch-embedding checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    patch_embed.load_state_dict(selected, strict=True)


def build_erpcore_reconstruction_model(
    *,
    legacy_root: Path,
    cnn_checkpoint: Path,
    prototype_checkpoint: Path | None,
    spec: ReconstructionSpec,
    seed: int = 0,
    unfreeze_cnn: bool = False,
) -> ReconstructionModel:
    """Build the clean model, loading prototype data only when ``spec`` needs it."""

    _add_legacy_root(legacy_root)
    import modeling_finetune  # noqa: F401  # registers the LaBraM timm models
    from Channels_definition import ERPCORE_12_CHANNELS, ERPCORE_28_CHANNELS
    from timm.models import create_model

    seed_everything(seed)
    # just for cnn
    backbone = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=12,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )

    expected_names = tuple(ERPCORE_28_CHANNELS)
    provider = None
    if spec.requires_prototype:
        if prototype_checkpoint is None:
            raise ValueError(
                "prototype_checkpoint is required for this ReconstructionSpec"
            )
        prototype_payload = torch.load(prototype_checkpoint, map_location="cpu")
        prototype_bank = prototype_payload["channel_prototypes"]
        prototype_names = tuple(
            str(name).strip().upper() for name in prototype_payload["ch_names"]
        )
        if prototype_names != expected_names:
            raise ValueError(
                "prototype channel order differs from ERP-Core full order: "
                f"prototype={prototype_names}, expected={expected_names}"
            )
        provider = PrototypeProvider(prototype_bank, channel_names=prototype_names)
    stable_core = StableCore(
        embed_dim=int(backbone.embed_dim),
        num_layers=1,
        num_heads=8,
        dropout=0.1,
    )
    observed_positions = [expected_names.index(name) for name in ERPCORE_12_CHANNELS]
    missing_positions = [
        index for index, name in enumerate(expected_names) if name not in ERPCORE_12_CHANNELS
    ]
    model = ReconstructionModel(
        patch_embed=backbone.patch_embed,
        stable_core=stable_core,
        prototype_provider=provider,
        full_num_channels=len(expected_names),
        observed_channel_positions=observed_positions,
        missing_channel_positions=missing_positions,
        patch_size=200,
        num_t=1,
        train_patch_embed=unfreeze_cnn,
    )
    _load_base_patch_embed(model.patch_embed, cnn_checkpoint)
    model.patch_embed.train(bool(unfreeze_cnn))

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    allowed_prefixes = (
        ("stable_core.", "patch_embed.")
        if unfreeze_cnn
        else ("stable_core.",)
    )
    invalid_names = [
        name for name in trainable_names if not name.startswith(allowed_prefixes)
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if invalid_names:
        raise RuntimeError(f"unexpected trainable parameters: {invalid_names}")
    stable_core_count = sum(
        parameter.numel() for parameter in model.stable_core.parameters()
        if parameter.requires_grad
    )
    if stable_core_count != 1_449_000:
        raise RuntimeError(
            "ERP-Core StableCore must have 1,449,000 trainable parameters, "
            f"got {stable_core_count}"
        )
    cnn_count = sum(
        parameter.numel() for parameter in model.patch_embed.parameters()
        if parameter.requires_grad
    )
    if unfreeze_cnn and cnn_count == 0:
        raise RuntimeError("unfreeze_cnn requested but patch_embed has no trainable parameters")
    if not unfreeze_cnn and trainable_count != stable_core_count:
        raise RuntimeError("frozen CNN run contains trainable parameters outside StableCore")
    return model


def _cosine_schedule(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> np.ndarray:
    """Match the historical per-update warmup plus cosine schedule."""

    total = int(epochs) * int(steps_per_epoch)
    if total <= 0:
        raise ValueError(
            f"invalid schedule length: epochs={epochs}, steps_per_epoch={steps_per_epoch}"
        )
    warmup_steps = min(int(warmup_epochs) * int(steps_per_epoch), total)
    warmup = (
        np.linspace(0.0, float(base_value), warmup_steps)
        if warmup_steps > 0
        else np.asarray([], dtype=np.float64)
    )
    remaining = total - warmup_steps
    if remaining <= 0:
        return warmup
    cosine = np.asarray(
        [
            float(final_value)
            + 0.5
            * (float(base_value) - float(final_value))
            * (1.0 + np.cos(np.pi * step / remaining))
            for step in range(remaining)
        ],
        dtype=np.float64,
    )
    return np.concatenate((warmup, cosine))


def _sample_cslpae_outputs(
    *,
    model: ReconstructionModel,
    dataset,
    property_name: str,
    batch_size: int,
    device: torch.device,
    input_scale: float,
    spec: ReconstructionSpec,
) -> tuple[dict, dict, int, int]:
    left, right, num_groups, samples_per_group = dataset.sample_cslpae_pair_batch(
        property_name,
        batch_size,
    )
    left = move_batch(left, device, input_scale=input_scale)
    right = move_batch(right, device, input_scale=input_scale)
    left_output = model.forward_reconstruction(left, spec)
    right_output = model.forward_reconstruction(right, spec)
    return left_output, right_output, num_groups, samples_per_group


def _compose_swapped_pair(
    *,
    model: ReconstructionModel,
    left: dict,
    right: dict,
    spec: ReconstructionSpec,
    component: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if component == "subject":
        left_d_sub, right_d_sub = right["d_sub"], left["d_sub"]
        left_d_task, right_d_task = left["d_task"], right["d_task"]
    elif component == "task":
        left_d_sub, right_d_sub = left["d_sub"], right["d_sub"]
        left_d_task, right_d_task = right["d_task"], left["d_task"]
    else:
        raise ValueError(f"unsupported swapped component: {component}")
    pred_left = model.compose_prediction(
        left["base"],
        left_d_sub,
        left_d_task,
        spec.composition_mode,
    )
    pred_right = model.compose_prediction(
        right["base"],
        right_d_sub,
        right_d_task,
        spec.composition_mode,
    )
    return pred_left, pred_right


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _save_last_checkpoint(
    *,
    output_dir: Path,
    model: ReconstructionModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    config: dict,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint-last.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Saved final checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path


def train_erpcore(
    args: argparse.Namespace,
    spec: ReconstructionSpec,
    *,
    legacy_root: Path,
    cnn_checkpoint: Path,
    prototype_checkpoint: Path | None,
) -> None:
    """Train the current clean losses; zero-weight losses are never evaluated."""

    if args.dataset != "erpcore":
        raise ValueError(f"unsupported dataset: {args.dataset}")
    if args.sampling != "cslpae":
        raise ValueError(f"unsupported sampler in the clean runner: {args.sampling}")
    if args.optimizer != "adamw" or args.schedule != "cosine":
        raise ValueError(
            f"unsupported optimizer/schedule: {args.optimizer}/{args.schedule}"
        )
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must both be positive")
    if args.num_workers < 0 or args.max_train_steps < 0 or args.warmup_epochs < 0:
        raise ValueError(
            "num_workers, max_train_steps, and warmup_epochs must be non-negative"
        )
    if args.log_interval < 1:
        raise ValueError("log_interval must be positive")
    if not math.isfinite(args.cnn_lr_mult) or args.cnn_lr_mult <= 0.0:
        raise ValueError(
            f"cnn_lr_mult must be finite and positive, got {args.cnn_lr_mult}"
        )
    weights = {
        "sub_contra_weight": args.sub_contra_weight,
        "task_contra_weight": args.task_contra_weight,
        "swap_sub_weight": args.swap_sub_weight,
        "swap_task_weight": args.swap_task_weight,
        "recon_weight": args.recon_weight,
        "missing_mse_weight": args.missing_mse_weight,
        "missing_weight": args.missing_mse_weight,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError(f"loss weights must be finite and non-negative: {weights}")
    if not any(value > 0.0 for value in weights.values()):
        raise ValueError("at least one loss weight must be positive")
    if args.missing_mse_weight != 0.0:
        raise ValueError(
            "separate missing MSE is intentionally disabled; keep "
            "--missing-mse-weight 0"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _add_legacy_root(legacy_root)
    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset

    seed_everything(args.seed)
    cudnn.benchmark = True
    dataset_train, dataset_test, dataset_val = prepare_ERPCORE_cslp_dataset(
        args.data_path,
        sampling_rate=args.sampling_rate,
        normalize_method=args.norm_method,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=torch.utils.data.RandomSampler(dataset_train),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if len(data_loader_train) == 0:
        raise ValueError("ERP-Core training DataLoader is empty")

    model = build_erpcore_reconstruction_model(
        legacy_root=legacy_root,
        cnn_checkpoint=cnn_checkpoint,
        prototype_checkpoint=prototype_checkpoint,
        spec=spec,
        seed=args.seed,
        unfreeze_cnn=args.unfreeze_cnn,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    stable_parameters = [
        parameter for parameter in model.stable_core.parameters()
        if parameter.requires_grad
    ]
    cnn_parameters = [
        parameter for parameter in model.patch_embed.parameters()
        if parameter.requires_grad
    ]
    optimizer_groups = [
        {
            "params": stable_parameters,
            "lr": args.lr,
            "lr_scale": 1.0,
            "group_name": "stable_core",
        }
    ]
    if args.unfreeze_cnn:
        optimizer_groups.append(
            {
                "params": cnn_parameters,
                "lr": args.lr * args.cnn_lr_mult,
                "lr_scale": args.cnn_lr_mult,
                "group_name": "labram_temporal_conv",
            }
        )
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    stable_trainable_count = sum(parameter.numel() for parameter in stable_parameters)
    cnn_trainable_count = sum(parameter.numel() for parameter in cnn_parameters)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.opt_eps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    steps_per_epoch = len(data_loader_train)
    lr_schedule = _cosine_schedule(
        args.lr,
        args.min_lr,
        args.epochs,
        steps_per_epoch,
        args.warmup_epochs,
    )

    config = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "seed": args.seed,
        "scope": spec.scope.value,
        "missing_fill": spec.missing_fill.value,
        "output_base": spec.output_base.value,
        "component_mode": spec.component_mode.value,
        "composition_mode": spec.composition_mode.value,
        "sub_contra_weight": args.sub_contra_weight,
        "task_contra_weight": args.task_contra_weight,
        "swap_sub_weight": args.swap_sub_weight,
        "swap_task_weight": args.swap_task_weight,
        "recon_weight": args.recon_weight,
        "missing_mse_weight": args.missing_mse_weight,
        "missing_weight": args.missing_mse_weight,
        "sampler": args.sampling,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "opt_eps": args.opt_eps,
        "lr": args.lr,
        "base_lr": args.lr,
        "cnn_lr": args.lr * args.cnn_lr_mult if args.unfreeze_cnn else None,
        "cnn_lr_mult": args.cnn_lr_mult,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "schedule": args.schedule,
        "num_workers": args.num_workers,
        "sampling_rate": args.sampling_rate,
        "norm_method": args.norm_method,
        "input_scale": args.input_scale,
        "max_train_steps": args.max_train_steps,
        "device": str(device),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "legacy_root": str(legacy_root),
        "cnn_checkpoint": str(cnn_checkpoint),
        "prototype_checkpoint": (
            str(prototype_checkpoint) if spec.requires_prototype else None
        ),
        "trainable_parameter_count": trainable_count,
        "stable_core_trainable_parameter_count": stable_trainable_count,
        "cnn_trainable_parameter_count": cnn_trainable_count,
        "unfreeze_cnn": args.unfreeze_cnn,
        "cnn_frozen": not args.unfreeze_cnn,
        "cnn_module": "patch_embed (modeling_finetune.TemporalConv)",
        "cnn_trainable_parameter_names": (
            [
                f"patch_embed.{name}"
                for name, parameter in model.patch_embed.named_parameters()
                if parameter.requires_grad
            ]
            if args.unfreeze_cnn
            else []
        ),
        "trainable_scope": (
            "stable_core_plus_labram_temporal_conv"
            if args.unfreeze_cnn
            else "stable_core_only"
        ),
        "patch_embedding": "trainable" if args.unfreeze_cnn else "frozen_eval",
        "checkpoint_selection": "final_epoch",
        "classification_validation": False,
        "split_sizes": {
            "train": len(dataset_train),
            "val": len(dataset_val),
            "test": len(dataset_test),
        },
        "steps_per_epoch": steps_per_epoch,
    }
    config_path = output_dir / "config.json"
    _write_json(config_path, config)
    print(f"Resolved config: {json.dumps(config, sort_keys=True)}", flush=True)
    print(f"Config saved: {config_path}", flush=True)

    global_step = 0
    last_epoch = -1
    stop_requested = False
    start_time = time.time()
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train(True)
        epoch_sums = {
            "loss": 0.0,
            "sub": 0.0,
            "task": 0.0,
            "swap_sub": 0.0,
            "swap_task": 0.0,
            "recon": 0.0,
            "weighted_sub": 0.0,
            "weighted_task": 0.0,
            "weighted_swap_sub": 0.0,
            "weighted_swap_task": 0.0,
            "weighted_recon": 0.0,
        }
        epoch_steps = 0
        for batch in data_loader_train:
            schedule_index = epoch * steps_per_epoch + epoch_steps
            lr = float(lr_schedule[schedule_index])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr * float(parameter_group["lr_scale"])
            optimizer.zero_grad(set_to_none=True)

            batch_size = int(batch["label"].shape[0])
            loss_terms: dict[str, torch.Tensor] = {}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if args.sub_contra_weight > 0.0 or args.swap_sub_weight > 0.0:
                    sub_left, sub_right, sub_groups, sub_samples = (
                        _sample_cslpae_outputs(
                            model=model,
                            dataset=dataset_train,
                            property_name="subject",
                            batch_size=batch_size,
                            device=device,
                            input_scale=args.input_scale,
                            spec=spec,
                        )
                    )
                    if args.sub_contra_weight > 0.0:
                        loss_terms["sub"] = symmetric_info_nce(
                            sub_left["z_sub"].reshape(sub_groups, sub_samples, -1),
                            sub_right["z_sub"].reshape(sub_groups, sub_samples, -1),
                            args.temperature,
                        )
                    if args.swap_sub_weight > 0.0:
                        pred_left, pred_right = _compose_swapped_pair(
                            model=model,
                            left=sub_left,
                            right=sub_right,
                            spec=spec,
                            component="subject",
                        )
                        loss_terms["swap_sub"] = swap_sub_reconstruction(
                            pred_left,
                            sub_left["target"],
                            pred_right,
                            sub_right["target"],
                        )
                if args.task_contra_weight > 0.0 or args.swap_task_weight > 0.0:
                    task_left, task_right, task_groups, task_samples = (
                        _sample_cslpae_outputs(
                            model=model,
                            dataset=dataset_train,
                            property_name="task",
                            batch_size=batch_size,
                            device=device,
                            input_scale=args.input_scale,
                            spec=spec,
                        )
                    )
                    if args.task_contra_weight > 0.0:
                        loss_terms["task"] = symmetric_info_nce(
                            task_left["z_task"].reshape(
                                task_groups,
                                task_samples,
                                -1,
                            ),
                            task_right["z_task"].reshape(
                                task_groups,
                                task_samples,
                                -1,
                            ),
                            args.temperature,
                        )
                    if args.swap_task_weight > 0.0:
                        pred_left, pred_right = _compose_swapped_pair(
                            model=model,
                            left=task_left,
                            right=task_right,
                            spec=spec,
                            component="task",
                        )
                        loss_terms["swap_task"] = swap_task_reconstruction(
                            pred_left,
                            task_left["target"],
                            pred_right,
                            task_right["target"],
                        )
                if args.recon_weight > 0.0:
                    moved_batch = move_batch(
                        batch,
                        device,
                        input_scale=args.input_scale,
                    )
                    reconstruction = model.forward_reconstruction(moved_batch, spec)
                    loss_terms["recon"] = reconstruction_mse(
                        reconstruction["pred"],
                        reconstruction["target"],
                    )
                loss_weights = {
                    "sub": args.sub_contra_weight,
                    "task": args.task_contra_weight,
                    "swap_sub": args.swap_sub_weight,
                    "swap_task": args.swap_task_weight,
                    "recon": args.recon_weight,
                }
                weighted_terms = {
                    name: loss_weights[name] * value
                    for name, value in loss_terms.items()
                }
                total_loss = sum(weighted_terms.values())

            loss_value = float(total_loss.detach().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite training loss: {loss_value}")
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_steps += 1
            epoch_sums["loss"] += loss_value
            for name in ("sub", "task", "swap_sub", "swap_task", "recon"):
                if name in loss_terms:
                    epoch_sums[name] += float(loss_terms[name].detach().item())
                    epoch_sums[f"weighted_{name}"] += float(
                        weighted_terms[name].detach().item()
                    )
            if global_step == 1 or global_step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={epoch_steps}/{steps_per_epoch} "
                    f"global_step={global_step} base_lr={lr:.9g} "
                    f"cnn_lr={(lr * args.cnn_lr_mult if args.unfreeze_cnn else 0.0):.9g} "
                    f"total_loss={loss_value:.9f} "
                    f"L_sub_contra={float(loss_terms.get('sub', torch.tensor(0.0)).detach().item()):.9f} "
                    f"L_task_contra={float(loss_terms.get('task', torch.tensor(0.0)).detach().item()):.9f} "
                    f"L_swap_sub={float(loss_terms.get('swap_sub', torch.tensor(0.0)).detach().item()):.9f} "
                    f"L_swap_task={float(loss_terms.get('swap_task', torch.tensor(0.0)).detach().item()):.9f} "
                    f"W_sub_contra={float(weighted_terms.get('sub', torch.tensor(0.0)).detach().item()):.9f} "
                    f"W_task_contra={float(weighted_terms.get('task', torch.tensor(0.0)).detach().item()):.9f} "
                    f"W_swap_sub={float(weighted_terms.get('swap_sub', torch.tensor(0.0)).detach().item()):.9f} "
                    f"W_swap_task={float(weighted_terms.get('swap_task', torch.tensor(0.0)).detach().item()):.9f}",
                    flush=True,
                )
            if args.max_train_steps and global_step >= args.max_train_steps:
                stop_requested = True
                break

        averages = {
            name: value / max(epoch_steps, 1) for name, value in epoch_sums.items()
        }
        print(
            f"epoch={epoch} summary={json.dumps(averages, sort_keys=True)}",
            flush=True,
        )
        if stop_requested:
            break

    elapsed_seconds = time.time() - start_time
    config["completed_epochs"] = last_epoch + 1
    config["completed_steps"] = global_step
    config["elapsed_seconds"] = elapsed_seconds
    checkpoint_path = _save_last_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=last_epoch,
        global_step=global_step,
        config=config,
    )
    config["checkpoint_last"] = str(checkpoint_path)
    _write_json(config_path, config)
    print(
        "Training complete: "
        f"epochs={last_epoch + 1}, steps={global_step}, "
        f"elapsed_seconds={elapsed_seconds:.3f}",
        flush=True,
    )


def get_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "train"), default="inspect")
    parser.add_argument("--legacy-root", type=Path, default=repo_root)
    parser.add_argument("--cnn-checkpoint", type=Path, default=None)
    parser.add_argument("--prototype-checkpoint", type=Path, default=None)
    parser.add_argument("--spec", choices=sorted(SPEC_PRESETS), default=None)
    parser.add_argument(
        "--scope",
        choices=[value.value for value in ReconstructionScope],
        default=ReconstructionScope.FULL.value,
    )
    parser.add_argument(
        "--missing-fill",
        choices=[value.value for value in MissingFillMode],
        default=MissingFillMode.NOT_APPLICABLE.value,
    )
    parser.add_argument(
        "--output-base",
        choices=[value.value for value in OutputBaseMode],
        default=OutputBaseMode.NONE.value,
    )
    parser.add_argument(
        "--component-mode",
        choices=[value.value for value in ComponentMode],
        default=ComponentMode.IDENTITY.value,
    )
    parser.add_argument(
        "--composition-mode",
        choices=[value.value for value in CompositionMode],
        default=CompositionMode.SUM.value,
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", choices=("erpcore",), default="erpcore")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument(
        "--norm-method",
        choices=("z_score", "0.1mv", "95"),
        default="z_score",
    )
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=("adamw",), default="adamw")
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cnn-lr-mult", type=float, default=0.1)
    parser.add_argument("--unfreeze-cnn", action="store_true")
    parser.add_argument("--freeze-cnn", action="store_false", dest="unfreeze_cnn")
    parser.set_defaults(unfreeze_cnn=False)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--schedule", choices=("cosine",), default="cosine")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sampling", choices=("cslpae",), default="cslpae")
    parser.add_argument("--sub-contra-weight", type=float, default=0.0)
    parser.add_argument("--task-contra-weight", type=float, default=0.0)
    parser.add_argument("--swap-sub-weight", type=float, default=0.0)
    parser.add_argument("--swap-task-weight", type=float, default=0.0)
    parser.add_argument("--recon-weight", type=float, default=0.0)
    parser.add_argument("--missing-mse-weight", type=float, default=0.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    legacy_root = args.legacy_root.expanduser().resolve()
    cnn_checkpoint = args.cnn_checkpoint or legacy_root / "checkpoints/labram-base.pth"
    prototype_checkpoint = args.prototype_checkpoint or (
        legacy_root / "docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth"
    )
    spec = reconstruction_spec_from_args(args)
    if args.mode == "train":
        if args.output_dir is None:
            raise ValueError("--output-dir is required in train mode")
        if not args.run_name:
            raise ValueError("--run-name is required in train mode")
        if args.data_path is None:
            raise ValueError("--data-path is required in train mode")
        train_erpcore(
            args,
            spec,
            legacy_root=legacy_root,
            cnn_checkpoint=cnn_checkpoint,
            prototype_checkpoint=prototype_checkpoint,
        )
        return

    model = build_erpcore_reconstruction_model(
        legacy_root=legacy_root,
        cnn_checkpoint=cnn_checkpoint,
        prototype_checkpoint=prototype_checkpoint,
        spec=spec,
        seed=args.seed,
        unfreeze_cnn=args.unfreeze_cnn,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        json.dumps(
            {
                "mode": "inspect_only_no_training",
                "spec": {
                    "scope": spec.scope.value,
                    "missing_fill": spec.missing_fill.value,
                    "output_base": spec.output_base.value,
                    "component_mode": spec.component_mode.value,
                    "composition_mode": spec.composition_mode.value,
                },
                "device": str(device),
                "trainable_parameters": trainable_count,
                "unfreeze_cnn": args.unfreeze_cnn,
                "cnn_frozen": not args.unfreeze_cnn,
                "cnn_lr_mult": args.cnn_lr_mult,
                "stable_core_trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.stable_core.parameters()
                    if parameter.requires_grad
                ),
                "cnn_trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.patch_embed.parameters()
                    if parameter.requires_grad
                ),
                "cnn_module": (
                    f"{type(model.patch_embed).__module__}."
                    f"{type(model.patch_embed).__name__}"
                ),
                "cnn_trainable_parameter_names": [
                    f"patch_embed.{name}"
                    for name, parameter in model.patch_embed.named_parameters()
                    if parameter.requires_grad
                ],
                "prototype_channel_order": (
                    list(model.prototype_provider.channel_names)
                    if model.prototype_provider is not None
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
