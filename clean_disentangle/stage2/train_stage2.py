"""Unified Stage2 downstream classifier for four Transformer input modes.

Modes are ``full``, ``observed_only``, ``prototype`` and ``dynamic``.  The
classifier forward receives only the raw view required by the selected mode;
the dynamic path never constructs a reconstruction target from ``x_full``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clean_disentangle.modeling import ComponentMode, CompositionMode
from clean_disentangle.prototype import PrototypeProvider, select_positions


MODES = ("full", "observed_only", "prototype", "dynamic")
DEFAULT_STAGE1 = Path("outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337")


def add_legacy_root(root: Path) -> None:
    root = root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not (root / "modeling_finetune.py").is_file():
        raise FileNotFoundError(f"LaBraM source not found: {root}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_backbone(backbone: nn.Module, checkpoint_path: Path) -> None:
    import utils

    payload = torch.load(checkpoint_path, map_location="cpu")
    source: Any = payload
    if isinstance(payload, dict):
        for key in ("model", "module"):
            if isinstance(payload.get(key), dict):
                source = payload[key]
                break
    if not isinstance(source, dict):
        raise TypeError(f"checkpoint has no state_dict: {checkpoint_path}")
    student = {
        key[len("student.") :]: value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("student.")
    }
    if student:
        source = student
    expected = backbone.state_dict()
    source = dict(source)
    for key in ("head.weight", "head.bias"):
        if key in source and key in expected and tuple(source[key].shape) != tuple(expected[key].shape):
            source.pop(key)
    for key in list(source):
        if "relative_position_index" in key:
            source.pop(key)
    print(f"LABRAM_PRETRAINED_CHECKPOINT={checkpoint_path.resolve()}")
    utils.load_state_dict(backbone, source)


def patch_tokens(backbone: nn.Module, eeg: torch.Tensor, expected_channels: int, *, patch_size: int = 200, num_t: int = 1, trainable: bool = False) -> torch.Tensor:
    if eeg.ndim == 3:
        if eeg.shape[-1] != patch_size * num_t:
            raise ValueError(f"unexpected EEG length: {tuple(eeg.shape)}")
        eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], num_t, patch_size)
    if eeg.ndim != 4 or eeg.shape[1] != expected_channels:
        raise ValueError(f"expected [B,{expected_channels},{num_t},{patch_size}], got {tuple(eeg.shape)}")
    if trainable and torch.is_grad_enabled():
        tokens = backbone.patch_embed(eeg)
    else:
        with torch.no_grad():
            tokens = backbone.patch_embed(eeg)
    expected = (eeg.shape[0], expected_channels * num_t, backbone.embed_dim)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"patch token shape {tuple(tokens.shape)} != {expected}")
    return tokens


def load_prototypes(path: Path, expected_names: tuple[str, ...]) -> tuple[PrototypeProvider, list[int]]:
    payload = torch.load(path, map_location="cpu")
    names = tuple(str(value).strip().upper() for value in payload["ch_names"])
    if names != expected_names:
        raise ValueError(f"prototype channel order mismatch: {names} != {expected_names}")
    indices = payload.get("input_chans_index")
    if indices is None:
        import utils
        indices = utils.get_input_chans(list(expected_names))
    if len(indices) != len(expected_names) + 1:
        raise ValueError("prototype input_chans_index must include CLS plus every channel")
    return PrototypeProvider(payload["channel_prototypes"], channel_names=names), [int(v) for v in indices]


def build_stage1_dynamic(config: dict[str, Any], checkpoint: Path):
    from clean_disentangle.modeling import MISSING_PROTOTYPE_SPEC
    from clean_disentangle.run import build_erpcore_reconstruction_model

    expected = ("missing", "prototype", "prototype", "identity", "sum")
    actual = tuple(config.get(key) for key in ("scope", "missing_fill", "output_base", "component_mode", "composition_mode"))
    if actual != expected:
        raise RuntimeError(f"dynamic requires Stage1 C config, got {actual}")
    model = build_erpcore_reconstruction_model(
        legacy_root=Path(config["legacy_root"]),
        cnn_checkpoint=Path(config["cnn_checkpoint"]),
        prototype_checkpoint=Path(config["prototype_checkpoint"]),
        spec=MISSING_PROTOTYPE_SPEC,
        seed=int(config.get("seed", 0)),
        unfreeze_cnn=False,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.patch_embed.eval()
    model.stable_core.eval()
    return model


def formal_probe_match(checkpoint: Path, formal_checkpoint: Path | None = None) -> None:
    if formal_checkpoint is None:
        probe_config = checkpoint.parent.parent / "evaluation/probe/probe_config.json"
        if not probe_config.is_file():
            raise FileNotFoundError(f"dynamic checkpoint has no formal probe config: {probe_config}")
        formal = Path(str(load_json(probe_config)["checkpoint"])).expanduser().resolve()
    else:
        formal = formal_checkpoint.expanduser().resolve()
    actual = checkpoint.expanduser().resolve()
    print(f"FORMAL_PROBE_CHECKPOINT={formal}")
    print(f"STAGE1_DYNAMIC_CHECKPOINT={actual}")
    print(f"MATCH={'YES' if formal == actual else 'NO'}")
    if formal != actual:
        raise RuntimeError("dynamic checkpoint does not match formal Stage1 probe checkpoint")


class TransformerClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, input_mode: str, channel_positions: list[int], *, dynamic_model=None, prototype_provider=None, prototype_positions=None, last_n_blocks: int = 12, train_cnn: bool = False):
        super().__init__()
        if input_mode not in MODES:
            raise ValueError(f"unknown input mode: {input_mode}")
        self.backbone = backbone
        self.input_mode = input_mode
        self.channel_positions = list(channel_positions)
        self.dynamic_model = dynamic_model
        self.prototype_provider = prototype_provider
        self.prototype_positions = prototype_positions
        self.full_channels = 28
        self.observed_channels = 12
        self.num_t = 1
        self.last_n_blocks = int(last_n_blocks)
        self.train_cnn = bool(train_cnn)
        if self.input_mode == "dynamic" and self.train_cnn:
            raise ValueError("dynamic mode uses the frozen Stage1 TemporalConv; TRAIN_CNN must be 0")
        if not 0 <= self.last_n_blocks <= len(backbone.blocks):
            raise ValueError(f"last_n_blocks must be in [0,{len(backbone.blocks)}]")
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.train_cnn:
            for parameter in backbone.patch_embed.parameters():
                parameter.requires_grad = True
        if self.dynamic_model is not None:
            for parameter in self.dynamic_model.parameters():
                parameter.requires_grad = False
        for block in backbone.blocks[-self.last_n_blocks:] if self.last_n_blocks else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in backbone.norm.parameters():
            parameter.requires_grad = True
        if backbone.fc_norm is not None:
            for parameter in backbone.fc_norm.parameters():
                parameter.requires_grad = True
        from modeling_adabrain import LinearWithConstraint

        self.num_input_tokens = 12 if input_mode == "observed_only" else 28
        self.classifier_input_dim = (self.num_input_tokens + 1) * int(backbone.embed_dim)
        self.task_head = LinearWithConstraint(self.classifier_input_dim, 12, max_norm=1.0, flatten=True)
        for parameter in self.task_head.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dynamic_model is not None:
            self.dynamic_model.eval()
            self.dynamic_model.patch_embed.eval()
            self.dynamic_model.stable_core.eval()
        self.backbone.patch_embed.train(bool(mode and self.train_cnn))
        for block in self.backbone.blocks[:-self.last_n_blocks] if self.last_n_blocks else self.backbone.blocks:
            block.eval()
        return self

    def _dynamic_complete(self, x_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            corrector = self.dynamic_model
            h_obs = corrector.patch_tokens(x_obs, expected_channels=12)
            observed = corrector.observed_token_positions.to(x_obs.device)
            missing = corrector.missing_token_positions.to(x_obs.device)
            provider = corrector.prototype_provider
            p_miss = provider.get_missing(x_obs.shape[0], missing_channel_positions=corrector.missing_channel_positions, num_t=1, device=x_obs.device, dtype=h_obs.dtype)
            context = h_obs.new_zeros(x_obs.shape[0], 28, corrector.stable_core.embed_dim)
            context = context.index_copy(1, observed, h_obs).index_copy(1, missing, p_miss)
            rep = corrector.stable_core.encode_tokens(context)
            components = corrector.build_components(rep, missing, ComponentMode.IDENTITY)
            pred = corrector.compose_prediction(
                p_miss,
                components["d_sub"],
                components["d_task"],
                CompositionMode.SUM,
            )
            complete = context.index_copy(1, missing, pred)
            return complete

    def build_transformer_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "full":
            return patch_tokens(self.backbone, x, 28, trainable=self.train_cnn)
        if self.input_mode == "observed_only":
            return patch_tokens(self.backbone, x, 12, trainable=self.train_cnn)
        if self.input_mode == "dynamic":
            return self._dynamic_complete(x)
        h_obs = patch_tokens(self.backbone, x, 12, trainable=self.train_cnn)
        observed = torch.as_tensor(self._observed_positions, dtype=torch.long, device=x.device)
        missing = torch.as_tensor(self._missing_positions, dtype=torch.long, device=x.device)
        p_miss = self.prototype_provider.get_missing(x.shape[0], missing_channel_positions=missing, num_t=1, device=x.device, dtype=h_obs.dtype)
        complete = h_obs.new_zeros(x.shape[0], 28, self.backbone.embed_dim)
        complete = complete.index_copy(1, observed, h_obs)
        if self.input_mode == "prototype":
            return complete.index_copy(1, missing, p_miss)
        raise RuntimeError(f"unhandled input mode: {self.input_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.build_transformer_input(x)
        batch_size, num_tokens, _ = tokens.shape
        x = torch.cat((self.backbone.cls_token.expand(batch_size, -1, -1), tokens), dim=1)
        indices = torch.as_tensor(self.channel_positions, dtype=torch.long, device=x.device)
        if self.backbone.pos_embed is not None:
            used = self.backbone.pos_embed[:, indices]
            pos = used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, 1, -1).flatten(1, 2)
            x = x + torch.cat((used[:, :1, :].expand(batch_size, -1, -1), pos), dim=1)
        if self.backbone.time_embed is not None:
            x[:, 1:] = x[:, 1:] + self.backbone.time_embed[:, :1, :].unsqueeze(1).expand(batch_size, num_tokens, -1, -1).flatten(1, 2)
        x = self.backbone.pos_drop(x)
        for block in self.backbone.blocks:
            x = block(x, rel_pos_bias=None)
        x = self.backbone.norm(x)
        if self.backbone.fc_norm is not None:
            x = self.backbone.fc_norm(x)
        return self.task_head(x)


def make_model(args: argparse.Namespace, stage1_config: dict[str, Any] | None):
    add_legacy_root(args.legacy_root)
    import modeling_finetune  # noqa: F401
    from timm.models import create_model
    from Channels_definition import ERPCORE_12_CHANNELS, ERPCORE_28_CHANNELS

    backbone = create_model("labram_base_patch200_200", pretrained=False, num_classes=12, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    load_backbone(backbone, args.labram_checkpoint)
    full_names = tuple(ERPCORE_28_CHANNELS)
    observed_positions = [full_names.index(name) for name in ERPCORE_12_CHANNELS]
    missing_positions = [i for i, name in enumerate(full_names) if name not in ERPCORE_12_CHANNELS]
    prototype_provider = None
    target_indices = None
    prototype_path = args.prototype_checkpoint
    if args.input_mode in ("prototype", "dynamic"):
        prototype_provider, target_indices = load_prototypes(prototype_path, full_names)
    if args.input_mode == "observed_only":
        import utils
        channel_positions = utils.get_input_chans(list(ERPCORE_12_CHANNELS))
    elif args.input_mode == "full":
        import utils
        channel_positions = utils.get_input_chans(list(ERPCORE_28_CHANNELS))
    else:
        channel_positions = target_indices or []
    dynamic_model = None
    if args.input_mode == "dynamic":
        if args.require_probe_match:
            formal_probe_match(args.stage1_checkpoint, args.formal_probe_checkpoint)
        dynamic_model = build_stage1_dynamic(stage1_config, args.stage1_checkpoint)
    model = TransformerClassifier(backbone, args.input_mode, channel_positions, dynamic_model=dynamic_model, prototype_provider=prototype_provider, prototype_positions=target_indices, last_n_blocks=args.last_n_blocks, train_cnn=args.train_cnn)
    model._observed_positions = observed_positions
    model._missing_positions = missing_positions
    return model, full_names, ERPCORE_12_CHANNELS


def validate_dynamic_metadata(args: argparse.Namespace, config: dict[str, Any]) -> None:
    expected_spec = {
        "scope": "missing",
        "missing_fill": "prototype",
        "output_base": "prototype",
        "component_mode": "identity",
        "composition_mode": "sum",
    }
    mismatches = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in expected_spec.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("dynamic requires a compatible Missing+Prototype+D Stage1 checkpoint: " + "; ".join(mismatches))
    if int(config.get("full_num_channels", 28)) != 28:
        raise RuntimeError(f"dynamic requires 28 full channels, got {config.get('full_num_channels')!r}")
    if int(config.get("embed_dim", 200)) != 200:
        raise RuntimeError(f"dynamic requires embedding dimension 200, got {config.get('embed_dim')!r}")
    expected_observed = ["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2"]
    observed = config.get("observed_channels")
    if observed is not None and list(observed) != expected_observed:
        raise RuntimeError(f"dynamic observed channel order mismatch: {observed!r} != {expected_observed!r}")
    print("Stage1 dynamic compatibility: PASS")


def print_resolved_summary(args: argparse.Namespace, model: nn.Module, output_dir: Path) -> None:
    stage1_used = args.input_mode == "dynamic"
    prototype_used = args.input_mode in ("prototype", "dynamic")
    blocks = list(range(max(0, len(model.backbone.blocks) - args.last_n_blocks), len(model.backbone.blocks)))
    print("=" * 60)
    print("Resolved Experiment")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Mode: {args.input_mode}")
    print(f"Seed: {args.seed}")
    print(f"Input: {'28 full channels' if args.input_mode == 'full' else '12 observed channels'}")
    print(f"Transformer tokens: {model.num_input_tokens} + CLS")
    print(f"Classifier input dim: {model.classifier_input_dim}")
    print(f"Stage1 checkpoint: {args.stage1_checkpoint.resolve() if stage1_used else 'NOT USED'}")
    print(f"Stage1 StableCore: {'USED, frozen/eval' if stage1_used else 'NOT USED'}")
    print(f"Prototype: {args.prototype_checkpoint.resolve() if prototype_used else 'NOT USED'}")
    print(f"LaBraM pretrained: {args.labram_checkpoint.resolve()}")
    print(f"Transformer depth: {len(model.backbone.blocks)}")
    print(f"Trainable block indices: {blocks}")
    print(f"CNN / patch_embed: {'TRAINABLE' if args.train_cnn else 'FROZEN'}")
    print(f"Epochs / batch size: {args.epochs} / {args.batch_size}")
    print(f"Optimizer / LR / weight decay: AdamW / {args.lr} / {args.weight_decay}")
    print(f"CNN LR: {args.lr * args.cnn_lr_mult if args.train_cnn else 'NOT USED (CNN frozen)'}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 60)


@torch.no_grad()
def evaluate(model, loader, device):
    import utils
    model.eval(); outputs, targets = [], []
    for batch in loader:
        source = batch["x_full"] if model.input_mode == "full" else batch["x_obs"]
        outputs.append(model(source.to(device)).cpu())
        targets.append(batch["label"].cpu())
    output = torch.cat(outputs).numpy(); target = torch.cat(targets).numpy()
    return {k: float(v) for k, v in utils.get_metrics(output, target, ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"], False).items()}


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    stage1_config = load_json(args.stage1_config) if args.input_mode == "dynamic" else None
    if args.input_mode == "dynamic":
        validate_dynamic_metadata(args, stage1_config)
        if args.require_probe_match:
            formal_probe_match(args.stage1_checkpoint, args.formal_probe_checkpoint)
    if args.dry_run:
        for required in (args.labram_checkpoint, args.data_path):
            if not required.exists():
                raise FileNotFoundError(required)
        if args.input_mode in ("prototype", "dynamic") and not args.prototype_checkpoint.exists():
            raise FileNotFoundError(args.prototype_checkpoint)
        if args.input_mode == "dynamic" and not args.stage1_checkpoint.exists():
            raise FileNotFoundError(args.stage1_checkpoint)
        print(f"DRY_RUN_PASS mode={args.input_mode} seed={args.seed} last_n_blocks={args.last_n_blocks}")
        return
    model, full_names, observed_names = make_model(args, stage1_config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    cnn_params = [p for name, p in trainable if name.startswith("backbone.patch_embed.")]
    other_params = [p for name, p in trainable if not name.startswith("backbone.patch_embed.")]
    groups = [{"params": other_params, "lr": args.lr, "base_lr": args.lr}]
    if cnn_params:
        groups.append({"params": cnn_params, "lr": args.lr * args.cnn_lr_mult, "base_lr": args.lr * args.cnn_lr_mult})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, eps=1e-8)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for _, parameter in trainable}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer parameters do not exactly match requires_grad=True parameters")
    trainable_names = [name for name, _ in trainable]
    print(f"MODE={args.input_mode}")
    print_resolved_summary(args, model, args.output_dir)
    print(f"num_transformer_input_tokens={model.num_input_tokens}")
    print(f"classifier_input_dim={model.classifier_input_dim}")
    print(f"transformer_total_blocks={len(model.backbone.blocks)}")
    print(f"trainable_last_n={args.last_n_blocks}")
    print(f"Trainable parameter names: {json.dumps(trainable_names)}")
    print(f"Trainable parameter count: {sum(p.numel() for _, p in trainable)}")
    print(f"Frozen parameter count: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    if args.audit_only:
        from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset
        _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(args.data_path, sampling_rate=200, normalize_method="z_score")
        batch = next(iter(DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)))
        source = batch["x_full"] if args.input_mode == "full" else batch["x_obs"]
        with torch.no_grad(): logits = model(source.to(device))
        print(f"AUDIT_PASS raw={tuple(source.shape)} logits={tuple(logits.shape)}")
        return
    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset
    train_ds, test_ds, val_ds = prepare_ERPCORE_cslp_dataset(args.data_path, sampling_rate=200, normalize_method="z_score")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {"stage": "stage2", "exp_name": args.exp_name, "stage2_input_mode": args.input_mode, "train_cnn": bool(args.train_cnn), "cnn_lr_mult": args.cnn_lr_mult, "labram_checkpoint": str(args.labram_checkpoint.resolve()), "stage1_checkpoint": str(args.stage1_checkpoint.resolve()) if args.input_mode == "dynamic" else None, "stage1_config": str(args.stage1_config.resolve()) if args.input_mode == "dynamic" else None, "prototype_checkpoint": str(args.prototype_checkpoint.resolve()) if args.input_mode in ("prototype", "dynamic") else None, "prototype_source": "fixed_channel_prototype_bank" if args.input_mode in ("prototype", "dynamic") else None, "observed_channels": list(observed_names), "observed_positions": model._observed_positions, "missing_channels": [name for index, name in enumerate(full_names) if index in model._missing_positions], "missing_positions": model._missing_positions, "num_transformer_input_tokens": model.num_input_tokens, "classifier_input_dim": model.classifier_input_dim, "transformer_total_blocks": len(model.backbone.blocks), "trainable_last_n": args.last_n_blocks, "trainable_block_indices": list(range(max(0, len(model.backbone.blocks) - args.last_n_blocks), len(model.backbone.blocks))), "classifier_protocol": "all_token_linear_with_constraint", "optimizer": "adamw", "lr": args.lr, "cnn_lr": args.lr * args.cnn_lr_mult if args.train_cnn else None, "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs, "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "full_channel_access": "NONE" if args.input_mode != "full" else "ALLOWED", "require_probe_match": bool(args.require_probe_match), "formal_probe_checkpoint": str(args.formal_probe_checkpoint.resolve()) if args.formal_probe_checkpoint else None, "checkpoint_selection": ["best-bacc", "best-acc", "last"], "trainable_parameter_names": trainable_names}
    (args.output_dir / "config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    criterion = nn.CrossEntropyLoss(); best_bacc = -math.inf; best_acc = -math.inf
    fields = ["epoch", "train_loss", "val_accuracy", "val_balanced_accuracy", "val_cohen_kappa", "val_f1_weighted", "test_accuracy", "test_balanced_accuracy", "test_cohen_kappa", "test_f1_weighted"]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=fields).writeheader()
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        resume_config = payload.get("config", {})
        for key in ("stage2_input_mode", "classifier_input_dim", "trainable_last_n"):
            if resume_config.get(key) != resolved.get(key):
                raise RuntimeError(f"resume config mismatch for {key}: {resume_config.get(key)!r} != {resolved.get(key)!r}")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    def save(path, epoch): torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": resolved}, path)
    for epoch in range(start_epoch, args.epochs):
        model.train(True); losses=[]
        lr = args.lr * (float(epoch+1)/max(args.warmup_epochs,1) if epoch < args.warmup_epochs else 0.5*(1+math.cos(math.pi*(epoch-args.warmup_epochs)/max(args.epochs-args.warmup_epochs,1))))
        scale = lr / args.lr if args.lr else 1.0
        for group in optimizer.param_groups: group["lr"] = group["base_lr"] * scale
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); source=batch["x_full"] if args.input_mode == "full" else batch["x_obs"]; logits=model(source.float().to(device)); loss=criterion(logits,batch["label"].to(device)); loss.backward(); optimizer.step(); losses.append(float(loss.detach().item()))
        val=evaluate(model,val_loader,device); test=evaluate(model,test_loader,device); rec={"epoch":epoch,"train_loss":float(np.mean(losses)),**{f"val_{k}":v for k,v in val.items()},**{f"test_{k}":v for k,v in test.items()}}
        with (args.output_dir/"metrics.csv").open("a",newline="",encoding="utf-8") as handle: csv.DictWriter(handle,fieldnames=fields).writerow({k:rec.get(k,"") for k in fields})
        print(json.dumps(rec,sort_keys=True),flush=True); save(args.output_dir/"checkpoint-last.pth",epoch)
        if val["balanced_accuracy"] > best_bacc: best_bacc=val["balanced_accuracy"]; save(args.output_dir/"checkpoint-best-bacc.pth",epoch)
        if val["accuracy"] > best_acc: best_acc=val["accuracy"]; save(args.output_dir/"checkpoint-best-acc.pth",epoch)


def get_args():
    parser=argparse.ArgumentParser(description=__doc__); root=Path(__file__).resolve().parents[2]; stage1=root/DEFAULT_STAGE1
    parser.add_argument("--input-mode", choices=MODES, default="dynamic"); parser.add_argument("--exp-name", default="stage2_default"); parser.add_argument("--stage1-checkpoint",type=Path,default=stage1/"checkpoints/checkpoint-last.pth"); parser.add_argument("--stage1-config",type=Path,default=stage1/"config.json"); parser.add_argument("--labram-checkpoint",type=Path,default=Path("../LabraM-Git-Diff/checkpoints/labram-base.pth").resolve()); parser.add_argument("--prototype-checkpoint",type=Path,default=Path("../LabraM-Git-Diff/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth").resolve()); parser.add_argument("--data-path",type=Path,default=root/"../CSLP-AE/data_preparation/simple_data.pt"); parser.add_argument("--output-dir",type=Path,default=root/"outputs/stage2"); parser.add_argument("--legacy-root",type=Path,default=root/"../LabraM-Git-Diff"); parser.add_argument("--device",default="cuda"); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--epochs",type=int,default=50); parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--num-workers",type=int,default=4); parser.add_argument("--lr",type=float,default=5e-5); parser.add_argument("--cnn-lr-mult",type=float,default=0.1); parser.add_argument("--train-cnn",action="store_true"); parser.add_argument("--weight-decay",type=float,default=0.05); parser.add_argument("--warmup-epochs",type=int,default=5); parser.add_argument("--last-n-blocks",type=int,default=12); parser.add_argument("--audit-only",action="store_true"); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--resume",type=Path,default=None); parser.add_argument("--require-probe-match",action="store_true"); parser.add_argument("--formal-probe-checkpoint",type=Path,default=None); return parser.parse_args()


if __name__ == "__main__": main(get_args())
