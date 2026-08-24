"""Composable full/missing token reconstruction with a scope-agnostic stable core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from .prototype import (
    PrototypeProvider,
    channel_positions_to_token_positions,
    select_positions,
)


class ReconstructionScope(str, Enum):
    FULL = "full"
    MISSING = "missing"


class MissingFillMode(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    PROTOTYPE = "prototype"
    ZERO = "zero"


class OutputBaseMode(str, Enum):
    NONE = "none"
    PROTOTYPE = "prototype"


class ComponentMode(str, Enum):
    IDENTITY = "identity"


class CompositionMode(str, Enum):
    SUM = "sum"


@dataclass(frozen=True)
class ReconstructionSpec:
    """Orthogonal choices for one reconstruction experiment."""

    scope: ReconstructionScope
    missing_fill: MissingFillMode
    output_base: OutputBaseMode
    component_mode: ComponentMode = ComponentMode.IDENTITY
    composition_mode: CompositionMode = CompositionMode.SUM

    def __post_init__(self) -> None:
        fields = {
            "scope": (self.scope, ReconstructionScope),
            "missing_fill": (self.missing_fill, MissingFillMode),
            "output_base": (self.output_base, OutputBaseMode),
            "component_mode": (self.component_mode, ComponentMode),
            "composition_mode": (self.composition_mode, CompositionMode),
        }
        for name, (value, expected_type) in fields.items():
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}, got {type(value).__name__}"
                )
        if self.scope is ReconstructionScope.FULL:
            if self.missing_fill is not MissingFillMode.NOT_APPLICABLE:
                raise ValueError(
                    "FULL reconstruction requires missing_fill=NOT_APPLICABLE"
                )
        elif self.missing_fill is MissingFillMode.NOT_APPLICABLE:
            raise ValueError(
                "MISSING reconstruction requires missing_fill=PROTOTYPE or ZERO"
            )

    @property
    def requires_prototype(self) -> bool:
        """Whether this exact combination needs prototype data at runtime."""

        return (
            self.missing_fill is MissingFillMode.PROTOTYPE
            or self.output_base is OutputBaseMode.PROTOTYPE
        )


FULL_DIRECT_SPEC = ReconstructionSpec(
    scope=ReconstructionScope.FULL,
    missing_fill=MissingFillMode.NOT_APPLICABLE,
    output_base=OutputBaseMode.NONE,
)
FULL_PROTOTYPE_SPEC = ReconstructionSpec(
    scope=ReconstructionScope.FULL,
    missing_fill=MissingFillMode.NOT_APPLICABLE,
    output_base=OutputBaseMode.PROTOTYPE,
)
MISSING_DIRECT_SPEC = ReconstructionSpec(
    scope=ReconstructionScope.MISSING,
    missing_fill=MissingFillMode.ZERO,
    output_base=OutputBaseMode.NONE,
)
MISSING_PROTOTYPE_SPEC = ReconstructionSpec(
    scope=ReconstructionScope.MISSING,
    missing_fill=MissingFillMode.PROTOTYPE,
    output_base=OutputBaseMode.PROTOTYPE,
)


class StableCore(nn.Module):
    """Shared/subject/task token encoders with no reconstruction policy knowledge."""

    def __init__(
        self,
        *,
        embed_dim: int = 200,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_heads < 1 or embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must divide embed_dim={embed_dim}"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.embed_dim = embed_dim

        def make_encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.shared_encoder = make_encoder()
        self.sub_encoder = make_encoder()
        self.task_encoder = make_encoder()
        self.shared_norm = nn.LayerNorm(embed_dim)
        self.sub_norm = nn.LayerNorm(embed_dim)
        self.task_norm = nn.LayerNorm(embed_dim)

    def encode_tokens(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"token D={tokens.shape[-1]} does not match embed_dim={self.embed_dim}"
            )
        if tokens.shape[1] < 1:
            raise ValueError("StableCore requires at least one token")
        if not torch.isfinite(tokens).all():
            raise ValueError("StableCore input contains NaN or Inf")

        shared_tokens = self.shared_norm(self.shared_encoder(tokens))
        sub_tokens = self.sub_norm(self.sub_encoder(shared_tokens))
        task_tokens = self.task_norm(self.task_encoder(shared_tokens))
        return {
            "shared_tokens": shared_tokens,
            "sub_tokens": sub_tokens,
            "task_tokens": task_tokens,
            "z_sub": sub_tokens.mean(dim=1),
            "z_task": task_tokens.mean(dim=1),
        }

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_tokens(tokens)


def build_components(
    representation: Mapping[str, torch.Tensor],
    positions: Sequence[int] | torch.Tensor,
    mode: ComponentMode,
) -> dict[str, torch.Tensor]:
    """Transform selected branch tokens into reconstruction components."""

    if mode is not ComponentMode.IDENTITY:
        raise NotImplementedError(f"Unsupported component mode: {mode}")
    try:
        sub_tokens = representation["sub_tokens"]
        task_tokens = representation["task_tokens"]
    except KeyError as error:
        raise KeyError(f"StableCore representation lacks {error.args[0]!r}") from error
    d_sub = select_positions(sub_tokens, positions)
    d_task = select_positions(task_tokens, positions)
    if d_sub.shape != d_task.shape:
        raise RuntimeError(
            f"component shapes differ: {tuple(d_sub.shape)} vs {tuple(d_task.shape)}"
        )
    return {"d_sub": d_sub, "d_task": d_task}


def get_output_base(
    context: Mapping[str, Any],
    mode: OutputBaseMode,
) -> torch.Tensor:
    """Return only the additive baseline selected by output-base policy."""

    target = context["target"]
    if mode is OutputBaseMode.NONE:
        return torch.zeros_like(target)
    if mode is OutputBaseMode.PROTOTYPE:
        prototype = context["prototype_selected"]
        if prototype is None:
            raise RuntimeError(
                "output_base=PROTOTYPE requires an available PrototypeProvider"
            )
        if prototype.shape != target.shape:
            raise RuntimeError(
                "prototype/target shapes differ: "
                f"{tuple(prototype.shape)} vs {tuple(target.shape)}"
            )
        return prototype
    raise NotImplementedError(f"Unsupported output-base mode: {mode}")


def compose_prediction(
    base: torch.Tensor,
    d_sub: torch.Tensor,
    d_task: torch.Tensor,
    mode: CompositionMode = CompositionMode.SUM,
) -> torch.Tensor:
    """Compose components without knowing scope, positions, sampler, or target."""

    if not (base.shape == d_sub.shape == d_task.shape):
        raise ValueError(
            "composition shapes differ: "
            f"base={tuple(base.shape)}, d_sub={tuple(d_sub.shape)}, "
            f"d_task={tuple(d_task.shape)}"
        )
    if mode is CompositionMode.SUM:
        return base + d_sub + d_task
    raise NotImplementedError(f"Unsupported composition mode: {mode}")


class ReconstructionModel(nn.Module):
    """One reconstruction pipeline configured by an immutable specification."""

    def __init__(
        self,
        *,
        patch_embed: nn.Module,
        stable_core: StableCore,
        prototype_provider: PrototypeProvider | None = None,
        full_num_channels: int,
        observed_channel_positions: Sequence[int],
        missing_channel_positions: Sequence[int],
        patch_size: int,
        num_t: int,
        train_patch_embed: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.stable_core = stable_core
        self.prototype_provider = prototype_provider
        self.full_num_channels = int(full_num_channels)
        self.patch_size = int(patch_size)
        self.num_t = int(num_t)
        self.train_patch_embed = bool(train_patch_embed)
        if self.full_num_channels < 1:
            raise ValueError("full_num_channels must be >= 1")
        if self.patch_size < 1 or self.num_t < 1:
            raise ValueError(
                f"patch_size and num_t must be >= 1, got {self.patch_size}, {self.num_t}"
            )
        if prototype_provider is not None:
            if prototype_provider.num_channels != self.full_num_channels:
                raise ValueError(
                    f"prototype channels={prototype_provider.num_channels} do not match "
                    f"full_num_channels={self.full_num_channels}"
                )
            if prototype_provider.embed_dim != stable_core.embed_dim:
                raise ValueError(
                    f"prototype D={prototype_provider.embed_dim} does not match "
                    f"StableCore D={stable_core.embed_dim}"
                )

        observed = torch.as_tensor(observed_channel_positions, dtype=torch.long).flatten()
        missing = torch.as_tensor(missing_channel_positions, dtype=torch.long).flatten()
        if observed.numel() == 0 or missing.numel() == 0:
            raise ValueError("observed and missing channel positions must both be non-empty")
        combined = torch.cat((observed, missing))
        expected = torch.arange(self.full_num_channels, dtype=torch.long)
        if torch.unique(combined).numel() != self.full_num_channels or not torch.equal(
            torch.sort(combined).values,
            expected,
        ):
            raise ValueError(
                "observed/missing channel positions must be a disjoint full partition: "
                f"observed={observed.tolist()}, missing={missing.tolist()}"
            )
        self.register_buffer("observed_channel_positions", observed, persistent=False)
        self.register_buffer("missing_channel_positions", missing, persistent=False)

        for parameter in self.patch_embed.parameters():
            parameter.requires_grad = self.train_patch_embed
        self.patch_embed.train(self.train_patch_embed)

    @property
    def full_token_positions(self) -> torch.Tensor:
        return torch.arange(self.full_num_channels * self.num_t, dtype=torch.long)

    @property
    def observed_token_positions(self) -> torch.Tensor:
        return channel_positions_to_token_positions(
            self.observed_channel_positions,
            self.num_t,
        )

    @property
    def missing_token_positions(self) -> torch.Tensor:
        return channel_positions_to_token_positions(
            self.missing_channel_positions,
            self.num_t,
        )

    def train(self, mode: bool = True) -> "ReconstructionModel":
        super().train(mode)
        self.patch_embed.train(mode if self.train_patch_embed else False)
        return self

    def patch_tokens(self, eeg: torch.Tensor, *, expected_channels: int) -> torch.Tensor:
        """Apply the tokenizer and return flattened ``[B,C*num_t,D]`` tokens."""

        if eeg.ndim == 3:
            if eeg.shape[-1] != self.patch_size * self.num_t:
                raise ValueError(
                    f"EEG samples={eeg.shape[-1]} do not match patch_size*num_t="
                    f"{self.patch_size * self.num_t}"
                )
            eeg = eeg.reshape(
                eeg.shape[0],
                eeg.shape[1],
                self.num_t,
                self.patch_size,
            )
        elif eeg.ndim == 4:
            if eeg.shape[2:] != (self.num_t, self.patch_size):
                raise ValueError(
                    f"EEG patch shape={tuple(eeg.shape[2:])} does not match "
                    f"{(self.num_t, self.patch_size)}"
                )
        else:
            raise ValueError(f"EEG must be [B,C,T] or [B,C,num_t,patch], got {tuple(eeg.shape)}")
        if eeg.shape[1] != int(expected_channels):
            raise ValueError(
                f"EEG channels={eeg.shape[1]} do not match expected={expected_channels}"
            )

        if self.train_patch_embed:
            tokens = self.patch_embed(eeg)
        else:
            with torch.no_grad():
                tokens = self.patch_embed(eeg)
        expected_shape = (
            eeg.shape[0],
            int(expected_channels) * self.num_t,
            self.stable_core.embed_dim,
        )
        if tokens.shape != expected_shape:
            raise ValueError(
                f"patch tokens have shape {tuple(tokens.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(tokens).all():
            raise ValueError("patch tokens contain NaN or Inf")
        return tokens

    def prepare_reconstruction_context(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        scope: ReconstructionScope,
        missing_fill: MissingFillMode,
        prototype_required: bool,
    ) -> dict[str, Any]:
        """Prepare input, target, positions, and prototypes without composition logic."""

        provider = self.prototype_provider
        if prototype_required and provider is None:
            raise RuntimeError(
                "this ReconstructionSpec requires prototype data, but no "
                "PrototypeProvider was configured"
            )

        if scope is ReconstructionScope.FULL:
            if missing_fill is not MissingFillMode.NOT_APPLICABLE:
                raise ValueError("FULL context requires missing_fill=NOT_APPLICABLE")
            h_full = self.patch_tokens(
                batch["x_full"],
                expected_channels=self.full_num_channels,
            )
            target_positions = self.full_token_positions.to(h_full.device)
            prototype_selected = None
            if prototype_required:
                prototype_selected = provider.get_full(
                    h_full.shape[0],
                    num_t=self.num_t,
                    device=h_full.device,
                    dtype=h_full.dtype,
                )
            input_tokens = h_full
            target = h_full.detach()
            target_channel_positions = torch.arange(
                self.full_num_channels,
                dtype=torch.long,
                device=h_full.device,
            )
        elif scope is ReconstructionScope.MISSING:
            if missing_fill is MissingFillMode.NOT_APPLICABLE:
                raise ValueError("MISSING context requires PROTOTYPE or ZERO fill")
            h_obs = self.patch_tokens(
                batch["x_obs"],
                expected_channels=int(self.observed_channel_positions.numel()),
            )
            target_positions = self.missing_token_positions.to(h_obs.device)
            observed_positions = self.observed_token_positions.to(h_obs.device)
            prototype_selected = None
            if prototype_required:
                prototype_selected = provider.get_missing(
                    h_obs.shape[0],
                    missing_channel_positions=self.missing_channel_positions,
                    num_t=self.num_t,
                    device=h_obs.device,
                    dtype=h_obs.dtype,
                )

            input_tokens = h_obs.new_zeros(
                h_obs.shape[0],
                self.full_num_channels * self.num_t,
                self.stable_core.embed_dim,
            )
            input_tokens = input_tokens.index_copy(1, observed_positions, h_obs)
            if missing_fill is MissingFillMode.PROTOTYPE:
                if prototype_selected is None:
                    raise RuntimeError(
                        "missing_fill=PROTOTYPE requires an available PrototypeProvider"
                    )
                fill_tokens = prototype_selected
            elif missing_fill is MissingFillMode.ZERO:
                fill_tokens = h_obs.new_zeros(
                    h_obs.shape[0],
                    target_positions.numel(),
                    self.stable_core.embed_dim,
                )
            else:
                raise NotImplementedError(f"Unsupported missing fill: {missing_fill}")
            input_tokens = input_tokens.index_copy(1, target_positions, fill_tokens)

            h_full_target = self.patch_tokens(
                batch["x_full"],
                expected_channels=self.full_num_channels,
            )
            target = select_positions(h_full_target, target_positions).detach()
            target_channel_positions = self.missing_channel_positions.to(h_obs.device)
        else:
            raise NotImplementedError(f"Unsupported reconstruction scope: {scope}")

        if prototype_selected is not None and prototype_selected.shape != target.shape:
            raise RuntimeError(
                "context prototype/target shapes differ: "
                f"{tuple(prototype_selected.shape)} vs {tuple(target.shape)}"
            )
        return {
            "input_tokens": input_tokens,
            "target": target,
            "target_positions": target_positions,
            "prototype_selected": prototype_selected,
            "scope": scope,
            "metadata": {
                "num_t": self.num_t,
                "full_num_channels": self.full_num_channels,
                "target_channel_positions": target_channel_positions,
                "target_token_positions": target_positions,
                "observed_channel_positions": self.observed_channel_positions.to(
                    input_tokens.device
                ),
                "missing_channel_positions": self.missing_channel_positions.to(
                    input_tokens.device
                ),
                "missing_fill": missing_fill,
            },
        }

    def build_components(
        self,
        representation: Mapping[str, torch.Tensor],
        positions: Sequence[int] | torch.Tensor,
        mode: ComponentMode,
    ) -> dict[str, torch.Tensor]:
        """Component-transform extension seam; the pipeline calls only this method."""

        return build_components(representation, positions, mode)

    def get_output_base(
        self,
        context: Mapping[str, Any],
        mode: OutputBaseMode,
    ) -> torch.Tensor:
        return get_output_base(context, mode)

    def compose_prediction(
        self,
        base: torch.Tensor,
        d_sub: torch.Tensor,
        d_task: torch.Tensor,
        mode: CompositionMode,
    ) -> torch.Tensor:
        """Composition-policy extension seam; the pipeline calls only this method."""

        return compose_prediction(base, d_sub, d_task, mode)

    def _forward_reconstruction(
        self,
        batch: Mapping[str, torch.Tensor],
        spec: ReconstructionSpec,
    ) -> dict[str, Any]:
        context = self.prepare_reconstruction_context(
            batch,
            scope=spec.scope,
            missing_fill=spec.missing_fill,
            prototype_required=spec.requires_prototype,
        )
        representation = self.stable_core.encode_tokens(context["input_tokens"])
        components = self.build_components(
            representation,
            context["target_positions"],
            spec.component_mode,
        )
        base = self.get_output_base(context, spec.output_base)
        pred = self.compose_prediction(
            base,
            components["d_sub"],
            components["d_task"],
            spec.composition_mode,
        )
        if pred.shape != context["target"].shape:
            raise RuntimeError(
                f"prediction/target shapes differ: {tuple(pred.shape)} vs "
                f"{tuple(context['target'].shape)}"
            )
        if not torch.isfinite(pred).all() or not torch.isfinite(context["target"]).all():
            raise RuntimeError("prediction or target contains NaN or Inf")
        return {
            **representation,
            **components,
            "base": base,
            "pred": pred,
            "target": context["target"],
            "input_tokens": context["input_tokens"],
            "target_positions": context["target_positions"],
            "prototype_selected": context["prototype_selected"],
            "scope": context["scope"],
            "metadata": context["metadata"],
            "spec": spec,
        }

    def forward_reconstruction(
        self,
        batch: Mapping[str, torch.Tensor],
        spec: ReconstructionSpec,
    ) -> dict[str, Any]:
        if not isinstance(spec, ReconstructionSpec):
            raise TypeError(f"spec must be ReconstructionSpec, got {type(spec).__name__}")
        return self._forward_reconstruction(batch, spec)

    def forward_full_direct(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.forward_reconstruction(batch, FULL_DIRECT_SPEC)

    def forward_full_prototype(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.forward_reconstruction(batch, FULL_PROTOTYPE_SPEC)

    def forward_missing_direct(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.forward_reconstruction(batch, MISSING_DIRECT_SPEC)

    def forward_missing_prototype(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.forward_reconstruction(batch, MISSING_PROTOTYPE_SPEC)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        spec: ReconstructionSpec,
    ) -> dict[str, Any]:
        return self.forward_reconstruction(batch, spec)
