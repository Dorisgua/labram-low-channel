"""Prototype-bank access and explicit channel-to-token selection utilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _as_unique_positions(
    positions: Sequence[int] | torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    position_tensor = torch.as_tensor(positions, dtype=torch.long).flatten()
    if position_tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if torch.any(position_tensor < 0):
        raise ValueError(f"{name} contains negative positions: {position_tensor.tolist()}")
    if torch.unique(position_tensor).numel() != position_tensor.numel():
        raise ValueError(f"{name} contains duplicates: {position_tensor.tolist()}")
    return position_tensor


def channel_positions_to_token_positions(
    channel_positions: Sequence[int] | torch.Tensor,
    num_t: int,
) -> torch.Tensor:
    """Expand channel positions into channel-major flattened token positions."""

    num_t = int(num_t)
    if num_t < 1:
        raise ValueError(f"num_t must be >= 1, got {num_t}")
    channels = _as_unique_positions(channel_positions, name="channel_positions")
    offsets = torch.arange(num_t, dtype=torch.long, device=channels.device)
    return (channels[:, None] * num_t + offsets[None, :]).reshape(-1)


def select_positions(tokens: torch.Tensor, positions: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Select flattened token positions from a ``[B, N, D]`` tensor."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
    position_tensor = _as_unique_positions(positions, name="token_positions").to(tokens.device)
    if int(position_tensor.max()) >= tokens.shape[1]:
        raise IndexError(
            f"token position {int(position_tensor.max())} exceeds N={tokens.shape[1]}"
        )
    return tokens.index_select(1, position_tensor)


class PrototypeProvider(nn.Module):
    """Expose full and selected tokens from one channel-ordered prototype bank.

    The bank may be shaped ``[C, D]`` (one prototype repeated over ``num_t``)
    or ``[C, num_t, D]`` (time-token-specific prototypes).
    """

    def __init__(
        self,
        prototype_bank: torch.Tensor,
        *,
        channel_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        bank = torch.as_tensor(prototype_bank, dtype=torch.float32)
        if bank.ndim not in {2, 3}:
            raise ValueError(
                "prototype_bank must be [C,D] or [C,num_t,D], "
                f"got {tuple(bank.shape)}"
            )
        if not torch.isfinite(bank).all():
            raise ValueError("prototype_bank contains NaN or Inf")
        if bank.shape[0] < 1 or bank.shape[-1] < 1:
            raise ValueError(f"invalid prototype_bank shape: {tuple(bank.shape)}")

        if channel_names is None:
            names = tuple(str(index) for index in range(bank.shape[0]))
        else:
            names = tuple(str(name).strip().upper() for name in channel_names)
            if len(names) != bank.shape[0]:
                raise ValueError(
                    f"channel_names has {len(names)} entries but bank has {bank.shape[0]} channels"
                )
            if len(set(names)) != len(names):
                raise ValueError(f"channel_names contains duplicates: {names}")

        self.register_buffer("prototype_bank", bank.contiguous())
        self.channel_names = names

    @property
    def num_channels(self) -> int:
        return int(self.prototype_bank.shape[0])

    @property
    def embed_dim(self) -> int:
        return int(self.prototype_bank.shape[-1])

    def get_full(
        self,
        batch_size: int,
        *,
        num_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return channel-major full prototype tokens shaped ``[B,C*num_t,D]``."""

        batch_size = int(batch_size)
        num_t = int(num_t)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_t < 1:
            raise ValueError(f"num_t must be >= 1, got {num_t}")

        bank = self.prototype_bank
        if bank.ndim == 2:
            bank = bank[:, None, :].expand(-1, num_t, -1)
        elif bank.shape[1] != num_t:
            raise ValueError(
                f"prototype num_t={bank.shape[1]} does not match runtime num_t={num_t}"
            )
        full = bank.reshape(self.num_channels * num_t, self.embed_dim)
        full = full.to(device=device, dtype=dtype)
        return full.unsqueeze(0).expand(batch_size, -1, -1)

    def get_selected(
        self,
        batch_size: int,
        *,
        channel_positions: Sequence[int] | torch.Tensor,
        num_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        full = self.get_full(
            batch_size,
            num_t=num_t,
            device=device,
            dtype=dtype,
        )
        token_positions = channel_positions_to_token_positions(channel_positions, num_t)
        return select_positions(full, token_positions)

    def get_missing(
        self,
        batch_size: int,
        *,
        missing_channel_positions: Sequence[int] | torch.Tensor,
        num_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return missing prototypes and audit that they are a full-bank selection."""

        full = self.get_full(
            batch_size,
            num_t=num_t,
            device=device,
            dtype=dtype,
        )
        token_positions = channel_positions_to_token_positions(
            missing_channel_positions,
            num_t,
        )
        missing = self.get_selected(
            batch_size,
            channel_positions=missing_channel_positions,
            num_t=num_t,
            device=device,
            dtype=dtype,
        )
        expected = select_positions(full, token_positions)
        if not torch.equal(missing, expected):
            max_diff = float((missing - expected).abs().max().item())
            raise RuntimeError(
                "Missing prototype is not the corresponding full-bank selection: "
                f"max_abs_diff={max_diff:.9g}"
            )
        return missing
