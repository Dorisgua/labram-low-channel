"""AdaBrain-compatible classification head for the LaBraM backbone."""

import torch
import torch.nn as nn


class LinearWithConstraint(nn.Linear):
    """Linear layer with AdaBrain's max-norm weight constraint."""

    def __init__(self, *args, max_norm=1.0, flatten=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.flatten = flatten

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data,
            p=2,
            dim=0,
            maxnorm=self.max_norm,
        )
        if self.flatten:
            x = x.flatten(start_dim=1)
        return super().forward(x)


class AdaBrainLaBraMWrapper(nn.Module):
    """Classify from selected post-Transformer LaBraM tokens.

    The backbone can process a completed target channel space while the task
    head reads only selected real-channel tokens.  The CLS token is always
    retained.
    """

    def __init__(
        self,
        backbone,
        num_channels,
        input_num_channels,
        num_t,
        num_classes,
        readout_channel_indices=None,
    ):
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if input_num_channels <= 0:
            raise ValueError(
                f"input_num_channels must be positive, got {input_num_channels}"
            )
        if num_t <= 0:
            raise ValueError(f"num_t must be positive, got {num_t}")

        self.backbone = backbone
        self.backbone.head = nn.Identity()
        self.num_channels = num_channels
        self.input_num_channels = input_num_channels
        self.num_t = num_t
        # Keep expected_tokens as the backbone-token count for compatibility
        # with existing logging and callers.
        self.expected_tokens = num_channels * num_t + 1

        if readout_channel_indices is None:
            readout_channel_indices = list(range(num_channels))
        readout_channel_indices = torch.as_tensor(
            readout_channel_indices,
            dtype=torch.long,
        )
        if readout_channel_indices.ndim != 1:
            raise ValueError(
                "readout_channel_indices must be one-dimensional, got "
                f"shape {tuple(readout_channel_indices.shape)}"
            )
        if readout_channel_indices.numel() == 0:
            raise ValueError("readout_channel_indices must not be empty")
        if readout_channel_indices.unique().numel() != readout_channel_indices.numel():
            raise ValueError(
                "readout_channel_indices must not contain duplicates: "
                f"{readout_channel_indices.tolist()}"
            )
        if (
            readout_channel_indices.min().item() < 0
            or readout_channel_indices.max().item() >= num_channels
        ):
            raise ValueError(
                "readout_channel_indices out of range for "
                f"{num_channels} backbone channels: "
                f"{readout_channel_indices.tolist()}"
            )

        self.register_buffer(
            "readout_channel_indices",
            readout_channel_indices,
            persistent=False,
        )
        self.readout_num_channels = int(readout_channel_indices.numel())
        self.expected_readout_tokens = self.readout_num_channels * num_t + 1

        input_dim = self.expected_readout_tokens * backbone.embed_dim
        self.task_head = LinearWithConstraint(
            input_dim,
            num_classes,
            max_norm=1.0,
            flatten=True,
        )

    def forward(self, x, input_chans=None):
        if x.ndim != 4:
            raise ValueError(
                "AdaBrain LaBraM input must be shaped "
                f"[batch, channels, temporal_patches, patch_size], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_num_channels:
            raise ValueError(
                "AdaBrain input channel-count mismatch: "
                f"got {x.shape[1]}, expected {self.input_num_channels} real channels"
            )
        if input_chans is not None and len(input_chans) != self.input_num_channels + 1:
            raise ValueError(
                "LaBraM input_chans length mismatch: "
                f"got {len(input_chans)}, expected {self.input_num_channels + 1} "
                "(real channels plus CLS)"
            )

        tokens = self.backbone(
            x,
            input_chans=input_chans,
            return_all_tokens=True,
        )
        if tokens.ndim != 3:
            raise ValueError(
                "AdaBrain all-token head expects backbone output shaped "
                f"[batch, tokens, embedding], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] != self.expected_tokens:
            raise ValueError(
                "AdaBrain backbone token-count mismatch: "
                f"got {tokens.shape[1]}, expected {self.expected_tokens} "
                f"({self.num_channels} channels * {self.num_t} temporal patches + 1 CLS)"
            )

        cls_token = tokens[:, :1]
        patch_tokens = tokens[:, 1:].reshape(
            tokens.shape[0],
            self.num_channels,
            self.num_t,
            tokens.shape[-1],
        )
        readout_patch_tokens = patch_tokens.index_select(
            dim=1,
            index=self.readout_channel_indices,
        )
        readout_tokens = torch.cat(
            (cls_token, readout_patch_tokens.flatten(1, 2)),
            dim=1,
        )
        if readout_tokens.shape[1] != self.expected_readout_tokens:
            raise ValueError(
                "AdaBrain readout token-count mismatch: "
                f"got {readout_tokens.shape[1]}, expected "
                f"{self.expected_readout_tokens} "
                f"({self.readout_num_channels} channels * {self.num_t} "
                "temporal patches + 1 CLS)"
            )
        return self.task_head(readout_tokens)

    def get_num_layers(self):
        return self.backbone.get_num_layers()

    def no_weight_decay(self):
        # AdaBrain alignment: position, CLS, and time embeddings use weight decay.
        return set()


class AdaBrainLaBraMMLPWrapper(AdaBrainLaBraMWrapper):
    """CBraMod-style flattened token MLP head on top of LaBraM tokens."""

    def __init__(
        self,
        backbone,
        num_channels,
        input_num_channels,
        num_t,
        num_classes,
        readout_channel_indices=None,
        dropout=0.1,
    ):
        super().__init__(
            backbone=backbone,
            num_channels=num_channels,
            input_num_channels=input_num_channels,
            num_t=num_t,
            num_classes=num_classes,
            readout_channel_indices=readout_channel_indices,
        )
        input_dim = self.expected_readout_tokens * backbone.embed_dim
        hidden_dim = num_t * backbone.embed_dim
        self.task_head = nn.Sequential(
            LinearWithConstraint(input_dim, hidden_dim, max_norm=1.0, flatten=True),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(hidden_dim, backbone.embed_dim, max_norm=1.0),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(backbone.embed_dim, num_classes, max_norm=1.0),
        )
