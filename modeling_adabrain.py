"""Minimal AdaBrain classification layer required by Stage2."""

import torch
import torch.nn as nn

# The upstream AdaBrain backbone wrappers are intentionally omitted. Stage2
# builds its own head and only reuses this max-norm linear layer.
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
