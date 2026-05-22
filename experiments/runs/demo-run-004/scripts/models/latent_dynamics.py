from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsMLP(nn.Module):
    """
    1-step latent dynamics baseline:

        inputs: z0 (B,256), MU (B,3)
        outputs: dz (B,256)
        z1_pred = z0 + dz

    This satisfies Stage 2 on the current dataset which provides only (before, after)
    and MU per sample. When you have lattice sequences, you can swap this with the
    TrackingTransformer model and a sequence dataset.
    """

    def __init__(
        self,
        z_dim: int = 256,
        mu_dim: int = 3,
        hidden: int = 1024,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.mu_dim = int(mu_dim)

        layers = []
        in_dim = z_dim + mu_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z0: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z0, mu], dim=-1)
        dz = self.net(x)
        return z0 + dz


def latent_dynamics_loss(z1_pred: torch.Tensor, z1_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mse = F.mse_loss(z1_pred, z1_true)
    return mse, {"mse": mse.detach()}