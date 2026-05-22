from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MuNormalizer:
    """Affine normalizer for MU parameters.

    Assumes MU is shape (..., P). Normalizes each component with (x - mean)/std.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self):
        self.mean = torch.as_tensor(self.mean).float()
        self.std = torch.as_tensor(self.std).float()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.mean) / (self.std + 1e-12)


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, n_freqs: int = 16, s_scale: float = 1.0):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.s_scale = float(s_scale)

        # wavelengths in arbitrary units; user can interpret based on lattice length scale
        # log-spaced frequencies
        k = torch.arange(self.n_freqs).float()
        self.register_buffer("freq", 2.0 ** k)  # (K,)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s shape (...,) -> (..., 2*K)"""
        s = s[..., None] * self.s_scale  # (...,1)
        arg = 2.0 * math.pi * s * self.freq  # (...,K)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class ElementTokenizer(nn.Module):
    """Tokenize lattice element parameters.

    Inputs:
      - mu_t: (..., P)
      - s_t: (...,) cumulative position

    Output:
      - h_t: (..., d_token=512)

    Architecture:
      normalized_MU -> 3-layer MLP (GELU) -> R^512
      positional enc ([sin,cos]) -> linear -> R^512
      sum -> h_t
    """

    def __init__(
        self,
        mu_dim: int = 3,
        d_token: int = 512,
        n_pos_freqs: int = 16,
        mu_normalizer: Optional[MuNormalizer] = None,
    ):
        super().__init__()
        self.mu_dim = int(mu_dim)
        self.d_token = int(d_token)
        self.mu_normalizer = mu_normalizer

        self.mu_mlp = nn.Sequential(
            nn.Linear(mu_dim, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token),
        )

        self.pos_enc = SinCosPositionalEncoding(n_freqs=n_pos_freqs)
        self.pos_proj = nn.Linear(2 * n_pos_freqs, d_token)

    def forward(self, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.mu_normalizer is not None:
            mu = self.mu_normalizer(mu)

        e = self.mu_mlp(mu)
        p = self.pos_proj(self.pos_enc(s))
        return e + p
