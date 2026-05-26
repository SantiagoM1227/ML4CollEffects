from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class MuNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self):
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        return (mu - self.mean) / (self.std + 1e-12)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "MuNormalizer":
        return cls(mean=state["mean"], std=state["std"])


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, n_freqs: int = 16, s_scale: float = 1.0):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.s_scale = float(s_scale)
        self.register_buffer("freq", 2.0 ** torch.arange(self.n_freqs, dtype=torch.float32))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s[..., None] * self.s_scale
        arg = 2.0 * math.pi * s * self.freq
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class ElementTokenizer(nn.Module):
    """Tokenize lattice element parameters into d_token embeddings."""

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
        self.n_pos_freqs = int(n_pos_freqs)
        self.mu_normalizer = mu_normalizer

        self.mu_mlp = nn.Sequential(
            nn.Linear(self.mu_dim, self.d_token),
            nn.GELU(),
            nn.Linear(self.d_token, self.d_token),
            nn.GELU(),
            nn.Linear(self.d_token, self.d_token),
        )
        self.pos_enc = SinCosPositionalEncoding(n_freqs=self.n_pos_freqs)
        self.pos_proj = nn.Linear(2 * self.n_pos_freqs, self.d_token)

    def forward(self, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.mu_normalizer is not None:
            mu = self.mu_normalizer(mu)
        e = self.mu_mlp(mu)
        p = self.pos_proj(self.pos_enc(s))
        return e + p
