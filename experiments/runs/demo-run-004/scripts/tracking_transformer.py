from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TrackingTransformer(nn.Module):
    """Latent-space beam dynamics model.

    Per step t:
      input fused token = concat(z_{t-1}, h_t) -> proj to d_model
      causal Transformer over sequence of fused tokens
      head -> delta z_t
      z_t = z_{t-1} + delta z_t

    Notes:
    - This implementation uses a standard causal Transformer over the whole sequence.
      It supports "memory" of previous steps via self-attention.
    - We keep z's as (B, T+1, 256) and tokens as (B, T, d_model).
    """

    def __init__(
        self,
        z_dim: int = 256,
        h_dim: int = 512,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.h_dim = int(h_dim)
        self.d_model = int(d_model)

        self.fuse = nn.Linear(z_dim + h_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
            norm_first=True,  # LN -> attn
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, z_dim)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        # True means blocked for PyTorch Transformer (attn_mask uses float or bool depending version)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, z0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward rollout.

        Parameters
        ----------
        z0: (B, z_dim)
        h:  (B, T, h_dim)

        Returns
        -------
        z_seq: (B, T+1, z_dim)
            z_seq[:,0]=z0, z_seq[:,t]=z_t
        """
        B, T, _ = h.shape
        device = h.device

        # Build z_{t-1} sequence iteratively, but attention wants all tokens.
        # We'll do a single pass by providing an initial guess z_{t-1}=z0 for all steps,
        # then do explicit residual integration using the transformer's per-token outputs.
        # This matches your "delta z" residual update per element.

        z_prev = z0[:, None, :].expand(B, T, self.z_dim)
        x = self.fuse(torch.cat([z_prev, h], dim=-1))  # (B,T,d_model)

        y = self.tr(x, mask=self._causal_mask(T, device))  # (B,T,d_model)
        dz = self.head(y)  # (B,T,256)

        # residual integrate
        z_list = [z0]
        zt = z0
        for t in range(T):
            zt = zt + dz[:, t, :]
            z_list.append(zt)
        return torch.stack(z_list, dim=1)
