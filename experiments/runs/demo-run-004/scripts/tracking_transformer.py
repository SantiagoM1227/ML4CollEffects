from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self.ln1(x)
        attn_out, w = self.attn(y, y, y, attn_mask=attn_mask, need_weights=need_weights, average_attn_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, w


class TrackingTransformer(nn.Module):
    """Autoregressive latent dynamics with causal self-attention over fused tokens."""

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
        self.n_layers = int(n_layers)

        self.fuse = nn.Linear(self.z_dim + self.h_dim, self.d_model)
        self.layers = nn.ModuleList([CausalBlock(self.d_model, n_heads, dropout=dropout) for _ in range(self.n_layers)])
        self.head = nn.Linear(self.d_model, self.z_dim)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        z0: torch.Tensor,
        h: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        z0: (B,z_dim)
        h: (B,T,h_dim)
        returns z_seq: (B,T+1,z_dim), and optional attention map (L,H,T,T)
        """
        bsz, T, _ = h.shape
        device = h.device

        z_list: List[torch.Tensor] = [z0]
        token_list: List[torch.Tensor] = []
        zt = z0
        attn_last: Optional[torch.Tensor] = None

        for t in range(T):
            token_t = self.fuse(torch.cat([zt, h[:, t, :]], dim=-1))
            token_list.append(token_t)
            x = torch.stack(token_list, dim=1)  # (B,t+1,d_model)
            mask = self._causal_mask(x.shape[1], device)

            all_layer_weights = []
            for layer in self.layers:
                x, w = layer(x, mask, need_weights=return_attention)
                if return_attention and w is not None:
                    all_layer_weights.append(w)

            dz_t = self.head(x[:, -1, :])
            zt = zt + dz_t
            z_list.append(zt)

            if return_attention and all_layer_weights:
                # stack as (L,B,H,t+1,t+1) for current step
                attn_last = torch.stack(all_layer_weights, dim=0)

        z_seq = torch.stack(z_list, dim=1)

        if not return_attention:
            return z_seq

        if attn_last is None:
            attn_last = torch.zeros(self.n_layers, bsz, 1, T, T, device=device)
        # return only first sample attention for visualization: (L,H,T,T)
        return z_seq, attn_last[:, 0, ...]
