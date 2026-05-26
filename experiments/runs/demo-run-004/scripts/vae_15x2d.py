from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_IDX_6D = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
)


def compute_global_ranges_6d(
    clouds: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
    pad_frac: float = 0.05,
) -> Tuple[Tuple[float, float], ...]:
    """Compute global, fixed ranges per 6D coordinate.

    clouds expected shape (N, Np, 6) or (Np, 6).
    """
    if clouds.ndim == 2:
        if clouds.shape[1] != 6:
            raise ValueError(f"Expected shape (Np,6), got {clouds.shape}")
        flat = clouds
    elif clouds.ndim == 3:
        if clouds.shape[2] != 6:
            raise ValueError(f"Expected shape (N,Np,6), got {clouds.shape}")
        flat = clouds.reshape(-1, 6)
    else:
        raise ValueError(f"Expected 2D or 3D clouds, got {clouds.shape}")

    ranges = []
    for k in range(6):
        x = flat[:, k]
        lo = float(np.percentile(x, lo_pct))
        hi = float(np.percentile(x, hi_pct))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo >= hi:
            lo = float(np.min(x))
            hi = float(np.max(x))
        if lo >= hi:
            lo, hi = lo - 1.0, hi + 1.0
        pad = pad_frac * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    return tuple(ranges)


def cloud6d_to_15x2d_hist(
    z: np.ndarray,
    *,
    bins: int = 64,
    ranges: Optional[Tuple[Tuple[float, float], ...]] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert (Np,6) cloud into (15,bins,bins) pairwise histograms."""
    if z.ndim != 2 or z.shape[1] != 6:
        raise ValueError(f"Expected shape (Np,6), got {z.shape}")

    if ranges is None:
        ranges = compute_global_ranges_6d(z)

    out = np.zeros((15, bins, bins), dtype=np.float32)
    for i, (a, b) in enumerate(PAIR_IDX_6D):
        H, _, _ = np.histogram2d(
            z[:, a],
            z[:, b],
            bins=bins,
            range=[ranges[a], ranges[b]],
            density=False,
        )
        H = H.astype(np.float32)
        if normalize:
            H = H / (H.sum() + eps)
        out[i] = H
    return out


class ConvVAE2D(nn.Module):
    """Convolutional VAE for (15,bins,bins) normalized histogram targets."""

    def __init__(
        self,
        in_channels: int = 15,
        bins: int = 64,
        latent_dim: int = 256,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.bins = int(bins)
        self.latent_dim = int(latent_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, stride=1, padding=1),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.bins, self.bins)
            h = self.enc(dummy)
            self._enc_shape = h.shape[1:]
            self._enc_flat = int(h.numel())

        self.fc_mu = nn.Linear(self._enc_flat, self.latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, self.latent_dim)

        c, _, _ = self._enc_shape
        self.fc_dec = nn.Linear(self.latent_dim, self._enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.in_channels, 3, stride=1, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.shape[0], *self._enc_shape)
        return self.dec(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decode_logits(z)
        b, c, h, w = logits.shape
        # channel-wise spatial distribution per pair: nonnegative and sums to 1
        probs = torch.softmax(logits.view(b, c, h * w), dim=-1).view(b, c, h, w)
        return probs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z


def _channelwise_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Returns mean JS over batch+channels for spatial distributions."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=(-1, -2))
    kl_qm = (q * (q.log() - m.log())).sum(dim=(-1, -2))
    js = 0.5 * (kl_pm + kl_qm)
    return js.mean()


def vae_loss(
    xhat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
    js_weight: float = 1.0,
    mae_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mse = F.mse_loss(xhat, x)
    mae = F.l1_loss(xhat, x)
    js = _channelwise_js_divergence(xhat, x)

    # Standard VAE KL: E_b[ -0.5 * sum_d(1 + logvar - mu^2 - exp(logvar)) ]
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl = kl_per_sample.mean()

    recon = mse + mae_weight * mae + js_weight * js
    loss = recon + float(beta) * kl

    return loss, {
        "recon": recon.detach(),
        "mse": mse.detach(),
        "mae": mae.detach(),
        "js": js.detach(),
        "kl": kl.detach(),
    }
