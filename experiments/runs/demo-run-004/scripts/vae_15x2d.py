from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


def cloud6d_to_15x2d_hist(
    z: np.ndarray,
    *,
    bins: int = 64,
    ranges: Optional[Tuple[Tuple[float, float], ...]] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert an Np x 6 cloud into 15 x bins x bins 2D histograms.

    Parameters
    ----------
    z: np.ndarray
        shape [Np,6]
    bins: int
        number of bins per dimension for each 2D histogram.
    ranges: optional
        tuple of 6 (min,max) ranges. If None, computed from data via percentiles.
    normalize: bool
        if True, each 2D histogram is normalized to sum to 1.

    Returns
    -------
    hist: np.ndarray
        shape [15, bins, bins], float32
    """
    assert z.ndim == 2 and z.shape[1] == 6

    if ranges is None:
        # robust range per coordinate
        ranges = []
        for k in range(6):
            lo = float(np.percentile(z[:, k], 0.5))
            hi = float(np.percentile(z[:, k], 99.5))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(z[:, k].min()), float(z[:, k].max())
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            ranges.append((lo, hi))
        ranges = tuple(ranges)

    out = np.zeros((15, bins, bins), dtype=np.float32)

    for i, (a, b) in enumerate(PAIR_IDX_6D):
        H, xedges, yedges = np.histogram2d(
            z[:, a], z[:, b],
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
    """Simple convolutional VAE for (15, H, W) inputs.

    Assumes input is already normalized (e.g., per-channel sum=1).
    Produces latent z in R^latent_dim (default 256).
    """

    def __init__(
        self,
        in_channels: int = 15,
        bins: int = 64,
        latent_dim: int = 256,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bins = bins
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, stride=1, padding=1),
            nn.GELU(),
        )

        # figure out flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, bins, bins)
            h = self.enc(dummy)
            self._enc_shape = h.shape[1:]
            flat = int(h.numel())

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat)
        c, h, w = self._enc_shape
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, 3, stride=1, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.shape[0], *self._enc_shape)
        xhat = self.dec(h)
        return xhat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z


def vae_loss(
    xhat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> Tuple[torch.Tensor, dict]:
    recon = F.mse_loss(xhat, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, {"recon": recon.detach(), "kl": kl.detach()}
