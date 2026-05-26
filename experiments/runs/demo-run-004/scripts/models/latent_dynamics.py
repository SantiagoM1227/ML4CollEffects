from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def latent_dynamics_loss(z1_pred: torch.Tensor, z1_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mse = F.mse_loss(z1_pred, z1_true)
    mae = F.l1_loss(z1_pred, z1_true)
    rel_l2 = torch.norm(z1_pred - z1_true, dim=-1) / (torch.norm(z1_true, dim=-1) + 1e-12)
    rel_l2 = rel_l2.mean()
    loss = mse + 0.25 * mae
    return loss, {"mse": mse.detach(), "mae": mae.detach(), "rel_l2": rel_l2.detach()}


def temporal_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def wasserstein_1d(x: np.ndarray, y: np.ndarray, n_bins: int = 256) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    cdf_x = np.cumsum(hx)
    cdf_y = np.cumsum(hy)
    dx = (hi - lo) / n_bins
    return float(np.sum(np.abs(cdf_x - cdf_y)) * dx)
