from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def save_loss_curves(history: Dict[str, Sequence[float]], out_png: str | Path, title: str) -> None:
    out_png = Path(out_png)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    for k, v in history.items():
        ax.plot(list(v), label=k)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_latent_histograms(
    mu: np.ndarray,
    logvar: np.ndarray,
    out_png: str | Path,
    max_dims: int = 32,
) -> None:
    out_png = Path(out_png)
    D = mu.shape[1]
    dshow = min(D, max_dims)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].hist(mu[:, :dshow].reshape(-1), bins=120, alpha=0.9)
    axes[0].set_title(f"Latent mu distribution (first {dshow} dims)")

    axes[1].hist(logvar[:, :dshow].reshape(-1), bins=120, alpha=0.9)
    axes[1].set_title(f"Latent logvar distribution (first {dshow} dims)")

    for ax in axes:
        ax.grid(alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_recon_grid_15ch(
    x: np.ndarray,
    xhat: np.ndarray,
    out_png: str | Path,
    *,
    n_channels: int = 6,
    eps: float = 1e-18,
) -> None:
    """
    x, xhat: (15,B,B). Writes a 2 x n_channels comparison with LogNorm.
    """
    out_png = Path(out_png)
    n_channels = min(n_channels, x.shape[0])

    vmax = float(max(x.max(), xhat.max()))
    pos = np.concatenate([x[x > 0], xhat[xhat > 0]], axis=0)
    vmin = float(np.percentile(pos, 5)) if pos.size else 1e-12
    vmin = max(vmin, 1e-12)

    fig, axes = plt.subplots(2, n_channels, figsize=(2.4 * n_channels, 5.0))
    for i in range(n_channels):
        axes[0, i].imshow(x[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[0, i].set_title(f"X ch {i}")
        axes[1, i].imshow(xhat[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[1, i].set_title(f"Xhat ch {i}")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)