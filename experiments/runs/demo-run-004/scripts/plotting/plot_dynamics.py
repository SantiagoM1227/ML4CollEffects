from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

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


def save_hist_triplet(
    truth: np.ndarray,
    pred: np.ndarray,
    out_png: str | Path,
    *,
    n_channels: int = 6,
    eps: float = 1e-18,
) -> None:
    """
    truth, pred: (15,B,B). Saves (truth, pred, absdiff) for first n_channels.
    """
    out_png = Path(out_png)
    n_channels = min(n_channels, truth.shape[0])

    vmax = float(max(truth.max(), pred.max()))
    pos = np.concatenate([truth[truth > 0], pred[pred > 0]], axis=0)
    vmin = float(np.percentile(pos, 5)) if pos.size else 1e-12
    vmin = max(vmin, 1e-12)

    fig, axes = plt.subplots(n_channels, 3, figsize=(9, 2.2 * n_channels))
    for i in range(n_channels):
        axes[i, 0].imshow(truth[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[i, 0].set_title(f"Truth ch {i}")
        axes[i, 1].imshow(pred[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[i, 1].set_title(f"Pred ch {i}")
        axes[i, 2].imshow(np.abs(truth[i] - pred[i]), origin="lower", aspect="auto")
        axes[i, 2].set_title("|diff|")
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)