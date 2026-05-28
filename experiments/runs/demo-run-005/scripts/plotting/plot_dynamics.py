from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_dynamics_curves(history: Dict[str, Sequence[float]], out_png: str | Path) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history.get("train_mse", []), label="train_mse")
    axes[0, 0].plot(history.get("val_mse", []), label="val_mse")
    axes[0, 0].set_title("Latent MSE")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(history.get("train_rel_l2", []), label="train_rel_l2")
    axes[0, 1].plot(history.get("val_rel_l2", []), label="val_rel_l2")
    axes[0, 1].set_title("Relative L2")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(history.get("val_temporal_corr", []), label="val_temporal_corr")
    axes[1, 0].plot(history.get("val_wasserstein", []), label="val_wasserstein")
    axes[1, 0].set_title("Temporal corr / Wasserstein")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(history.get("val_drift", []), label="val_drift")
    axes[1, 1].set_title("Rollout drift")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_attention_map(attn: np.ndarray, out_png: str | Path) -> None:
    """
    attn expected shape (L,H,T,T) or compatible.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if attn.ndim == 5:
        attn = attn[:, 0, ...]
    if attn.ndim != 4:
        return

    avg = attn.mean(axis=(0, 1))

    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(avg, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("Average attention map")
    ax.set_xlabel("key timestep")
    ax.set_ylabel("query timestep")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_hist_triplet(truth: np.ndarray, pred: np.ndarray, out_png: str | Path, n_channels: int = 6) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n_channels = min(n_channels, truth.shape[0])
    fig, axes = plt.subplots(n_channels, 3, figsize=(9, 2.2 * n_channels))

    for i in range(n_channels):
        axes[i, 0].imshow(truth[i], origin="lower", aspect="auto")
        axes[i, 0].set_title(f"Truth ch{i}")
        axes[i, 1].imshow(pred[i], origin="lower", aspect="auto")
        axes[i, 1].set_title(f"Pred ch{i}")
        axes[i, 2].imshow(np.abs(truth[i] - pred[i]), origin="lower", aspect="auto")
        axes[i, 2].set_title("|diff|")
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
