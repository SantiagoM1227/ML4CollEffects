from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_loss_curves(history: Dict[str, Sequence[float]], out_png: str | Path, title: str) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    for k, v in history.items():
        ax.plot(list(v), label=k)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_kl_recon_curve(history: Dict[str, Sequence[float]], out_png: str | Path) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    recon = np.asarray(history.get("val_recon", []), dtype=np.float64)
    kl = np.asarray(history.get("val_kl", []), dtype=np.float64)
    if recon.size and kl.size:
        ax.plot(recon, kl, "-o", ms=3)
    ax.set_xlabel("val_recon")
    ax.set_ylabel("val_kl")
    ax.set_title("KL vs Reconstruction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_latent_histograms(mu: np.ndarray, logvar: np.ndarray, out_png: str | Path, max_dims: int = 64) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    if mu.size == 0:
        mu = np.zeros((1, 1), dtype=np.float32)
        logvar = np.zeros((1, 1), dtype=np.float32)
    dshow = min(mu.shape[1], max_dims)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].hist(mu[:, :dshow].reshape(-1), bins=120)
    axes[0, 0].set_title(f"mu distribution (first {dshow} dims)")

    axes[0, 1].hist(logvar[:, :dshow].reshape(-1), bins=120)
    axes[0, 1].set_title(f"logvar distribution (first {dshow} dims)")

    axes[1, 0].plot(mu.mean(axis=0)[:dshow])
    axes[1, 0].set_title("mu mean per latent dim")

    axes[1, 1].plot(logvar.mean(axis=0)[:dshow])
    axes[1, 1].set_title("logvar mean per latent dim")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_recon_grid_15ch(x: np.ndarray, xhat: np.ndarray, out_png: str | Path, n_channels: int = 6, eps: float = 1e-18) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)

    n_channels = min(n_channels, x.shape[0])
    pos = np.concatenate([x[x > 0], xhat[xhat > 0]], axis=0) if (np.any(x > 0) or np.any(xhat > 0)) else np.array([])
    vmin = float(np.percentile(pos, 5)) if pos.size else 1e-12
    vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    vmin = max(vmin, 1e-12)
    vmax = max(vmax, 10.0 * vmin)

    fig, axes = plt.subplots(3, n_channels, figsize=(2.8 * n_channels, 7.0))
    for i in range(n_channels):
        axes[0, i].imshow(x[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[0, i].set_title(f"Truth ch{i}")

        axes[1, i].imshow(xhat[i] + eps, origin="lower", aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[1, i].set_title(f"Recon ch{i}")

        axes[2, i].imshow(np.abs(xhat[i] - x[i]), origin="lower", aspect="auto")
        axes[2, i].set_title("|diff|")

        for r in range(3):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def save_latent_covariance_heatmap(mu: np.ndarray, out_png: str | Path, max_dims: int = 64) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    if mu.size == 0:
        return
    dshow = min(mu.shape[1], max_dims)
    cov = np.cov(mu[:, :dshow], rowvar=False)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cov, origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title(f"Latent covariance (first {dshow} dims)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def save_latent_pca_umap(mu: np.ndarray, out_png: str | Path) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    if mu.size == 0 or mu.shape[0] < 2:
        return

    pca = _pca_2d(mu.astype(np.float64))
    emb = pca
    emb_name = "PCA"

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(mu.astype(np.float64))
        emb_name = "UMAP"
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(pca[:, 0], pca[:, 1], s=8, alpha=0.6)
    axes[0].set_title("Latent PCA")
    axes[1].scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.6)
    axes[1].set_title(f"Latent {emb_name}")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_mu_logvar_evolution(mu_means: Sequence[float], logvar_means: Sequence[float], out_png: str | Path) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(mu_means), label="mu_mean")
    ax.plot(list(logvar_means), label="logvar_mean")
    ax.set_title("Latent moments evolution")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_gradient_norms(grad_norms: Sequence[float], out_png: str | Path) -> None:
    out_png = Path(out_png)
    _ensure_parent(out_png)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(grad_norms))
    ax.set_title("Gradient norm (per optimization step)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
