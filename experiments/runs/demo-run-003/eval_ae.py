from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_fno1d_token import (
    Config,
    load_npz,
    CloudDataset,
    Latent1DTokenAE,
    chamfer_l2,
)


@dataclass
class EvalCfg:
    dataset_path: str
    ae_ckpt: str
    meta_path: str

    out_dir: str = "./experiments/runs/demo-run-003/output_eval_ae"
    split: str = "test"
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_batches: int = 50
    n_examples: int = 6


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def cloud_stats(cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cloud: [B,Np,6]
    returns mean: [B,6], std: [B,6]
    """
    mean = cloud.mean(dim=1)
    std = cloud.std(dim=1, unbiased=False)
    return mean, std


@torch.no_grad()
def cov6(cloud: torch.Tensor) -> torch.Tensor:
    """
    cloud: [B,Np,6] -> covariance per sample: [B,6,6]
    """
    B, Np, D = cloud.shape
    x = cloud - cloud.mean(dim=1, keepdim=True)
    return (x.transpose(1, 2) @ x) / max(1, Np)


def save_scatter(x, y, out_path: str, title: str, xlabel="true", ylabel="pred"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.6)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_cloud_overlay(true_cloud: np.ndarray, pred_cloud: np.ndarray, out_path: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].scatter(true_cloud[:, 2], true_cloud[:, 0], s=1, alpha=0.35, label="true")
    axes[0].scatter(pred_cloud[:, 2], pred_cloud[:, 0], s=1, alpha=0.35, label="pred")
    axes[0].set_title("zeta vs x")
    axes[0].set_xlabel("zeta")
    axes[0].set_ylabel("x")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(true_cloud[:, 0], true_cloud[:, 3], s=1, alpha=0.35, label="true")
    axes[1].scatter(pred_cloud[:, 0], pred_cloud[:, 3], s=1, alpha=0.35, label="pred")
    axes[1].set_title("x vs px")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("px")
    axes[1].grid(alpha=0.25)

    axes[2].scatter(true_cloud[:, 1], true_cloud[:, 4], s=1, alpha=0.35, label="true")
    axes[2].scatter(pred_cloud[:, 1], pred_cloud[:, 4], s=1, alpha=0.35, label="pred")
    axes[2].set_title("y vs py")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("py")
    axes[2].grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hist(values: np.ndarray, out_path: str, title: str, bins: int = 64, xlim: Tuple[float, float] | None = None):
    v = np.asarray(values, dtype=np.float64)
    plt.figure(figsize=(7, 4))
    plt.hist(v, bins=bins, density=True, alpha=0.85)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
def percentile_range_np(a: np.ndarray, lo=0.5, hi=99.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def make_grid_from_subset(clouds: np.ndarray, dim_u: int, dim_v: int, grid_n: int = 96, lo: float = 0.5, hi: float = 99.5):
    """
    clouds: [Ns,Np,6] numpy
    returns u_grid [grid_n], v_grid [grid_n]
    """
    u = clouds[:, :, dim_u].reshape(-1)
    v = clouds[:, :, dim_v].reshape(-1)
    ur = percentile_range_np(u, lo, hi)
    vr = percentile_range_np(v, lo, hi)
    u_grid = np.linspace(ur[0], ur[1], grid_n, dtype=np.float64)
    v_grid = np.linspace(vr[0], vr[1], grid_n, dtype=np.float64)
    return u_grid, v_grid


def soft_kde2d_torch(u: torch.Tensor, v: torch.Tensor, u_grid: torch.Tensor, v_grid: torch.Tensor, su: float, sv: float):
    """
    u,v: [Np]
    u_grid: [Hu], v_grid: [Wv]
    returns rho: [Hu,Wv] (not normalized)
    """
    du = (u[:, None, None] - u_grid[None, :, None]) / su   # [Np,Hu,1]
    dv = (v[:, None, None] - v_grid[None, None, :]) / sv   # [Np,1,Wv]
    w = torch.exp(-0.5 * (du**2 + dv**2))                  # [Np,Hu,Wv]
    return w.mean(dim=0)                                   # [Hu,Wv]


def normalize_density_2d(rho: torch.Tensor, u_grid: torch.Tensor, v_grid: torch.Tensor):
    du = (u_grid[-1] - u_grid[0]) / max(1, u_grid.numel() - 1)
    dv = (v_grid[-1] - v_grid[0]) / max(1, v_grid.numel() - 1)
    mass = rho.sum() * du * dv
    return rho / (mass + 1e-12)


@torch.no_grad()
def cloud_to_plane_kde(
    cloud: torch.Tensor,  # [Np,6]
    dim_u: int,
    dim_v: int,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    sigma_steps: float = 1.5,
) -> torch.Tensor:
    """
    Returns normalized 2D density plane [Hu,Wv] for dims (dim_u, dim_v)
    """
    u = cloud[:, dim_u]
    v = cloud[:, dim_v]
    # sigma ~ sigma_steps * grid_step
    su = float((u_grid[1] - u_grid[0]) * sigma_steps)
    sv = float((v_grid[1] - v_grid[0]) * sigma_steps)
    rho = soft_kde2d_torch(u, v, u_grid, v_grid, su, sv)
    rho = normalize_density_2d(rho, u_grid, v_grid)
    return rho


def save_plane_triplet(true_rho: np.ndarray, pred_rho: np.ndarray, out_path: str, title: str):
    """
    Save: true, pred, diff (pred-true)
    """
    diff = pred_rho - true_rho
    vmax = float(max(np.max(np.abs(true_rho)), np.max(np.abs(pred_rho)), 1e-12))
    dmax = float(np.max(np.abs(diff)) + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(true_rho, origin="lower", aspect="auto")
    axes[0].set_title("true")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_rho, origin="lower", aspect="auto")
    axes[1].set_title("pred")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="bwr", vmin=-dmax, vmax=dmax)
    axes[2].set_title("pred - true")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def main():
    cfg = EvalCfg(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        ae_ckpt=os.environ.get("DEMO003_AE", ""),
        meta_path=os.environ.get("DEMO003_META", ""),
        out_dir=os.environ.get("OUT_DIR", "./output_eval_ae"),
        split=os.environ.get("SPLIT", "test"),
    )
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")
    if not cfg.ae_ckpt:
        raise ValueError("Set DEMO003_AE")
    if not cfg.meta_path:
        raise ValueError("Set DEMO003_META")

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "examples"))
    ensure_dir(os.path.join(cfg.out_dir, "scatter"))
    ensure_dir(os.path.join(cfg.out_dir, "latent"))

    meta = json.loads(Path(cfg.meta_path).read_text())
    tr_cfg = meta["config"]

    # dataset
    data = load_npz(cfg.dataset_path)
    ds = CloudDataset(data, cfg.split, Np=int(tr_cfg.get("Np", 4096)))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    # Build common grids for KDE plane plots from a subset of the split
    # (use true clouds so ranges are stable)
    subN = min(50, len(ds))
    sub_clouds = data["X_cloud"][data[cfg.split]][:subN]  # [subN,Np,6]

    grid_n = 96
    u_x, v_px = make_grid_from_subset(sub_clouds, 0, 3, grid_n=grid_n)
    u_y, v_py = make_grid_from_subset(sub_clouds, 1, 4, grid_n=grid_n)
    u_z, v_d  = make_grid_from_subset(sub_clouds, 2, 5, grid_n=grid_n)

    x_grid  = torch.tensor(u_x, dtype=torch.float32, device=cfg.device)
    px_grid = torch.tensor(v_px, dtype=torch.float32, device=cfg.device)
    y_grid  = torch.tensor(u_y, dtype=torch.float32, device=cfg.device)
    py_grid = torch.tensor(v_py, dtype=torch.float32, device=cfg.device)
    z_grid  = torch.tensor(u_z, dtype=torch.float32, device=cfg.device)
    d_grid  = torch.tensor(v_d, dtype=torch.float32, device=cfg.device)

    ensure_dir(os.path.join(cfg.out_dir, "planes"))

    # AE
    ae = Latent1DTokenAE(
        token_dim=int(tr_cfg["token_dim"]),
        particle_hidden=int(tr_cfg["particle_hidden"]),
        token_hidden=int(tr_cfg["token_hidden"]),
    ).to(cfg.device)

    ckpt = torch.load(cfg.ae_ckpt, map_location=cfg.device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if isinstance(sd, dict):
        sd.pop("_metadata", None)
    ae.load_state_dict(sd, strict=False)
    ae.eval()

    M = int(tr_cfg.get("M", 64))
    Np = int(tr_cfg.get("Np", 4096))
    edges = torch.linspace(0.0, 1.0, M + 1, device=cfg.device, dtype=torch.float32)

    # accumulators
    chamfers: List[float] = []
    mean_mse: List[float] = []
    std_mse: List[float] = []
    cov_mse: List[float] = []

    # latent diagnostics
    s_all: List[np.ndarray] = []
    occ_all: List[np.ndarray] = []  # occupancy histogram per batch

    # scatter: mean(x), mean(delta) true vs pred
    true_mx: List[float] = []
    pred_mx: List[float] = []
    true_md: List[float] = []
    pred_md: List[float] = []

    ex_saved = 0
    nb = 0

    for Xb, _Yb, _mu in loader:
        Xb = Xb.to(cfg.device)

        Z, pos, Xhat = ae(Xb, M=M, Np=Np)

        # recon metrics: use subsample for speed and stability
        K = min(1024, Xb.shape[1])
        idx = torch.randperm(Xb.shape[1], device=cfg.device)[:K]
        ch = chamfer_l2(Xhat[:, idx], Xb[:, idx])
        chamfers.append(float(ch.item()))

        m_t, s_t = cloud_stats(Xb)
        m_p, s_p = cloud_stats(Xhat)

        mean_mse.append(float(torch.mean((m_p - m_t) ** 2).item()))
        std_mse.append(float(torch.mean((s_p - s_t) ** 2).item()))

        C_t = cov6(Xb)
        C_p = cov6(Xhat)
        cov_mse.append(float(torch.mean((C_p - C_t) ** 2).item()))

        true_mx.extend(m_t[:, 0].detach().cpu().numpy().tolist())
        pred_mx.extend(m_p[:, 0].detach().cpu().numpy().tolist())
        true_md.extend(m_t[:, 5].detach().cpu().numpy().tolist())
        pred_md.extend(m_p[:, 5].detach().cpu().numpy().tolist())

        # latent s distribution + occupancy
        s_lat = ae.particles_to_s(Xb)  # [B,Np]
        s_all.append(s_lat.detach().cpu().numpy().reshape(-1))

        bin_id = torch.bucketize(s_lat, edges) - 1
        bin_id = bin_id.clamp(0, M - 1)
        occ = torch.bincount(bin_id.reshape(-1), minlength=M).float()
        occ = (occ / (occ.sum() + 1e-12)).detach().cpu().numpy()
        occ_all.append(occ)

        # save a few example overlays
        if ex_saved < cfg.n_examples:
            take = min(Xb.size(0), cfg.n_examples - ex_saved)
            for j in range(take):
                xt = Xb[j].detach().cpu().numpy()
                xp = Xhat[j].detach().cpu().numpy()
                save_cloud_overlay(
                    xt,
                    xp,
                    os.path.join(cfg.out_dir, "examples", f"ae_overlay_{ex_saved:03d}.png"),
                    title=f"AE recon example={ex_saved}",
                )
                # KDE plane plots: x-px, y-py, zeta-delta
                true_cloud_t = Xb[j]
                pred_cloud_t = Xhat[j]

                rho_xpx_t = cloud_to_plane_kde(true_cloud_t, 0, 3, x_grid, px_grid, sigma_steps=1.5).detach().cpu().numpy()
                rho_xpx_p = cloud_to_plane_kde(pred_cloud_t, 0, 3, x_grid, px_grid, sigma_steps=1.5).detach().cpu().numpy()
                save_plane_triplet(rho_xpx_t, rho_xpx_p,
                                os.path.join(cfg.out_dir, "planes", f"xpx_{ex_saved:03d}.png"),
                                title=f"AE plane x-px | ex={ex_saved}")

                rho_ypy_t = cloud_to_plane_kde(true_cloud_t, 1, 4, y_grid, py_grid, sigma_steps=1.5).detach().cpu().numpy()
                rho_ypy_p = cloud_to_plane_kde(pred_cloud_t, 1, 4, y_grid, py_grid, sigma_steps=1.5).detach().cpu().numpy()
                save_plane_triplet(rho_ypy_t, rho_ypy_p,
                                os.path.join(cfg.out_dir, "planes", f"ypy_{ex_saved:03d}.png"),
                                title=f"AE plane y-py | ex={ex_saved}")

                rho_zd_t = cloud_to_plane_kde(true_cloud_t, 2, 5, z_grid, d_grid, sigma_steps=1.5).detach().cpu().numpy()
                rho_zd_p = cloud_to_plane_kde(pred_cloud_t, 2, 5, z_grid, d_grid, sigma_steps=1.5).detach().cpu().numpy()
                save_plane_triplet(rho_zd_t, rho_zd_p,
                                os.path.join(cfg.out_dir, "planes", f"zd_{ex_saved:03d}.png"),
                                title=f"AE plane zeta-delta | ex={ex_saved}")
                ex_saved += 1

        nb += 1
        if nb >= cfg.max_batches:
            break

    # aggregate latent plots
    s_flat = np.concatenate(s_all, axis=0)
    occ_mean = np.mean(np.stack(occ_all, axis=0), axis=0)

    save_hist(s_flat, os.path.join(cfg.out_dir, "latent", "s_hist.png"), "latent s histogram", bins=64, xlim=(0.0, 1.0))

    # occupancy bar plot
    plt.figure(figsize=(8, 3.5))
    plt.plot(np.arange(M), occ_mean, lw=1.5)
    plt.title("Mean token-bin occupancy (probability)")
    plt.xlabel("bin index")
    plt.ylabel("prob")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "latent", "bin_occupancy.png"), dpi=150)
    plt.close()

    # scatter plots for mean(x), mean(delta)
    save_scatter(true_mx, pred_mx, os.path.join(cfg.out_dir, "scatter", "mean_x.png"), "AE: mean(x) true vs pred")
    save_scatter(true_md, pred_md, os.path.join(cfg.out_dir, "scatter", "mean_delta.png"), "AE: mean(delta) true vs pred")

    metrics = {
        "split": cfg.split,
        "n_samples": len(ds),
        "n_batches_used": nb,
        "chamfer_sub1024_mean": float(np.mean(chamfers)),
        "mean_mse_mean": float(np.mean(mean_mse)),
        "std_mse_mean": float(np.mean(std_mse)),
        "cov_mse_mean": float(np.mean(cov_mse)),
        # latent health
        "s_mean": float(np.mean(s_flat)),
        "s_std": float(np.std(s_flat)),
        "s_min": float(np.min(s_flat)),
        "s_max": float(np.max(s_flat)),
        "bin_occupancy_entropy": float(-(occ_mean * np.log(occ_mean + 1e-12)).sum()),
    }

    Path(os.path.join(cfg.out_dir, f"metrics_ae_{cfg.split}.json")).write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote AE eval outputs to:", cfg.out_dir)
    print("[OK] AE metrics:", metrics)


if __name__ == "__main__":
    main()