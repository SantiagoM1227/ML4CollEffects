import os
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------
# Models (must match training)
# ----------------------------
class FieldTokenAE(nn.Module):
    def __init__(self, in_ch=3, token_dim=64):
        super().__init__()
        self.token_dim = token_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),       # 32x32
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),       # 16x16
            nn.Conv2d(64, token_dim, 3, stride=2, padding=1), nn.GELU() # 8x8
        )
        self.dec = nn.Sequential(
            nn.Conv2d(token_dim, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, in_ch, 1),
        )

    def encode_tokens(self, f: torch.Tensor) -> torch.Tensor:
        h = self.enc(f)  # [B,C,8,8]
        B, C, H, W = h.shape
        return h.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,64,C]

    def decode_tokens(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, C = Z.shape
        side = int(math.isqrt(T))
        h = Z.reshape(B, side, side, C).permute(0, 3, 1, 2)  # [B,C,8,8]
        out = self.dec(h)                                     # [B,3,64,64]
        out = F.softplus(out)
        out = out / (out.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, _, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], self.compl_weight()[:, :, :n_modes])
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class LatentFNO1d(nn.Module):
    def __init__(self, token_dim=64, mu_dim=3, width=128, modes=16, depth=4, hidden_proj=128):
        super().__init__()
        self.token_dim = token_dim
        self.mu_dim = mu_dim
        in_channels = token_dim + mu_dim + 1
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, hidden_proj, kernel_size=1)
        self.proj2 = nn.Conv1d(hidden_proj, token_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, Z_in: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        B, T, C = Z_in.shape
        tgrid = torch.linspace(0, 1, T, device=Z_in.device).view(1, T, 1).expand(B, T, 1)
        mu_rep = mu.view(B, 1, self.mu_dim).expand(B, T, self.mu_dim)
        x = torch.cat([Z_in, mu_rep, tgrid], dim=-1)  # [B,T,C+mu+1]
        x = self.lift(x)                               # [B,T,width]
        x = x.permute(0, 2, 1)                         # [B,width,T]
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)                              # [B,token_dim,T]
        return x.permute(0, 2, 1)                      # [B,T,token_dim]


# ----------------------------
# Field builder 
# ----------------------------
def percentile_range(a: np.ndarray, lo=0.5, hi=99.5) -> Tuple[float, float]:
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def soft_kde2d(u, v, u_grid, v_grid, su, sv):
    du = (u[:, :, None, None] - u_grid[None, None, :, None]) / su
    dv = (v[:, :, None, None] - v_grid[None, None, None, :]) / sv
    w = torch.exp(-0.5 * (du**2 + dv**2))
    return w.mean(dim=1)


def normalize_density(rho, u_grid, v_grid):
    du = (u_grid[-1] - u_grid[0]) / max(1, u_grid.numel() - 1)
    dv = (v_grid[-1] - v_grid[0]) / max(1, v_grid.numel() - 1)
    mass = rho.sum(dim=(-2, -1), keepdim=True) * du * dv
    return rho / (mass + 1e-12)


class FieldBuilder:
    def __init__(self, X_cloud_train_subset: np.ndarray, grid_n: int = 64, lo: float = 0.5, hi: float = 99.5, sigma_steps: float = 2.0):
        sub = X_cloud_train_subset
        xr  = percentile_range(sub[:, :, 0].ravel(), lo, hi)
        yr  = percentile_range(sub[:, :, 1].ravel(), lo, hi)
        zr  = percentile_range(sub[:, :, 2].ravel(), lo, hi)
        pxr = percentile_range(sub[:, :, 3].ravel(), lo, hi)
        pyr = percentile_range(sub[:, :, 4].ravel(), lo, hi)
        dr  = percentile_range(sub[:, :, 5].ravel(), lo, hi)

        self.x_grid  = torch.linspace(xr[0],  xr[1],  grid_n)
        self.y_grid  = torch.linspace(yr[0],  yr[1],  grid_n)
        self.z_grid  = torch.linspace(zr[0],  zr[1],  grid_n)
        self.px_grid = torch.linspace(pxr[0], pxr[1], grid_n)
        self.py_grid = torch.linspace(pyr[0], pyr[1], grid_n)
        self.d_grid  = torch.linspace(dr[0],  dr[1],  grid_n)

        sx  = float((self.x_grid[1]  - self.x_grid[0])  * sigma_steps)
        sy  = float((self.y_grid[1]  - self.y_grid[0])  * sigma_steps)
        sz  = float((self.z_grid[1]  - self.z_grid[0])  * sigma_steps)
        spx = float((self.px_grid[1] - self.px_grid[0]) * sigma_steps)
        spy = float((self.py_grid[1] - self.py_grid[0]) * sigma_steps)
        sd  = float((self.d_grid[1]  - self.d_grid[0])  * sigma_steps)

        self.sigmas = (sx, sy, sz, spx, spy, sd)

    def to(self, device: str):
        self.x_grid  = self.x_grid.to(device)
        self.y_grid  = self.y_grid.to(device)
        self.z_grid  = self.z_grid.to(device)
        self.px_grid = self.px_grid.to(device)
        self.py_grid = self.py_grid.to(device)
        self.d_grid  = self.d_grid.to(device)
        return self

    def cloud_to_fields(self, cloud_batch: torch.Tensor) -> torch.Tensor:
        sx, sy, sz, spx, spy, sd = self.sigmas
        x  = cloud_batch[:, :, 0]
        y  = cloud_batch[:, :, 1]
        zt = cloud_batch[:, :, 2]
        px = cloud_batch[:, :, 3]
        py = cloud_batch[:, :, 4]
        de = cloud_batch[:, :, 5]

        rho_x = normalize_density(soft_kde2d(x,  px, self.x_grid,  self.px_grid, sx,  spx), self.x_grid,  self.px_grid)
        rho_y = normalize_density(soft_kde2d(y,  py, self.y_grid,  self.py_grid, sy,  spy), self.y_grid,  self.py_grid)
        rho_z = normalize_density(soft_kde2d(zt, de, self.z_grid,  self.d_grid,  sz,  sd),  self.z_grid,  self.d_grid)
        return torch.stack([rho_x, rho_y, rho_z], dim=1)  # [B,3,H,W]


# ----------------------------
# Dataset for eval
# ----------------------------
class OperatorFieldDataset(Dataset):
    def __init__(self, X, Y, MU, idx, field_builder: FieldBuilder):
        self.X = X
        self.Y = Y
        self.MU = MU
        self.idx = np.array(idx, dtype=np.int64)
        self.fb = field_builder

    def __len__(self): return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        x = torch.from_numpy(self.X[i]).float().unsqueeze(0)
        y = torch.from_numpy(self.Y[i]).float().unsqueeze(0)
        mu = torch.from_numpy(self.MU[i]).float()
        Fin = self.fb.cloud_to_fields(x)[0]
        Fout = self.fb.cloud_to_fields(y)[0]
        return Fin, Fout, mu, i


# ----------------------------
# Observables on decoded fields
# ----------------------------
def plane_mass(f: np.ndarray) -> float:
    # f: [3,H,W] already normalized by construction, but measure sum anyway
    return float(f.sum())


def plane_centroid_and_rms(f2d: np.ndarray) -> Tuple[float, float, float, float]:
    # f2d: [H,W], treat indices as coordinates (observable proxy)
    H, W = f2d.shape
    uu = np.linspace(0.0, 1.0, H)
    vv = np.linspace(0.0, 1.0, W)
    U, V = np.meshgrid(uu, vv, indexing="ij")
    m = f2d.sum() + 1e-12
    mu_u = float((U * f2d).sum() / m)
    mu_v = float((V * f2d).sum() / m)
    sig_u = float(np.sqrt(((U - mu_u) ** 2 * f2d).sum() / m))
    sig_v = float(np.sqrt(((V - mu_v) ** 2 * f2d).sum() / m))
    return mu_u, mu_v, sig_u, sig_v


def shannon_entropy(f2d: np.ndarray) -> float:
    p = np.clip(f2d, 1e-12, None)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p)).sum())


def field_observables(field: np.ndarray) -> Dict[str, float]:
    # field: [3,H,W]
    out = {}
    names = ["xpx", "ypy", "zd"]
    for ch, nm in enumerate(names):
        f2d = field[ch]
        out[f"{nm}_mass_sum"] = float(f2d.sum())
        out[f"{nm}_entropy"] = shannon_entropy(f2d)
        cu, cv, su, sv = plane_centroid_and_rms(f2d)
        out[f"{nm}_centroid_u"] = cu
        out[f"{nm}_centroid_v"] = cv
        out[f"{nm}_sigma_u"] = su
        out[f"{nm}_sigma_v"] = sv
    return out
def plane_moments_from_density(
    f2d: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute physical moments from a 2D density on a regular grid.
    Returns mean_u, mean_v, sigma_u, sigma_v, cov_uv, corr_uv.
    """
    f = np.clip(f2d.astype(np.float64), 0.0, None)

    du = (u_grid[-1] - u_grid[0]) / max(1, len(u_grid) - 1)
    dv = (v_grid[-1] - v_grid[0]) / max(1, len(v_grid) - 1)

    mass = f.sum() * du * dv + eps
    p = f / mass

    U, V = np.meshgrid(u_grid, v_grid, indexing="ij")  # [H,W]

    mu_u = float((U * p).sum() * du * dv)
    mu_v = float((V * p).sum() * du * dv)

    var_u = float(((U - mu_u) ** 2 * p).sum() * du * dv)
    var_v = float(((V - mu_v) ** 2 * p).sum() * du * dv)
    cov_uv = float(((U - mu_u) * (V - mu_v) * p).sum() * du * dv)

    sig_u = float(np.sqrt(max(var_u, 0.0)))
    sig_v = float(np.sqrt(max(var_v, 0.0)))
    corr = float(cov_uv / (sig_u * sig_v + eps))

    return {
        "mass": float(mass),
        "mean_u": mu_u,
        "mean_v": mu_v,
        "sigma_u": sig_u,
        "sigma_v": sig_v,
        "cov_uv": cov_uv,
        "corr_uv": corr,
    }


def sixd_like_observables_from_field(field_3chw: np.ndarray, fb: FieldBuilder) -> Dict[str, float]:
    """
    Convert your 3 planes (x,px), (y,py), (zeta,delta) into 6D-like
    observables: means and sigmas for x,px,y,py,zeta,delta.
    Also returns per-plane correlation coefficients.
    """
    # planes
    xpx = field_3chw[0]
    ypy = field_3chw[1]
    zd  = field_3chw[2]

    # physical grids (torch -> numpy)
    x  = fb.x_grid.detach().cpu().numpy()
    px = fb.px_grid.detach().cpu().numpy()
    y  = fb.y_grid.detach().cpu().numpy()
    py = fb.py_grid.detach().cpu().numpy()
    z  = fb.z_grid.detach().cpu().numpy()
    d  = fb.d_grid.detach().cpu().numpy()

    mxpx = plane_moments_from_density(xpx, x, px)
    mypy = plane_moments_from_density(ypy, y, py)
    mzd  = plane_moments_from_density(zd,  z, d)

    out = {
        # centroids
        "mean_x": mxpx["mean_u"],
        "mean_px": mxpx["mean_v"],
        "mean_y": mypy["mean_u"],
        "mean_py": mypy["mean_v"],
        "mean_zeta": mzd["mean_u"],
        "mean_delta": mzd["mean_v"],

        # sigmas
        "sigma_x": mxpx["sigma_u"],
        "sigma_px": mxpx["sigma_v"],
        "sigma_y": mypy["sigma_u"],
        "sigma_py": mypy["sigma_v"],
        "sigma_zeta": mzd["sigma_u"],
        "sigma_delta": mzd["sigma_v"],

        # per-plane correlations (optional but useful)
        "corr_x_px": mxpx["corr_uv"],
        "corr_y_py": mypy["corr_uv"],
        "corr_zeta_delta": mzd["corr_uv"],
    }
    return out

# ----------------------------
# Plot helpers
# ----------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_plane_triplet(true_f: np.ndarray, pred_f: np.ndarray, out_path: str, title: str = ""):
    # true_f/pred_f: [3,H,W]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    if title:
        fig.suptitle(title)

    plane_names = ["(x,px)", "(y,py)", "(zeta,delta)"]
    for r in range(3):
        im0 = axes[r, 0].imshow(true_f[r], origin="lower", aspect="auto", cmap="viridis")
        axes[r, 0].set_title(f"TRUE {plane_names[r]}")
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046)

        im1 = axes[r, 1].imshow(pred_f[r], origin="lower", aspect="auto", cmap="viridis")
        axes[r, 1].set_title(f"PRED {plane_names[r]}")
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hist(true_vals: List[float], pred_vals: List[float], out_path: str, title: str):
    plt.figure(figsize=(7, 4))
    plt.hist(true_vals, bins=50, alpha=0.6, density=True, label="true")
    plt.hist(pred_vals, bins=50, alpha=0.6, density=True, label="pred")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter(x: List[float], y: List[float], out_path: str, title: str):
    x = np.array(x); y = np.array(y)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.6)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------
# Main eval
# ----------------------------
@dataclass
class EvalCfg:
    dataset_path: str
    token_ae_ckpt: str
    op_ckpt: str
    out_dir: str = "./output"
    split: str = "test"
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    grid_n: int = 64
    percentile_lo: float = 0.5
    percentile_hi: float = 99.5
    sigma_steps: float = 2.0

    token_dim: int = 64
    op_width: int = 128
    op_modes: int = 16
    op_depth: int = 4
    op_hidden_proj: int = 128


def main():

    sixd_keys = None
    sixd_true: Dict[str, List[float]] = {}
    sixd_pred: Dict[str, List[float]] = {}


    cfg = EvalCfg(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        token_ae_ckpt=os.environ.get("TOKEN_AE_CKPT", ""),
        op_ckpt=os.environ.get("OP_CKPT", ""),
        out_dir=os.environ.get("OUT_DIR", "./output"),
        split=os.environ.get("SPLIT", "test"),
    )
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")
    if not cfg.token_ae_ckpt:
        raise ValueError("Set TOKEN_AE_CKPT")
    if not cfg.op_ckpt:
        raise ValueError("Set OP_CKPT")

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "examples"))

    raw = np.load(cfg.dataset_path, allow_pickle=True)
    X = raw["X_cloud"]; Y = raw["Y_cloud"]; MU = raw["MU"]
    split_idx = raw[cfg.split]

    fb = FieldBuilder(X[raw["train"][:50]], grid_n=cfg.grid_n, lo=cfg.percentile_lo, hi=cfg.percentile_hi, sigma_steps=cfg.sigma_steps).to(cfg.device)

    ds = OperatorFieldDataset(X, Y, MU, split_idx, fb)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    ae = FieldTokenAE(in_ch=3, token_dim=cfg.token_dim).to(cfg.device)
    op = LatentFNO1d(token_dim=cfg.token_dim, mu_dim=3, width=cfg.op_width, modes=cfg.op_modes, depth=cfg.op_depth, hidden_proj=cfg.op_hidden_proj).to(cfg.device)

    ae_ckpt = torch.load(cfg.token_ae_ckpt, map_location=cfg.device)
    op_ckpt = torch.load(cfg.op_ckpt, map_location=cfg.device)

    ae.load_state_dict(ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt)
    op.load_state_dict(op_ckpt["state_dict"] if "state_dict" in op_ckpt else op_ckpt)

    ae.eval()
    op.eval()

    # aggregate metrics
    mse_lat_sum = 0.0
    mse_field_sum = 0.0
    n = 0

    # per-sample observables
    rows = []
    obs_keys = None
    true_store: Dict[str, List[float]] = {}
    pred_store: Dict[str, List[float]] = {}

    with torch.no_grad():
        for Fin, Fout, mu, idx in loader:
            Fin = Fin.to(cfg.device)
            Fout = Fout.to(cfg.device)
            mu = mu.to(cfg.device)

            Zin = ae.encode_tokens(Fin)
            Zout = ae.encode_tokens(Fout)
            Zpred = op(Zin, mu)

            loss_lat = torch.mean((Zpred - Zout) ** 2)
            Fhat = ae.decode_tokens(Zpred)
            loss_field = torch.mean((Fhat - Fout) ** 2)

            b = Fin.size(0)
            mse_lat_sum += float(loss_lat.item()) * b
            mse_field_sum += float(loss_field.item()) * b
            n += b

            # observables on CPU
            Fout_np = Fout.cpu().numpy()
            Fhat_np = Fhat.cpu().numpy()
            mu_np = mu.cpu().numpy()
            idx_np = idx.numpy()

            for i in range(b):
                obs_t = field_observables(Fout_np[i])
                obs_p = field_observables(Fhat_np[i])

                s6_t = sixd_like_observables_from_field(Fout_np[i], fb)
                s6_p = sixd_like_observables_from_field(Fhat_np[i], fb)

                if sixd_keys is None:
                    sixd_keys = sorted(list(s6_t.keys()))
                    for k in sixd_keys:
                        sixd_true[k] = []
                        sixd_pred[k] = []

                for k in sixd_keys:
                    sixd_true[k].append(float(s6_t[k]))
                    sixd_pred[k].append(float(s6_p[k]))

                if obs_keys is None:
                    obs_keys = sorted(list(obs_t.keys()))
                    for k in obs_keys:
                        true_store[k] = []
                        pred_store[k] = []

                for k in obs_keys:
                    true_store[k].append(obs_t[k])
                    pred_store[k].append(obs_p[k])

                row = {
                    "sample_idx": int(idx_np[i]),
                    "kf1": float(mu_np[i, 0]),
                    "kd1": float(mu_np[i, 1]),
                    "kf2": float(mu_np[i, 2]),
                }
                for k in obs_keys:
                    row[f"{k}_true"] = float(obs_t[k])
                    row[f"{k}_pred"] = float(obs_p[k])
                    row[f"{k}_abs_err"] = float(obs_p[k] - obs_t[k])
                rows.append(row)

                for k in sixd_keys:
                    row[f"{k}_true"] = float(s6_t[k])
                    row[f"{k}_pred"] = float(s6_p[k])

    metrics = {
        "split": cfg.split,
        "n_samples": int(n),
        "mse_latent": mse_lat_sum / max(1, n),
        "mse_field": mse_field_sum / max(1, n),
    }
    Path(os.path.join(cfg.out_dir, f"metrics_{cfg.split}.json")).write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote metrics:", metrics)

    # CSV
    csv_path = os.path.join(cfg.out_dir, f"observables_{cfg.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("[OK] wrote", csv_path)

    # plots
    for k in obs_keys or []:
        save_hist(true_store[k], pred_store[k], os.path.join(cfg.out_dir, f"hist_{k}.png"), k)
        save_scatter(true_store[k], pred_store[k], os.path.join(cfg.out_dir, f"scatter_{k}.png"), k)
    
    for k in sixd_keys or []:
        save_scatter(
            sixd_true[k],
            sixd_pred[k],
            os.path.join(cfg.out_dir, f"scatter_6d_{k}.png"),
            f"6D-like {k}",
        )

    # example images
    # pick a few deterministic samples: first 5 in split
    ex_dir = os.path.join(cfg.out_dir, "examples")
    ex_idx = list(range(min(5, len(ds))))
    with torch.no_grad():
        for j, k in enumerate(ex_idx):
            Fin, Fout, mu, idx = ds[k]
            Fin = Fin.unsqueeze(0).to(cfg.device)
            Fout = Fout.unsqueeze(0).to(cfg.device)
            mu = mu.unsqueeze(0).to(cfg.device)

            Zpred = op(ae.encode_tokens(Fin), mu)
            Fhat = ae.decode_tokens(Zpred)

            true_np = Fout[0].cpu().numpy()
            pred_np = Fhat[0].cpu().numpy()
            save_plane_triplet(true_np, pred_np, os.path.join(ex_dir, f"sample_{int(idx)}.png"), title=f"idx={int(idx)}")

    print("[OK] wrote example images to", ex_dir)


if __name__ == "__main__":
    main()