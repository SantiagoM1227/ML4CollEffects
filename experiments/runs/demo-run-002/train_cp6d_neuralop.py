from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



# --- NeuralOperator FNO import (version-tolerant) ---
try:
    from neuralop.models import FNO
except Exception:
    from neuralop.models.fno import FNO



@dataclass
class Config:
    dataset_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # density field
    grid_n: int = 64
    percentile_lo: float = 0.5
    percentile_hi: float = 99.5
    sigma_steps: float = 2.0

    # conditioning
    mu_dim: int = 3

    # FNO2d (NeuralOperator) 
    fno_width: int = 64
    fno_modes: int = 12
    fno_layers: int = 4

    # training
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-6

    # low-rank regularization (Fourier)
    lambda_hf: float = 1e-4

    # output
    out_dir: str = "./output"
    ckpt_path: str = "./models/fno2d_planes.pt"
    meta_path: str = "./models/fno2d_planes_meta.json"

 # ----------------------------
# Utilities: density fields from clouds
# ----------------------------
def percentile_range(a: np.ndarray, lo=0.5, hi=99.5) -> Tuple[float, float]:
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def soft_kde2d(u, v, u_grid, v_grid, su, sv):
    """
    u,v: [B,N]
    u_grid: [Hu], v_grid: [Wv]
    returns rho: [B, Hu, Wv] (not normalized)
    """
    du = (u[:, :, None, None] - u_grid[None, None, :, None]) / su
    dv = (v[:, :, None, None] - v_grid[None, None, None, :]) / sv
    w = torch.exp(-0.5 * (du**2 + dv**2))   # [B,N,Hu,Wv]
    return w.mean(dim=1)                    # [B,Hu,Wv]


def normalize_density(rho, u_grid, v_grid):
    du = (u_grid[-1] - u_grid[0]) / max(1, u_grid.numel() - 1)
    dv = (v_grid[-1] - v_grid[0]) / max(1, v_grid.numel() - 1)
    mass = rho.sum(dim=(-2, -1), keepdim=True) * du * dv
    return rho / (mass + 1e-12)


class FieldBuilder:
    """
    Holds grids + sigmas, builds 3 density planes from a 6D cloud batch.
    """
    def __init__(self, X_cloud_train_subset: np.ndarray, grid_n: int, lo: float, hi: float, sigma_steps: float):
        # subset: [Ns, Np, 6]
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

        # sigma ~ few grid steps
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
        """
        cloud_batch: [B,Np,6]
        returns: [B,3,grid_n,grid_n] for (x,px), (y,py), (zeta,delta)
        """
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


def field_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # both [B,3,H,W], enforce mass normalization (sum only; consistent with your notebook)
    x_hat = x_hat / (x_hat.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    x     = x     / (x.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    #return torch.mean((x_hat - x) ** 2)
    w = torch.tensor([1.0, 1.0, 1.0], device=x.device).view(1,3,1,1)  # emphasize zd
    return torch.mean(w * (x_hat - x) ** 2)

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
        x = torch.from_numpy(self.X[i]).float().unsqueeze(0)  # [1,Np,6]
        y = torch.from_numpy(self.Y[i]).float().unsqueeze(0)
        mu_raw = torch.from_numpy(self.MU[i]).float()

        mu0 = torch.log10(mu_raw[0].clamp_min(1e-30))
        mu1 = mu_raw[1] * 1e3
        mu2 = torch.log10(mu_raw[2].clamp_min(1e-30))
        mu = torch.stack([mu0, mu1, mu2], dim=0)

        Fin = self.fb.cloud_to_fields(x)[0]   # [3,H,W]
        Fout = self.fb.cloud_to_fields(y)[0]
        return Fin, Fout, mu



class PlaneFNO2d(nn.Module):
    def __init__(self, mu_dim: int, width: int, modes: int, layers: int):
        super().__init__()
        self.mu_dim = mu_dim
        self.fno = FNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=3 + mu_dim,
            out_channels=3,
            n_layers=layers,
        )

    def forward(self, Fin: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        # Fin: [B,3,H,W], mu: [B,3]
        B, _, H, W = Fin.shape
        mu_img = mu.view(B, self.mu_dim, 1, 1).expand(B, self.mu_dim, H, W)
        x = torch.cat([Fin, mu_img], dim=1)  # [B,3+mu_dim,H,W]
        y = self.fno(x)                      # [B,3,H,W]
        y = F.softplus(y)
        y = y / (y.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return y

def high_freq_penalty(field: torch.Tensor, keep: int) -> torch.Tensor:
    # field: [B,3,H,W]
    ft = torch.fft.rfft2(field, norm="ortho")  # [B,3,H,W//2+1]
    mask = torch.ones_like(ft.real)
    mask[:, :, :keep, :keep] = 0.0
    power = ft.real**2 + ft.imag**2
    return (mask * power).mean()

def main():
    cfg = Config(dataset_path=os.environ.get("DATASET_PATH", ""))
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    raw = np.load(cfg.dataset_path, allow_pickle=True)
    X = raw["X_cloud"]; Y = raw["Y_cloud"]; MU = raw["MU"]
    train_idx = raw["train"]; val_idx = raw["val"]

    # grids from subset of BOTH X and Y (coverage)
    sub = np.concatenate([X[train_idx[:200]], Y[train_idx[:200]]], axis=0)
    fb = FieldBuilder(
        X_cloud_train_subset=sub,
        grid_n=cfg.grid_n,
        lo=cfg.percentile_lo,
        hi=cfg.percentile_hi,
        sigma_steps=cfg.sigma_steps,
    ).to(cfg.device)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.ckpt_path).parent).mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "grids": {
            "x_grid": fb.x_grid.detach().cpu().numpy().tolist(),
            "y_grid": fb.y_grid.detach().cpu().numpy().tolist(),
            "z_grid": fb.z_grid.detach().cpu().numpy().tolist(),
            "px_grid": fb.px_grid.detach().cpu().numpy().tolist(),
            "py_grid": fb.py_grid.detach().cpu().numpy().tolist(),
            "d_grid": fb.d_grid.detach().cpu().numpy().tolist(),
        },
    }
    Path(cfg.meta_path).write_text(json.dumps(meta, indent=2))

    ds_tr = OperatorFieldDataset(X, Y, MU, train_idx, fb)
    ds_va = OperatorFieldDataset(X, Y, MU, val_idx, fb)
    tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = PlaneFNO2d(
        mu_dim=cfg.mu_dim,
        width=cfg.fno_width,
        modes=cfg.fno_modes,
        layers=cfg.fno_layers,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        s = 0.0; n = 0

        for Fin, Fout, mu in tr:

            Fin = Fin.to(cfg.device)     # [B,3,H,W]
            Fout = Fout.to(cfg.device)   # [B,3,H,W]
            mu = mu.to(cfg.device)       # [B,3]

            Fhat = model(Fin, mu)

            loss_data = field_loss(Fhat, Fout)
            loss_hf = high_freq_penalty(Fhat, keep=cfg.fno_modes)
            loss = loss_data + cfg.lambda_hf * loss_hf


            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = Fin.size(0)
            s += float(loss.item()) * bsz
            n += bsz

        tr_loss = s / max(1, n)

        # val
        model.eval()
        sv = 0.0; nv = 0
        with torch.no_grad():
            for Fin, Fout, mu in va:
                Fin = Fin.to(cfg.device)
                Fout = Fout.to(cfg.device)
                mu = mu.to(cfg.device)

                Fhat = model(Fin, mu)
                vloss = field_loss(Fhat, Fout) + cfg.lambda_hf * high_freq_penalty(Fhat, keep=cfg.fno_modes)
                
                bsz = Fin.size(0)
                sv += float(vloss.item()) * bsz
                nv += bsz

        va_loss = sv / max(1, nv)

        if va_loss < best:
           best = va_loss
           best = va_loss
           sd = model.state_dict()
           sd.pop("_metadata", None)
           torch.save({"state_dict": sd, "config": asdict(cfg)}, cfg.ckpt_path)

        if epoch == 1 or epoch % 10 == 0:
            print(f"[E{epoch:03d}] train={tr_loss:.4e} val={va_loss:.4e}")
    
    if best_state is not None:
        best_state.pop("_metadata", None)
        model.load_state_dict(best_state, strict=False)

    final_state = best_state if best_state is not None else model.state_dict()
    model.load_state_dict(final_state, strict=False)
    
        

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, cfg.ckpt_path)
    Path(cfg.meta_path).write_text(json.dumps(meta, indent=2))
    print("[OK] wrote meta:", cfg.meta_path)
    print("[OK] best ckpt at:", cfg.ckpt_path)

if __name__ == "__main__":
    main()