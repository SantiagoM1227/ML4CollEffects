from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    dataset_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # density field
    grid_n: int = 64
    percentile_lo: float = 0.5
    percentile_hi: float = 99.5
    sigma_steps: float = 2.0  # sigma = sigma_steps * grid_step

    # stage A: token AE
    token_dim: int = 128
    ae_epochs: int = 50
    ae_batch_size: int = 16
    ae_lr: float = 1e-3
    ae_train_samples: int = 400
    ae_val_samples: int = 100
    ae_ckpt_path: str = "./models/token_ae.pt"

    # stage B: latent operator (FNO)
    op_epochs: int = 50
    op_batch_size: int = 8
    op_lr: float = 1e-3
    op_weight_decay: float = 1e-6
    op_width: int = 128
    op_modes: int = 16
    op_depth: int = 4
    op_hidden_proj: int = 128
    lambda_field: float = 1.0
    op_ckpt_path: str = "./models/latent_fno.pt"

    seed: int = 42


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
    w = torch.tensor([1.0, 1.0, 3.0], device=x.device).view(1,3,1,1)  # emphasize zd
    return torch.mean(w * (x_hat - x) ** 2)



   


import numpy as np

def frac_outside_range(clouds: np.ndarray, lo: float, hi: float, dim: int) -> float:
    v = clouds[:, :, dim].ravel()
    return float(np.mean((v < lo) | (v > hi)))

def check_fieldbuilder_coverage(fb: FieldBuilder, X: np.ndarray, Y: np.ndarray, idx: np.ndarray):
    # fb grids are torch tensors
    xlo, xhi   = float(fb.x_grid[0].cpu()),  float(fb.x_grid[-1].cpu())
    ylo, yhi   = float(fb.y_grid[0].cpu()),  float(fb.y_grid[-1].cpu())
    zlo, zhi   = float(fb.z_grid[0].cpu()),  float(fb.z_grid[-1].cpu())
    pxlo, pxhi = float(fb.px_grid[0].cpu()), float(fb.px_grid[-1].cpu())
    pylo, pyhi = float(fb.py_grid[0].cpu()), float(fb.py_grid[-1].cpu())
    dlo, dhi   = float(fb.d_grid[0].cpu()),  float(fb.d_grid[-1].cpu())

    subX = X[idx]
    subY = Y[idx]

    print("Outside fractions on this subset:")
    print("  X: x ",  frac_outside_range(subX, xlo,  xhi,  0), " px", frac_outside_range(subX, pxlo, pxhi, 3))
    print("  X: y ",  frac_outside_range(subX, ylo,  yhi,  1), " py", frac_outside_range(subX, pylo, pyhi, 4))
    print("  X: z ",  frac_outside_range(subX, zlo,  zhi,  2), " d ", frac_outside_range(subX, dlo,  dhi,  5))
    print("  Y: x ",  frac_outside_range(subY, xlo,  xhi,  0), " px", frac_outside_range(subY, pxlo, pxhi, 3))
    print("  Y: y ",  frac_outside_range(subY, ylo,  yhi,  1), " py", frac_outside_range(subY, pylo, pyhi, 4))
    print("  Y: z ",  frac_outside_range(subY, zlo,  zhi,  2), " d ", frac_outside_range(subY, dlo,  dhi,  5))
# ----------------------------
# Datasets
# ----------------------------
class FieldDatasetXY(Dataset):
    """
    For AE training: returns density fields built from either X_cloud or Y_cloud (random pick).
    """
    def __init__(self, X, Y, idx, field_builder: FieldBuilder):
        self.X = X
        self.Y = Y
        self.idx = np.array(idx, dtype=np.int64)
        self.fb = field_builder

    def __len__(self): return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        cloud = self.X[i] if (np.random.rand() < 0.5) else self.Y[i]
        cloud = torch.from_numpy(cloud).float().unsqueeze(0)  # [1,Np,6]
        f = self.fb.cloud_to_fields(cloud)[0]                 # [3,H,W]
        return f


class OperatorFieldDataset(Dataset):
    """
    For operator training: returns (Fin, Fout, mu)
    """
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
        mu_raw = torch.from_numpy(self.MU[i]).float()

        mu0 = torch.log10(mu_raw[0].clamp_min(1e-30))  # Q
        mu1 = mu_raw[1] * 1e3                          # a in mm (optional but helps)
        mu2 = torch.log10(mu_raw[2].clamp_min(1e-30))  # Z_scale
        mu = torch.stack([mu0, mu1, mu2], dim=0)

        Fin = self.fb.cloud_to_fields(x)[0]
        Fout = self.fb.cloud_to_fields(y)[0]
        return Fin, Fout, mu


# ----------------------------
# Models: Token AE
# ----------------------------
class FieldTokenAE(nn.Module):
    """
    Encode [B,3,64,64] -> tokens Z [B,T=64,C]
    Decode tokens back -> [B,3,64,64]
    Bottleneck is 8x8 => 64 tokens.
    """
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
            nn.Upsample(scale_factor=2, mode="nearest"),                # 16x16
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                # 32x32
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                # 64x64
            nn.Conv2d(32, in_ch, 1),
        )

    def encode_tokens(self, f: torch.Tensor) -> torch.Tensor:
        h = self.enc(f)  # [B,C,8,8]
        B, C, H, W = h.shape
        Z = h.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,64,C]
        return Z

    def decode_tokens(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, C = Z.shape
        side = int(math.isqrt(T))
        if side * side != T:
            raise ValueError(f"T={T} is not a perfect square; cannot reshape to 2D grid.")
        h = Z.reshape(B, side, side, C).permute(0, 3, 1, 2)  # [B,C,8,8]
        out = self.dec(h)                                    # [B,3,64,64]
        out = F.softplus(out)
        out = out / (out.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return out

    def forward(self, f: torch.Tensor):
        Z = self.encode_tokens(f)
        f_hat = self.decode_tokens(Z)
        return f_hat, Z


# ----------------------------
# Models: FNO blocks + LatentFNO1d
# ----------------------------
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
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.compl_weight()[:, :, :n_modes],
        )
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
    """
    (Z_in, mu) -> Z_out
    Z_in: [B, T, token_dim]
    mu:   [B, 3]
    """
    def __init__(self, token_dim=64, mu_dim=3, width=128, modes=16, depth=4, hidden_proj=128):
        super().__init__()
        self.token_dim = token_dim
        self.mu_dim = mu_dim

        in_channels = token_dim + mu_dim + 2  # + token coordinate
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, hidden_proj, kernel_size=1)
        self.proj2 = nn.Conv1d(hidden_proj, token_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, Z_in: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        B, T, C = Z_in.shape
        if C != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {C}")
        if mu.shape != (B, self.mu_dim):
            raise ValueError(f"Expected mu shape {(B, self.mu_dim)}, got {tuple(mu.shape)}")

        tgrid = torch.linspace(0, 1, T, device=Z_in.device).view(1, T, 1).expand(B, T, 1)
        mu_rep = mu.view(B, 1, self.mu_dim).expand(B, T, self.mu_dim)

        side = int(math.isqrt(T))
        if side * side != T:
            raise ValueError(f"T={T} is not a perfect square; cannot build 2D coords.")

        # 2D coords in [0,1]
        uu = torch.linspace(0, 1, side, device=Z_in.device)
        vv = torch.linspace(0, 1, side, device=Z_in.device)
        U, V = torch.meshgrid(uu, vv, indexing="ij")          # [side,side]
        uv = torch.stack([U, V], dim=-1).reshape(1, T, 2)     # [1,T,2]
        uv = uv.expand(B, T, 2)


        x = torch.cat([Z_in, mu_rep, uv], dim=-1)      # [B,T,C+mu+2]
        x = self.lift(x)                               # [B,T,width]
        x = x.permute(0, 2, 1)                         # [B,width,T]

        for blk in self.blocks:
            x = blk(x)

        x = self.act(self.proj1(x))
        x = self.proj2(x)                              # [B,token_dim,T]

        Z_delta = x.permute(0, 2, 1)                   # [B,T,token_dim]

        return Z_in + Z_delta
     




# ----------------------------
# Train/Eval loops
# ----------------------------
@torch.no_grad()
def eval_ae(ae: FieldTokenAE, loader: DataLoader, device: str) -> float:
    ae.eval()
    s = 0.0
    n = 0
    for f in loader:
        f = f.to(device)
        f_hat, _ = ae(f)
        loss = field_loss(f_hat, f)
        b = f.size(0)
        s += loss.item() * b
        n += b
    return s / max(1, n)


def train_token_ae(cfg: Config, X, Y, train_idx, val_idx, fb: FieldBuilder) -> FieldTokenAE:
    device = cfg.device
    ae = FieldTokenAE(in_ch=3, token_dim=cfg.token_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    train_ds = FieldDatasetXY(X, Y, train_idx[: cfg.ae_train_samples], fb)
    val_ds = FieldDatasetXY(X, Y, val_idx[: cfg.ae_val_samples], fb)

    train_loader = DataLoader(train_ds, batch_size=cfg.ae_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.ae_batch_size, shuffle=False, num_workers=0)

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.ae_epochs + 1):
        ae.train()
        s = 0.0
        n = 0
        for f in train_loader:
            f = f.to(device)
            f_hat, _ = ae(f)
            loss = field_loss(f_hat, f)

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = f.size(0)
            s += loss.item() * b
            n += b

        tr = s / max(1, n)
        va = eval_ae(ae, val_loader, device)

        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(f"[AE] epoch={epoch:03d} train={tr:.3e} val={va:.3e}")

    if best_state is not None:
        ae.load_state_dict(best_state)

    torch.save({"state_dict": ae.state_dict(), "config": cfg.__dict__}, cfg.ae_ckpt_path)
    print(f"[AE] saved checkpoint: {cfg.ae_ckpt_path}")

    return ae


@torch.no_grad()
def eval_operator(ae: FieldTokenAE, op: LatentFNO1d, loader: DataLoader, device: str, lambda_field: float) -> Tuple[float, float]:
    ae.eval()
    op.eval()
    s_lat = 0.0
    s_field = 0.0
    n = 0

    
    for Fin, Fout, mu in loader:
        Fin = Fin.to(device)
        Fout = Fout.to(device)
        mu = mu.to(device)


        Zin = ae.encode_tokens(Fin)
        Zout = ae.encode_tokens(Fout)
        Zpred = op(Zin, mu)

        with torch.no_grad():
            
            lat_id = torch.mean((Zin - Zout)**2).item()
            print("latent identity mse:", lat_id)


        lat = torch.mean((Zpred - Zout) ** 2)
        Fhat = ae.decode_tokens(Zpred)
        fld = field_loss(Fhat, Fout)



        b = Fin.size(0)
        s_lat += lat.item() * b
        s_field += fld.item() * b
        n += b

    return s_lat / max(1, n), s_field / max(1, n)


def train_operator(cfg: Config, ae: FieldTokenAE, X, Y, MU, train_idx, val_idx, fb: FieldBuilder) -> LatentFNO1d:
    device = cfg.device

    # freeze AE
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()

    op = LatentFNO1d(
        token_dim=cfg.token_dim,
        mu_dim=3,
        width=cfg.op_width,
        modes=cfg.op_modes,
        depth=cfg.op_depth,
        hidden_proj=cfg.op_hidden_proj,
    ).to(device)

    opt = torch.optim.Adam(op.parameters(), lr=cfg.op_lr, weight_decay=cfg.op_weight_decay)

    train_ds = OperatorFieldDataset(X, Y, MU, train_idx[: cfg.ae_train_samples], fb)
    val_ds = OperatorFieldDataset(X, Y, MU, val_idx[: cfg.ae_val_samples], fb)

    train_loader = DataLoader(train_ds, batch_size=cfg.op_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.op_batch_size, shuffle=False, num_workers=0)

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.op_epochs + 1):
        op.train()
        s = 0.0
        n = 0
        for Fin, Fout, mu in train_loader:
            Fin = Fin.to(device)
            Fout = Fout.to(device)
            mu = mu.to(device)

            with torch.no_grad():
                Zin = ae.encode_tokens(Fin)
                Zout = ae.encode_tokens(Fout)

            Zpred = op(Zin, mu)
            loss_lat = torch.mean((Zpred - Zout) ** 2)

            # decoded field regularization (recommended)
            Fhat = ae.decode_tokens(Zpred)
            loss_field = field_loss(Fhat, Fout)

            # moment loss on zd plane
            zd_hat = Fhat[:, 2]  # [B,H,W]
            zd_true = Fout[:, 2]

            z = fb.z_grid  # torch tensors on device
            d = fb.d_grid
            Z, D = torch.meshgrid(z, d, indexing="ij")  # [H,W]
            def plane_mean_sigma(f2d):
                du = (z[-1]-z[0]) / max(1, z.numel()-1)
                dv = (d[-1]-d[0]) / max(1, d.numel()-1)
                mass = (f2d.sum(dim=(-2,-1)) * du * dv).clamp_min(1e-12)  # [B]
                p = f2d / mass[:, None, None]
                mu_z = (p * Z).sum(dim=(-2,-1)) * du * dv
                mu_d = (p * D).sum(dim=(-2,-1)) * du * dv
                var_z = (p * (Z - mu_z[:,None,None])**2).sum(dim=(-2,-1)) * du * dv
                var_d = (p * (D - mu_d[:,None,None])**2).sum(dim=(-2,-1)) * du * dv
                return mu_z, mu_d, torch.sqrt(var_z.clamp_min(0.0)), torch.sqrt(var_d.clamp_min(0.0))


            mz_h, md_h, sz_h, sd_h = plane_mean_sigma(zd_hat)
            mz_t, md_t, sz_t, sd_t = plane_mean_sigma(zd_true)

            loss_zd_mom = ((mz_h - mz_t)**2 + (md_h - md_t)**2 + (sz_h - sz_t)**2 + (sd_h - sd_t)**2).mean()

            

           
            

            loss = loss = loss_lat + cfg.lambda_field * loss_field + 0.5 * loss_zd_mom

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = Fin.size(0)
            s += loss.item() * b
            n += b

        tr = s / max(1, n)
        v_lat, v_field = eval_operator(ae, op, val_loader, device, cfg.lambda_field)

        # pick best by field loss (you can change criterion)
        if v_field < best:
            best = v_field
            best_state = {k: v.detach().cpu().clone() for k, v in op.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[OP] epoch={epoch:03d} train={tr:.3e} | "
                f"val_lat={v_lat:.3e} val_field={v_field:.3e}"
            )

    if best_state is not None:
        op.load_state_dict(best_state)

    torch.save({"state_dict": op.state_dict(), "config": cfg.__dict__}, cfg.op_ckpt_path)
    print(f"[OP] saved checkpoint: {cfg.op_ckpt_path}")

    return op


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config(
        dataset_path=os.environ.get("DATASET_PATH", ""),
    )
    if not cfg.dataset_path:
        raise ValueError("Set Config.dataset_path or environment variable DATASET_PATH to the .npz dataset path.")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    raw = np.load(cfg.dataset_path, allow_pickle=True)
    X = raw["X_cloud"]
    Y = raw["Y_cloud"]
    MU = raw["MU"]
    train_idx = raw["train"]
    val_idx = raw["val"]

    print("[INFO] dataset:", cfg.dataset_path)
    print("[INFO] X:", X.shape, X.dtype, "MU:", MU.shape)
    print("[INFO] device:", cfg.device)

    # Build grids/sigmas from a subset of X train
    sub = np.concatenate([X[train_idx[:50]], Y[train_idx[:50]]], axis=0)
    fb = FieldBuilder(X_cloud_train_subset=sub, grid_n=cfg.grid_n,
                  lo=cfg.percentile_lo, hi=cfg.percentile_hi, sigma_steps=cfg.sigma_steps).to(cfg.device)
    #fb = FieldBuilder(X_cloud_train_subset=X[train_idx[:50]], grid_n=cfg.grid_n,
    #                 lo=cfg.percentile_lo, hi=cfg.percentile_hi, sigma_steps=cfg.sigma_steps).to(cfg.device)

    check_fieldbuilder_coverage(fb, X, Y, train_idx[:200])
    print("[INFO] field grids built. grid_n=", cfg.grid_n)

    # Stage A: train token AE
    ae = train_token_ae(cfg, X, Y, train_idx, val_idx, fb)

    # Stage B: train latent operator
    _ = train_operator(cfg, ae, X, Y, MU, train_idx, val_idx, fb)


if __name__ == "__main__":
    main()