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


DIM_NAMES = ["x", "y", "zeta", "px", "py", "delta"]
DIM_IDX = {"x": 0, "y": 1, "zeta": 2, "px": 3, "py": 4, "delta": 5}


@dataclass
class Config:
    dataset_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # CP factorization sketch
    n_bins: int = 64
    rank: int = 16
    pct_lo: float = 0.1
    pct_hi: float = 99.9
    sigma_steps: float = 1.0

    # conditioning
    mu_dim: int = 3

    # FNO1d (NeuralOperator)
    fno_width: int = 64
    fno_modes: int = 16
    fno_layers: int = 4

    # training
    epochs: int = 100
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-6

    # output
    out_dir: str = "./experiments/runs/demo-run-002/output"
    ckpt_path: str = "./experiments/runs/demo-run-002/models/cp6d_neuralop_fno.pt"
    meta_path: str = "./experiments/runs/demo-run-002/models/cp6d_neuralop_meta.json"


def percentile_range(a: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


class CP6DBuilder:
    """
    Same idea as before: build per-dim grids, then represent a cloud by
    rank anchors -> 1D soft hist factors per dim and per rank component.

    Returns:
      factors: [6, R, B] (each [R,B] normalized along B)
      amps:    [R] (normalized)
    """
    def __init__(self, clouds_subset: np.ndarray, n_bins: int, pct_lo: float, pct_hi: float, sigma_steps: float, rank: int, seed: int):
        self.n_bins = int(n_bins)
        self.rank = int(rank)
        self.sigma_steps = float(sigma_steps)
        self.rng = np.random.default_rng(seed)

        self.grids: Dict[str, np.ndarray] = {}
        self.deltas: Dict[str, float] = {}
        for name, j in DIM_IDX.items():
            lo, hi = percentile_range(clouds_subset[:, :, j].ravel(), pct_lo, pct_hi)
            g = np.linspace(lo, hi, self.n_bins).astype(np.float64)
            self.grids[name] = g
            self.deltas[name] = float(g[1] - g[0])

        self.tgrids = None

    def to(self, device: str):
        self.tgrids = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in self.grids.items()}
        return self

    @torch.no_grad()
    def cloud_to_cp(self, cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = cloud.device
        dtype = cloud.dtype
        Np = cloud.shape[0]
        R = self.rank
        B = self.n_bins

        idx = torch.randperm(Np, device=device)[:R] if Np >= R else torch.randint(0, Np, (R,), device=device)
        anchors = cloud[idx]  # [R,6]

        amps = torch.ones(R, device=device, dtype=dtype) / float(R)

        factors = torch.zeros(6, R, B, device=device, dtype=dtype)
        for d, name in enumerate(DIM_NAMES):
            grid = self.tgrids[name].to(device=device, dtype=dtype)  # [B]
            sigma = torch.as_tensor(self.sigma_steps * self.deltas[name], device=device, dtype=dtype)

            c = anchors[:, d].view(R, 1)
            g = grid.view(1, B)
            w = torch.exp(-0.5 * ((g - c) / (sigma + 1e-12)) ** 2)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
            factors[d] = w

        return factors, amps


class CP6DDataset(Dataset):
    def __init__(self, X, Y, MU, idx: np.ndarray):
        self.X = X
        self.Y = Y
        self.MU = MU
        self.idx = np.array(idx, dtype=np.int64)

    def __len__(self): return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        x = torch.from_numpy(self.X[i]).float()  # [Np,6]
        y = torch.from_numpy(self.Y[i]).float()
        mu_raw = torch.from_numpy(self.MU[i]).float()

        # same mu preprocessing you used before
        mu0 = torch.log10(mu_raw[0].clamp_min(1e-30))  # Q
        mu1 = mu_raw[1] * 1e3                          # a in mm
        mu2 = torch.log10(mu_raw[2].clamp_min(1e-30))  # Z_scale
        mu = torch.stack([mu0, mu1, mu2], dim=0)

        return x, y, mu


class CP6DNeuralOp(nn.Module):
    """
    Uses NeuralOperator's true FNO1d.

    For each dimension d:
      input:  [B, R + mu_dim, n_bins]
      output: [B, R, n_bins]
    Shared FNO across dims (same weights) for simplicity/regularization.
    """
    def __init__(self, rank: int, n_bins: int, mu_dim: int, width: int, modes: int, layers: int):
        super().__init__()
        self.rank = rank
        self.n_bins = n_bins
        self.mu_dim = mu_dim

        in_ch = rank + mu_dim
        out_ch = rank

        self.fno = FNO(
            n_modes=(modes,),
            hidden_channels=width,
            in_channels=in_ch,
            out_channels=out_ch,
            n_layers=layers,
            # factorization options exist in neuralop, but keep default first
        )

    def forward(self, factors_in: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        factors_in: [B,6,R,Bins]
        mu:         [B,mu_dim]
        return:     [B,6,R,Bins]
        """
        B, D, R, Bins = factors_in.shape
        assert D == 6 and R == self.rank and Bins == self.n_bins

        # expand mu into channels constant over space
        mu_ch = mu.unsqueeze(-1).expand(B, self.mu_dim, Bins)  # [B,mu_dim,Bins]

        outs = []
        for d in range(6):
            x = factors_in[:, d]                               # [B,R,Bins]
            x_in = torch.cat([x, mu_ch], dim=1)                # [B,R+mu_dim,Bins]
            y = self.fno(x_in)                                 # [B,R,Bins]
            y = F.softplus(y)
            y = y / (y.sum(dim=-1, keepdim=True) + 1e-12)      # normalize per rank
            outs.append(y)

        return torch.stack(outs, dim=1)  # [B,6,R,Bins]


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
    builder = CP6DBuilder(sub, n_bins=cfg.n_bins, pct_lo=cfg.pct_lo, pct_hi=cfg.pct_hi,
                         sigma_steps=cfg.sigma_steps, rank=cfg.rank, seed=cfg.seed).to(cfg.device)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.ckpt_path).parent).mkdir(parents=True, exist_ok=True)

    meta = {"config": asdict(cfg), "grids": {k: v.tolist() for k, v in builder.grids.items()}, "deltas": builder.deltas}
    Path(cfg.meta_path).write_text(json.dumps(meta, indent=2))

    ds_tr = CP6DDataset(X, Y, MU, train_idx)
    ds_va = CP6DDataset(X, Y, MU, val_idx)
    tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = CP6DNeuralOp(rank=cfg.rank, n_bins=cfg.n_bins, mu_dim=cfg.mu_dim,
                         width=cfg.fno_width, modes=cfg.fno_modes, layers=cfg.fno_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def build_cp_batch(cloud_batch: torch.Tensor) -> torch.Tensor:
        # cloud_batch: [B,Np,6] on device -> factors: [B,6,R,Bins]
        fs = []
        for b in range(cloud_batch.shape[0]):
            f, _a = builder.cloud_to_cp(cloud_batch[b])  # [6,R,B]
            fs.append(f)
        return torch.stack(fs, dim=0)

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        s = 0.0; n = 0

        for x, y, mu in tr:
            x = x.to(cfg.device)  # [B,Np,6]
            y = y.to(cfg.device)
            mu = mu.to(cfg.device)

            fx = build_cp_batch(x)
            fy = build_cp_batch(y)

            fhat = model(fx, mu)

            loss = torch.mean((fhat - fy) ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = x.size(0)
            s += float(loss.item()) * bsz
            n += bsz

        tr_loss = s / max(1, n)

        # val
        model.eval()
        sv = 0.0; nv = 0
        with torch.no_grad():
            for x, y, mu in va:
                x = x.to(cfg.device); y = y.to(cfg.device); mu = mu.to(cfg.device)
                fx = build_cp_batch(x)
                fy = build_cp_batch(y)
                fhat = model(fx, mu)
                vloss = torch.mean((fhat - fy) ** 2)
                bsz = x.size(0)
                sv += float(vloss.item()) * bsz
                nv += bsz
        va_loss = sv / max(1, nv)

        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"[E{epoch:03d}] train={tr_loss:.4e} val={va_loss:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, cfg.ckpt_path)
    print("[OK] saved ckpt:", cfg.ckpt_path)
    print("[OK] saved meta:", cfg.meta_path)


if __name__ == "__main__":
    main()