from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

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


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    dataset_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # coherent Np (generator default)
    Np: int = 4096

    # tokenization along learned 1D latent coordinate s
    M: int = 64  # number of tokens (bins along s)

    # AE latent/token dims
    token_dim: int = 128
    particle_hidden: int = 256
    token_hidden: int = 256

    # Stage A: token AE
    ae_epochs: int = 80
    ae_batch_size: int = 16
    ae_lr: float = 2e-3
    ae_weight_decay: float = 1e-6

    # Stage B: latent operator (NeuralOperator FNO1d on tokens)
    mu_dim: int = 3
    fno_width: int = 128
    fno_modes: int = 16
    fno_layers: int = 4

    op_epochs: int = 150
    op_batch_size: int = 8
    op_lr: float = 5e-4
    op_weight_decay: float = 1e-6

    # loss weights (operator stage)
    w_token: float = 1.0
    w_chamfer: float = 0.2     # start small (stochastic decoder)
    w_smooth: float = 0.05     # smoothness of tokens along s

    # output
    out_dir: str = "./models"
    ae_ckpt_path: str = "./models/demo003_latent1d_ae.pt"
    op_ckpt_path: str = "./models/demo003_latent1d_fno.pt"
    meta_path: str = "./models/demo003_latent1d_meta.json"


# ----------------------------
# Data
# ----------------------------
def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


class CloudDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], split: str, Np: int):
        self.X = data["X_cloud"][data[split]]  # [Ns,Np,6]
        self.Y = data["Y_cloud"][data[split]]
        self.MU = data["MU"][data[split]]
        self.Np = int(Np)

    def __len__(self) -> int:
        return self.X.shape[0]

    def _fix_np(self, cloud: np.ndarray) -> np.ndarray:
        N = cloud.shape[0]
        if N == self.Np:
            return cloud
        if N > self.Np:
            idx = np.random.choice(N, self.Np, replace=False)
            return cloud[idx]
        idx = np.random.choice(N, self.Np - N, replace=True)
        return np.concatenate([cloud, cloud[idx]], axis=0)

    def __getitem__(self, idx: int):
        x = self._fix_np(self.X[idx]).astype(np.float32)
        y = self._fix_np(self.Y[idx]).astype(np.float32)
        mu_raw = self.MU[idx].astype(np.float32)

        x = torch.from_numpy(x)  # [Np,6]
        y = torch.from_numpy(y)
        mu_raw = torch.from_numpy(mu_raw)

        # mu preprocessing (same as your other scripts)
        mu0 = torch.log10(mu_raw[0].clamp_min(1e-30))
        mu1 = mu_raw[1] * 1e3
        mu2 = torch.log10(mu_raw[2].clamp_min(1e-30))
        mu = torch.stack([mu0, mu1, mu2], dim=0)  # [3]

        return x, y, mu


# ----------------------------
# Loss: Chamfer
# ----------------------------
def chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B,Na,6], b: [B,Nb,6]
    """
    d = torch.cdist(a, b, p=2)
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def smoothness_1d_tokens(Z: torch.Tensor) -> torch.Tensor:
    """
    Z: [B,M,C]
    penalize 2nd finite difference along M
    """
    d2 = Z[:, 2:] - 2 * Z[:, 1:-1] + Z[:, :-2]
    return (d2 ** 2).mean()


# ----------------------------
# Stage A: Latent-1D Token AE
# ----------------------------
class Latent1DTokenAE(nn.Module):
    """
    Learns:
      - per-particle 1D latent coordinate s in [0,1]
      - per-token latent embedding Z_i (tokens on s-bins)
      - decoder that samples 6D points from tokens
    """
    def __init__(self, token_dim: int, particle_hidden: int, token_hidden: int):
        super().__init__()
        self.token_dim = token_dim

        # particle -> features
        self.phi = nn.Sequential(
            nn.Linear(6, particle_hidden),
            nn.GELU(),
            nn.Linear(particle_hidden, particle_hidden),
            nn.GELU(),
        )
        # particle -> scalar s
        self.s_head = nn.Sequential(
            nn.Linear(particle_hidden, particle_hidden),
            nn.GELU(),
            nn.Linear(particle_hidden, 1),
        )
        # particle -> token contribution
        self.z_head = nn.Sequential(
            nn.Linear(particle_hidden, token_dim),
        )

        # token -> gaussian params for 6D (means+logsigmas)
        self.dec = nn.Sequential(
            nn.Linear(token_dim + 1, token_hidden),  # + token position
            nn.GELU(),
            nn.Linear(token_hidden, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, 12),  # mean6 + log_sig6
        )

    def particles_to_s(self, cloud: torch.Tensor) -> torch.Tensor:
        """
        cloud: [B,Np,6] -> s: [B,Np] in [0,1]
        """
        h = self.phi(cloud)                  # [B,Np,H]
        s = self.s_head(h).squeeze(-1)       # [B,Np]
        s = torch.sigmoid(s)                 # [0,1]
        return s

    def encode_tokens(self, cloud: torch.Tensor, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cloud: [B,Np,6]
        Returns:
          Z:   [B,M,C]
          pos: [B,M] token centers in [0,1]
        """
        B, Np, _ = cloud.shape
        h = self.phi(cloud)                  # [B,Np,H]
        s = torch.sigmoid(self.s_head(h)).squeeze(-1)  # [B,Np]
        z = self.z_head(h)                   # [B,Np,C]

        # binning in s
        edges = torch.linspace(0.0, 1.0, M + 1, device=cloud.device, dtype=cloud.dtype)
        pos = 0.5 * (edges[:-1] + edges[1:])             # [M]
        pos = pos[None, :].expand(B, M)                  # [B,M]

        bin_id = torch.bucketize(s, edges) - 1
        bin_id = bin_id.clamp(0, M - 1)                  # [B,Np]

        Z = torch.zeros(B, M, self.token_dim, device=cloud.device, dtype=cloud.dtype)
        counts = torch.zeros(B, M, 1, device=cloud.device, dtype=cloud.dtype)

        for b in range(B):
            bi = bin_id[b]
            Z[b].index_add_(0, bi, z[b])
            counts[b].index_add_(0, bi, torch.ones(Np, 1, device=cloud.device, dtype=cloud.dtype))

        Z = Z / (counts + 1e-12)
        return Z, pos

    def decode_sample(self, Z: torch.Tensor, pos: torch.Tensor, Np: int) -> torch.Tensor:
        """
        Z:   [B,M,C]
        pos: [B,M]
        -> sampled cloud: [B,Np,6]
        """
        B, M, C = Z.shape
        device = Z.device
        dtype = Z.dtype

        x_in = torch.cat([Z, pos[:, :, None]], dim=-1)  # [B,M,C+1]
        out = self.dec(x_in)                            # [B,M,12]
        mean6 = out[..., :6]
        logsig = out[..., 6:].clamp(-8.0, 6.0)   # critical: prevents exp overflow
        sig6 = torch.exp(logsig)

        per = Np // M
        rem = Np - per * M

        clouds = []
        for b in range(B):
            pts = []
            for i in range(M):
                ni = per + (1 if i < rem else 0)
                eps = torch.randn(ni, 6, device=device, dtype=dtype)
                samp = mean6[b, i][None, :] + eps * sig6[b, i][None, :]  # [ni,6]
                pts.append(samp)
            clouds.append(torch.cat(pts, dim=0)[None, :, :])
        return torch.cat(clouds, dim=0)

    def forward(self, cloud: torch.Tensor, M: int, Np: int):
        Z, pos = self.encode_tokens(cloud, M)
        cloud_hat = self.decode_sample(Z, pos, Np)
        return Z, pos, cloud_hat


# ----------------------------
# Stage B: NeuralOperator FNO1d on latent tokens
# ----------------------------
class LatentTokenNeuralOp(nn.Module):
    """
    true NeuralOperator FNO1d on token sequence.
    input channels: token_dim + mu_dim + 1(pos)
    """
    def __init__(self, token_dim: int, mu_dim: int, width: int, modes: int, layers: int):
        super().__init__()
        self.token_dim = token_dim
        self.mu_dim = mu_dim

        self.fno = FNO(
            n_modes=(modes,),
            hidden_channels=width,
            in_channels=token_dim + mu_dim + 1,
            out_channels=token_dim,
            n_layers=layers,
        )

    def forward(self, Z_in: torch.Tensor, mu: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Z_in: [B,M,C], mu: [B,3], pos: [B,M]
        B, M, C = Z_in.shape
        mu_ch = mu[:, :, None].expand(B, self.mu_dim, M)   # [B,mu_dim,M]
        pos_ch = pos[:, None, :]                            # [B,1,M]
        x = Z_in.permute(0, 2, 1)                           # [B,C,M]
        x = torch.cat([x, mu_ch, pos_ch], dim=1)            # [B,C+mu+1,M]
        y = self.fno(x)                                     # [B,C,M]
        return y.permute(0, 2, 1)                           # [B,M,C]


# ----------------------------
# Training helpers
# ----------------------------
@torch.no_grad()
def eval_ae(ae: Latent1DTokenAE, loader: DataLoader, device: str, M: int, Np: int) -> float:
    ae.eval()
    s = 0.0
    n = 0
    for Xb, _Yb, _mu in loader:
        Xb = Xb.to(device)
        _Z, _pos, Xhat = ae(Xb, M=M, Np=Np)
        loss = chamfer_l2(Xhat, Xb)
        bsz = Xb.size(0)
        s += float(loss.item()) * bsz
        n += bsz
    return s / max(1, n)


def train_stage_a(cfg: Config, train_loader: DataLoader, val_loader: DataLoader) -> Latent1DTokenAE:
    ae = Latent1DTokenAE(cfg.token_dim, cfg.particle_hidden, cfg.token_hidden).to(cfg.device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay)

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.ae_epochs + 1):
        ae.train()
        s = 0.0
        n = 0

        for Xb, Yb, _mu in train_loader:
            # train AE on both X and Y for robustness
            cloud = Xb if (torch.rand(()) < 0.5) else Yb
            cloud = cloud.to(cfg.device)

            Z, pos, chat = ae(cloud, M=cfg.M, Np=cfg.Np)
            # sigma regularizer to prevent blow-up
            with torch.no_grad():
                Z_tmp, pos_tmp = Z, pos  # just naming
                
            x_in = torch.cat([Z_tmp, pos_tmp[:, :, None]], dim=-1)
            Z_tmp = Z.detach()
            pos_tmp = pos.detach()
            out = ae.dec(x_in)
            logsig = out[..., 6:].clamp(-8.0, 6.0)
            loss_sig = torch.mean(logsig ** 2)

            K = 1024
            idx = torch.randperm(cfg.Np, device=cloud.device)[:K]
            loss_ch = chamfer_l2(chat[:, idx], cloud[:, idx])
            loss_smooth = smoothness_1d_tokens(Z)

            loss = loss_ch + 0.05 * loss_smooth + 1e-3 * loss_sig

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss; skipping step")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()

            bsz = cloud.size(0)
            s += float(loss.item()) * bsz
            n += bsz

        tr = s / max(1, n)
        va = eval_ae(ae, val_loader, cfg.device, cfg.M, cfg.Np)

        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"[AE] E{epoch:03d} train={tr:.4e} val_chamfer={va:.4e}")

    if best_state is not None:
        ae.load_state_dict(best_state)

    sd = ae.state_dict()
    sd.pop("_metadata", None)
    torch.save({"state_dict": sd, "config": asdict(cfg)}, cfg.ae_ckpt_path)
    print("[AE] saved:", cfg.ae_ckpt_path)
    return ae


@torch.no_grad()
def eval_stage_b(ae: Latent1DTokenAE, op: LatentTokenNeuralOp, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    ae.eval()
    op.eval()

    s_tok = 0.0
    s_ch = 0.0
    s_smooth = 0.0
    n = 0

    for Xb, Yb, mu in loader:
        Xb = Xb.to(cfg.device)
        Yb = Yb.to(cfg.device)
        mu = mu.to(cfg.device)

        Zx, pos, _ = ae(Xb, M=cfg.M, Np=cfg.Np)
        Zy, _pos_y, _ = ae(Yb, M=cfg.M, Np=cfg.Np)

        Zhat = op(Zx, mu, pos)

        loss_tok = torch.mean((Zhat - Zy) ** 2)

        Yhat = ae.decode_sample(Zhat, pos, Np=cfg.Np)
        loss_ch = chamfer_l2(Yhat, Yb)

        loss_smooth = smoothness_1d_tokens(Zhat)

        bsz = Xb.size(0)
        s_tok += float(loss_tok.item()) * bsz
        s_ch += float(loss_ch.item()) * bsz
        s_smooth += float(loss_smooth.item()) * bsz
        n += bsz

    tok = s_tok / max(1, n)
    ch = s_ch / max(1, n)
    sm = s_smooth / max(1, n)
    score = cfg.w_token * tok + cfg.w_chamfer * ch + cfg.w_smooth * sm

    return {"token_mse": tok, "chamfer": ch, "smooth": sm, "val_score": score}


def train_stage_b(cfg: Config, ae: Latent1DTokenAE, train_loader: DataLoader, val_loader: DataLoader) -> LatentTokenNeuralOp:
    # freeze AE
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()

    op = LatentTokenNeuralOp(
        token_dim=cfg.token_dim,
        mu_dim=cfg.mu_dim,
        width=cfg.fno_width,
        modes=cfg.fno_modes,
        layers=cfg.fno_layers,
    ).to(cfg.device)

    opt = torch.optim.Adam(op.parameters(), lr=cfg.op_lr, weight_decay=cfg.op_weight_decay)

    best = float("inf")
    best_state = None

    for epoch in range(1, cfg.op_epochs + 1):
        op.train()
        s = 0.0
        n = 0

        for Xb, Yb, mu in train_loader:
            Xb = Xb.to(cfg.device)
            Yb = Yb.to(cfg.device)
            mu = mu.to(cfg.device)

            with torch.no_grad():
                Zx, pos, _ = ae(Xb, M=cfg.M, Np=cfg.Np)
                Zy, _pos_y, _ = ae(Yb, M=cfg.M, Np=cfg.Np)

            Zhat = op(Zx, mu, pos)

            loss_tok = torch.mean((Zhat - Zy) ** 2)

            # transport in cloud space via stochastic decode
            Yhat = ae.decode_sample(Zhat, pos, Np=cfg.Np)
            loss_ch = chamfer_l2(Yhat, Yb)

            loss_smooth = smoothness_1d_tokens(Zhat)

            loss = cfg.w_token * loss_tok + cfg.w_chamfer * loss_ch + cfg.w_smooth * loss_smooth

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss; skipping step")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op.parameters(), 1.0)
            opt.step()

            bsz = Xb.size(0)
            s += float(loss.item()) * bsz
            n += bsz

        tr = s / max(1, n)
        valm = eval_stage_b(ae, op, val_loader, cfg)

        if valm["val_score"] < best:
            best = valm["val_score"]
            best_state = {}
            for k, v in op.state_dict().items():
                if torch.is_tensor(v):
                    best_state[k] = v.detach().cpu().clone()
                else:
                    best_state[k] = v

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[OP] E{epoch:03d} train={tr:.4e} "
                f"val_tok={valm['token_mse']:.4e} val_ch={valm['chamfer']:.4e} "
                f"val_smooth={valm['smooth']:.4e} val_score={valm['val_score']:.4e}"
            )

    if best_state is not None:
        best_state.pop("_metadata", None)
        op.load_state_dict(best_state, strict=False)

    sd = op.state_dict()
    sd.pop("_metadata", None)
    torch.save({"state_dict": sd, "config": asdict(cfg)}, cfg.op_ckpt_path)
    print("[OP] saved:", cfg.op_ckpt_path)
    return op

# --- ADD THIS FUNCTION (near the bottom, above main()) ---
def train_stage_b_only():
    """
    Load a saved AE checkpoint and train only Stage B (LatentTokenNeuralOp).
    Requires env var DEMO003_AE pointing to the AE ckpt.
    """
    dataset_path = os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("Set DATASET_PATH")

    ae_ckpt_path = os.environ.get("DEMO003_AE", "")
    if not ae_ckpt_path:
        raise ValueError("Set DEMO003_AE to the AE checkpoint path (demo003_latent1d_ae.pt)")

    cfg = Config(dataset_path=dataset_path)

    # seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # data
    data = load_npz(cfg.dataset_path)
    train_ds = CloudDataset(data, "train", cfg.Np)
    val_ds = CloudDataset(data, "val", cfg.Np)

    train_loader_op = DataLoader(train_ds, batch_size=cfg.op_batch_size, shuffle=True, num_workers=0)
    val_loader_op = DataLoader(val_ds, batch_size=cfg.op_batch_size, shuffle=False, num_workers=0)

    # load AE
    ae = Latent1DTokenAE(cfg.token_dim, cfg.particle_hidden, cfg.token_hidden).to(cfg.device)
    ckpt = torch.load(ae_ckpt_path, map_location=cfg.device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if isinstance(sd, dict):
        sd.pop("_metadata", None)
    ae.load_state_dict(sd, strict=False)
    ae.eval()

    # ensure output dirs
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.op_ckpt_path).parent).mkdir(parents=True, exist_ok=True)
    Path(cfg.meta_path).write_text(json.dumps({"config": asdict(cfg)}, indent=2))

    # train only B
    _ = train_stage_b(cfg, ae, train_loader_op, val_loader_op)
# ----------------------------
# Main
# ----------------------------
def main1():
    dataset_path = os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("Set DATASET_PATH")

    cfg = Config(dataset_path=dataset_path)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.ae_ckpt_path).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.op_ckpt_path).parent).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = load_npz(cfg.dataset_path)
    train_ds = CloudDataset(data, "train", cfg.Np)
    val_ds = CloudDataset(data, "val", cfg.Np)

    train_loader_ae = DataLoader(train_ds, batch_size=cfg.ae_batch_size, shuffle=True, num_workers=0)
    val_loader_ae = DataLoader(val_ds, batch_size=cfg.ae_batch_size, shuffle=False, num_workers=0)

    train_loader_op = DataLoader(train_ds, batch_size=cfg.op_batch_size, shuffle=True, num_workers=0)
    val_loader_op = DataLoader(val_ds, batch_size=cfg.op_batch_size, shuffle=False, num_workers=0)

    # save meta
    Path(cfg.meta_path).write_text(json.dumps({"config": asdict(cfg)}, indent=2))
    print("[OK] wrote meta:", cfg.meta_path)
    print("[INFO] device:", cfg.device, "Np:", cfg.Np, "M:", cfg.M, "token_dim:", cfg.token_dim)

    # Stage A
    ae = train_stage_a(cfg, train_loader_ae, val_loader_ae)

    # Stage B
    _ = train_stage_b(cfg, ae, train_loader_op, val_loader_op)

def main():
    dataset_path = os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("Set DATASET_PATH")

    # Choose stage via env var:
    #   STAGE="A+B" (default) -> train AE then operator
    #   STAGE="A"            -> train AE only
    #   STAGE="B"            -> train operator only (loads AE from DEMO003_AE)
    stage = os.environ.get("STAGE", "B").strip().upper()

    cfg = Config(dataset_path=dataset_path)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.ae_ckpt_path).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.op_ckpt_path).parent).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = load_npz(cfg.dataset_path)
    train_ds = CloudDataset(data, "train", cfg.Np)
    val_ds = CloudDataset(data, "val", cfg.Np)

    train_loader_ae = DataLoader(train_ds, batch_size=cfg.ae_batch_size, shuffle=True, num_workers=0)
    val_loader_ae = DataLoader(val_ds, batch_size=cfg.ae_batch_size, shuffle=False, num_workers=0)

    train_loader_op = DataLoader(train_ds, batch_size=cfg.op_batch_size, shuffle=True, num_workers=0)
    val_loader_op = DataLoader(val_ds, batch_size=cfg.op_batch_size, shuffle=False, num_workers=0)

    # save meta
    Path(cfg.meta_path).write_text(json.dumps({"config": asdict(cfg)}, indent=2))
    print("[OK] wrote meta:", cfg.meta_path)
    print("[INFO] device:", cfg.device, "Np:", cfg.Np, "M:", cfg.M, "token_dim:", cfg.token_dim)
    print("[INFO] STAGE =", stage)

    if stage == "A":
        _ = train_stage_a(cfg, train_loader_ae, val_loader_ae)
        return

    if stage == "B":
        ae_ckpt_path = os.environ.get("DEMO003_AE", "").strip()
        if not ae_ckpt_path:
            raise ValueError("STAGE=B requires DEMO003_AE=/path/to/demo003_latent1d_ae.pt")

        # load AE checkpoint
        ae = Latent1DTokenAE(cfg.token_dim, cfg.particle_hidden, cfg.token_hidden).to(cfg.device)
        ckpt = torch.load(ae_ckpt_path, map_location=cfg.device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if isinstance(sd, dict):
            sd.pop("_metadata", None)
        ae.load_state_dict(sd, strict=False)
        ae.eval()

        _ = train_stage_b(cfg, ae, train_loader_op, val_loader_op)
        return

    # default: A+B
    ae = train_stage_a(cfg, train_loader_ae, val_loader_ae)
    _ = train_stage_b(cfg, ae, train_loader_op, val_loader_op)



if __name__ == "__main__":
    main()