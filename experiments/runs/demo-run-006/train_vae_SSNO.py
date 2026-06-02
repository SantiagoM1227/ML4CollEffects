# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence, Union, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#from neuralop.models.gino import GINO


# ---------------------------
# Data helpers
# ---------------------------
def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


class CloudDataset(Dataset):
    """
    Expects data dict with:
      X_cloud: [Ns, Np, 6]
      Y_cloud: [Ns, Np, 6]
      MU:      [Ns, m]
    Optionally:
      train / val / test: index arrays
    """
    def __init__(self, data: dict, split: str = "train"):
        self.X = data["X_cloud"]
        self.Y = data["Y_cloud"]
        self.MU = data["MU"]

        if split in data:
            self.idx = data[split].astype(np.int64)
        else:
            self.idx = np.arange(self.X.shape[0], dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        X = torch.from_numpy(self.X[j]).float()   # [Np,6]
        Y = torch.from_numpy(self.Y[j]).float()   # [Np,6]
        MU = torch.from_numpy(self.MU[j]).float() # [m]
        return X, Y, MU


def make_dummy_lattice_tokens(B: int, N: int, device: torch.device):
    """
    Placeholder lattice tokens:
      elem_params: [B,N,7] = [L,K1,K2,phi,Vrf,frf,phirf]
      elem_s: [B,N] cumulative s at entrance
    """
    elem_params = torch.zeros(B, N, 7, device=device, dtype=torch.float32)
    elem_params[..., 0] = 1.0  # L=1m
    elem_s = torch.cumsum(elem_params[..., 0], dim=1) - elem_params[..., 0]
    return elem_params, elem_s


# ---------------------------
# Physics/feature helpers
# ---------------------------
def beam_centroid_sigma(cloud: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    centroid = cloud.mean(dim=1)
    var = (cloud - centroid[:, None, :]).pow(2).mean(dim=1)
    sigma = torch.sqrt(var + eps)
    return centroid, sigma


PAIR_INDEX = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]


def _hist2d_bucketize(
    x: torch.Tensor,
    y: torch.Tensor,
    x_edges: torch.Tensor,
    y_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Non-differentiable 2D histogram via torch.bucketize.
    x,y: [B,Np]
    edges: [nbins+1]
    return: [B,nb,nb]
    """
    # avoids torch.searchsorted non-contiguous warning
    x = x.contiguous()
    y = y.contiguous()

    B, Np = x.shape
    nb = x_edges.numel() - 1

    ix = torch.bucketize(x, x_edges) - 1
    iy = torch.bucketize(y, y_edges) - 1
    ix = ix.clamp(0, nb - 1)
    iy = iy.clamp(0, nb - 1)

    flat = ix * nb + iy  # [B,Np]
    hist = torch.zeros(B, nb * nb, device=x.device, dtype=torch.float32)
    ones = torch.ones_like(flat, dtype=torch.float32)
    hist.scatter_add_(dim=1, index=flat, src=ones)
    return hist.view(B, nb, nb)


def cloud_to_pairwise_hists(
    cloud: torch.Tensor,
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert cloud [B,Np,6] -> [B,15,nb,nb] normalized histograms.
    Range per coord = centroid +/- clip_k*sigma.
    """
    B, Np, D = cloud.shape
    assert D == 6, f"Expected 6D, got {D}"

    centroid, sigma = beam_centroid_sigma(cloud, eps=eps)  # [B,6]

    edges = []
    t = torch.linspace(0.0, 1.0, nbins + 1, device=cloud.device, dtype=torch.float32)[None, :]
    for d in range(6):
        lo = centroid[:, d] - clip_k * sigma[:, d]
        hi = centroid[:, d] + clip_k * sigma[:, d]
        hi = torch.where((hi - lo) < 1e-9, lo + 1e-9, hi)
        ed = lo[:, None].to(torch.float32) * (1 - t) + hi[:, None].to(torch.float32) * t
        edges.append(ed)  # [B,nb+1]

    out = []
    for (i, j) in PAIR_INDEX:
        xi = cloud[:, :, i].to(torch.float32)
        yj = cloud[:, :, j].to(torch.float32)
        h = torch.zeros(B, nbins, nbins, device=cloud.device, dtype=torch.float32)
        for b in range(B):
            hb = _hist2d_bucketize(xi[b:b+1], yj[b:b+1], edges[i][b], edges[j][b])
            h[b] = hb[0]
        h = h / (float(Np) + eps)
        out.append(h)

    return torch.stack(out, dim=1)  # [B,15,nb,nb]


# ---------------------------
# Model components
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class BeamVAE(nn.Module):
    def __init__(self, in_ch: int = 15, latent_dim: int = 256, aux_dim: int = 12, base_ch: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.enc_feat_dim = (base_ch * 8) * 4 * 4

        self.fc = nn.Linear(self.enc_feat_dim + aux_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.enc_feat_dim)
        self.dec1 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec3 = UpBlock(base_ch * 2, base_ch)
        self.dec4 = UpBlock(base_ch, base_ch)
        self.dec_out = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.aux_sigma_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )
        self.aux_centroid_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )

    def encode(self, x_hist: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x_hist)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = h.flatten(1)
        h = torch.cat([h, aux], dim=1)
        h = F.gelu(self.fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(z.shape[0], -1, 4, 4)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        x_hat = self.dec_out(h)
        x_hat = torch.clamp(x_hat, min=0.0)
        x_hat = x_hat / (x_hat.sum(dim=(-1, -2), keepdim=True) + 1e-12)
        return x_hat

    def forward(self, x_hist: torch.Tensor, aux: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x_hist, aux)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        sigma_hat = self.aux_sigma_head(z)
        centroid_hat = self.aux_centroid_head(z)
        return {"mu": mu, "logvar": logvar, "z": z, "x_hat": x_hat, "sigma_hat": sigma_hat, "centroid_hat": centroid_hat}


@dataclass
class ElementNormScales:
    L: float = 1.0
    K1: float = 10.0
    K2: float = 10.0
    phi: float = 2.0 * 3.141592653589793
    Vrf: float = 1.0
    frf: float = 1.0
    phirf: float = 2.0 * 3.141592653589793


def normalize_elem_params(p: torch.Tensor, scales: ElementNormScales) -> torch.Tensor:
    L, K1, K2, phi, Vrf, frf, phirf = torch.unbind(p, dim=-1)
    L = L / scales.L
    K1 = K1 / scales.K1
    K2 = K2 / scales.K2
    phi = phi / scales.phi
    Vrf = torch.log1p(torch.clamp(Vrf, min=0.0))
    frf = torch.log1p(torch.clamp(frf, min=0.0))
    phirf = phirf / scales.phirf
    return torch.stack([L, K1, K2, phi, Vrf, frf, phirf], dim=-1)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, n_freq_pairs: int = 32, wl_min: float = 1e-2, wl_max: float = 1e3):
        super().__init__()
        wls = torch.logspace(torch.log10(torch.tensor(wl_min)), torch.log10(torch.tensor(wl_max)), steps=n_freq_pairs)
        freqs = 2.0 * 3.141592653589793 / wls
        self.register_buffer("freqs", freqs)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s[..., None] * self.freqs[None, None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B,N,64]


class ElementTokenizer(nn.Module):
    def __init__(self, d_model: int = 512, param_dim: int = 7, pos_dim: int = 64):
        super().__init__()
        self.pos = FourierPositionalEncoding(n_freq_pairs=32)
        self.pos_proj = nn.Linear(pos_dim, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, elem_params_norm: torch.Tensor, elem_s: torch.Tensor) -> torch.Tensor:
        h = self.mlp(elem_params_norm)
        pe = self.pos(elem_s)
        pe = self.pos_proj(pe)
        return h + pe


# ---------------------------
# Tracker: GINO (NeuralOperator)
# ---------------------------
class TrackingGNO(nn.Module):
    """
    Causal Neural Operator on the 1D lattice (elements as nodes).

    Implements a kernel integral/message-passing update:
      h_i^{l+1} = LN( h_i^l + sum_{j<=i} K_theta(xi_i, xi_j) * V(h_j^l) )

    Then predicts latent increments Δz_i and integrates:
      z_i = z0 + cumsum(Δz)
    """
    def __init__(
        self,
        latent_dim: int = 256,
        d_model: int = 512,
        mu_dim: int = 0,
        n_layers: int = 4,
        kernel_hidden: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.mu_dim = mu_dim
        self.n_layers = n_layers

        self.z_proj = nn.Linear(latent_dim, d_model)
        self.mu_proj = nn.Linear(mu_dim, d_model) if mu_dim > 0 else None

        # Node input features: concat(z0_embed, elem_token, mu_embed)
        in_dim = d_model + d_model + (d_model if mu_dim > 0 else 0)
        self.node_in = nn.Linear(in_dim, d_model)

        # Build "operator coordinates/features" xi_i used by kernel K_theta
        # Use (s_i, elem_token_i) as kernel inputs.
        # s_i is scalar; elem_token_i is d_model.
        self.xi_proj = nn.Sequential(
            nn.Linear(1 + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Kernel network: maps pair (xi_i, xi_j, s_i - s_j) -> mixing weights
        # Output is a scalar gate per head (simple + stable), then applied to value vectors.
        self.kernel = nn.Sequential(
            nn.Linear(2 * d_model + 1, kernel_hidden),
            nn.GELU(),
            nn.Linear(kernel_hidden, kernel_hidden),
            nn.GELU(),
            nn.Linear(kernel_hidden, 1),
        )

        self.value = nn.Linear(d_model, d_model, bias=False)

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(n_layers)
        ])

        self.to_dz = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    @staticmethod
    def causal_mask(N: int, device) -> torch.Tensor:
        # True where j > i (future), to be masked out
        return torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, z0: torch.Tensor, elem_tokens: torch.Tensor, elem_s: torch.Tensor, mu: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z0: [B,latent]
        elem_tokens: [B,N,d_model]
        elem_s: [B,N]
        mu: [B,mu_dim]
        returns: [B,N,latent]
        """
        B, N, _ = elem_tokens.shape
        device = elem_tokens.device

        z0d = self.z_proj(z0)[:, None, :].expand(B, N, -1)  # [B,N,d]
        pieces = [z0d, elem_tokens]

        if self.mu_proj is not None:
            if mu is None:
                raise ValueError("TrackingGNO constructed with mu_dim>0 but mu=None passed.")
            mud = self.mu_proj(mu)[:, None, :].expand(B, N, -1)
            pieces.append(mud)

        h = self.node_in(torch.cat(pieces, dim=-1))  # [B,N,d]

        # xi_i = f([s_i, token_i])
        s_in = elem_s[:, :, None].to(h.dtype)  # [B,N,1]
        xi = self.xi_proj(torch.cat([s_in, elem_tokens], dim=-1))  # [B,N,d]

        # Build pairwise tensors for kernel
        # xi_i: [B,N,1,d], xi_j: [B,1,N,d]
        xi_i = xi[:, :, None, :]
        xi_j = xi[:, None, :, :]

        # Δs = s_i - s_j, shape [B,N,N,1]
        ds = (elem_s[:, :, None] - elem_s[:, None, :]).to(h.dtype)[:, :, :, None]

        # kernel input: [B,N,N, 2d + 1]
        k_in = torch.cat([xi_i.expand(-1, -1, N, -1), xi_j.expand(-1, N, -1, -1), ds], dim=-1)

        # causal mask
        fut = self.causal_mask(N, device=device)  # [N,N]
        fut = fut[None, :, :].expand(B, -1, -1)   # [B,N,N]

        # message passing layers
        for l in range(self.n_layers):
            hn = self.norms[l](h)

            # values V(h_j)
            v = self.value(hn)  # [B,N,d]
            vj = v[:, None, :, :].expand(B, N, N, self.d_model)  # [B,N,N,d]

            # scalar gates a_ij
            a = self.kernel(k_in).squeeze(-1)  # [B,N,N]
            a = a.masked_fill(fut, 0.0)

            # normalize by number of allowed neighbors (avoid scale blow-up)
            denom = (~fut).to(h.dtype).sum(dim=-1).clamp_min(1.0)  # [B,N]
            a = a / denom[:, :, None]

            # aggregate: sum_j a_ij * v_j
            agg = torch.einsum("bij, bijd -> bid", a, vj)  # [B,N,d]

            h = h + agg
            h = h + self.ffns[l](self.norms[l](h))

        dz = self.to_dz(h)  # [B,N,latent]
        z_traj = z0[:, None, :] + torch.cumsum(dz, dim=1)
        return z_traj
# ---------------------------
# MU preprocessing
# ---------------------------
MUOp = Literal["identity", "scale", "log", "log10", "log_log", "signed_log", "signed_log_log"]
SpecItem = Union[float, int, str, dict]


def _apply_op(x: torch.Tensor, op: str, eps: float = 1e-12) -> torch.Tensor:
    op = op.lower()
    if op in ("identity", "none", ""):
        return x
    if op in ("log", "log1p"):
        return torch.log1p(torch.clamp(x, min=0.0))
    if op in ("log10", "log10_1p"):
        return torch.log10(1.0 + torch.clamp(x, min=0.0))
    if op == "log_log":
        y = torch.log1p(torch.clamp(x, min=0.0))
        return torch.log1p(y)
    if op in ("signed_log", "signed_log1p"):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    if op == "signed_log_log":
        y = torch.sign(x) * torch.log1p(torch.abs(x))
        return torch.sign(y) * torch.log1p(torch.abs(y))
    raise ValueError(f"Unknown MU op: {op}")


@dataclass
class MUPreprocessConfig:
    specs: Sequence[SpecItem]
    standardize: bool = True
    clip: Optional[float] = None
    eps: float = 1e-12


class MUPreprocessor(nn.Module):
    def __init__(self, cfg: MUPreprocessConfig):
        super().__init__()
        self.cfg = cfg
        self.m = len(cfg.specs)
        self.register_buffer("mean", torch.zeros(self.m, dtype=torch.float32))
        self.register_buffer("std", torch.ones(self.m, dtype=torch.float32))
        self.fitted = False

    def _parse_spec(self, spec: SpecItem):
        unit = 1.0
        op = "identity"
        clip = self.cfg.clip
        if isinstance(spec, (int, float)):
            unit = float(spec)
        elif isinstance(spec, str):
            op = spec
        elif isinstance(spec, dict):
            unit = float(spec.get("unit", 1.0))
            op = str(spec.get("op", "identity"))
            if "clip" in spec:
                clip = float(spec["clip"])
        else:
            raise TypeError(f"Unsupported MU spec item type: {type(spec)}")
        if unit == 0.0:
            unit = 1.0
        return unit, op, clip

    def transform(self, mu: torch.Tensor) -> torch.Tensor:
        assert mu.ndim == 2 and mu.shape[1] == self.m
        cols: List[torch.Tensor] = []
        for j, spec in enumerate(self.cfg.specs):
            unit, op, clip = self._parse_spec(spec)
            x = mu[:, j] / unit
            x = _apply_op(x, op, eps=self.cfg.eps)
            if clip is not None:
                x = torch.clamp(x, -clip, clip)
            cols.append(x)
        return torch.stack(cols, dim=1)

    @torch.no_grad()
    def fit(self, mu_train_numpy: np.ndarray):
        x = torch.tensor(mu_train_numpy, dtype=torch.float32)
        xt = self.transform(x)
        self.mean.copy_(xt.mean(dim=0))
        std = xt.std(dim=0)
        std = torch.where(std < 1e-12, torch.ones_like(std), std)
        self.std.copy_(std)
        self.fitted = True

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        xt = self.transform(mu)
        if self.cfg.standardize:
            if not self.fitted:
                raise RuntimeError("MUPreprocessor.standardize=True but fit() was not called.")
            xt = (xt - self.mean) / (self.std + self.cfg.eps)
        return xt


# ---------------------------
# Full model
# ---------------------------
class BeamLatentTrackingModel(nn.Module):
    def __init__(self, mu_dim: int, latent_dim: int = 256, d_model: int = 512):
        super().__init__()
        self.mu_dim = mu_dim
        self.latent_dim = latent_dim

        self.vae = BeamVAE(in_ch=15, latent_dim=latent_dim, aux_dim=12, base_ch=32)

        self.elem_scales = ElementNormScales()
        self.tokenizer = ElementTokenizer(d_model=d_model, param_dim=7, pos_dim=64)

        # GINO tracker
        self.tracker = TrackingGNO(latent_dim=latent_dim, d_model=d_model, mu_dim=mu_dim)

        self.mu_to_z = nn.Sequential(
            nn.Linear(mu_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self,
        X_cloud: torch.Tensor,
        MU: torch.Tensor,
        elem_params: torch.Tensor,
        elem_s: torch.Tensor,
        *,
        mode: str = "AR",
        z_prev_traj: Optional[torch.Tensor] = None,
        ar_nograd: bool = False,
    ) -> Dict[str, torch.Tensor]:
        _ = (mode, z_prev_traj, ar_nograd)  # kept for CLI compatibility

        x_hist = cloud_to_pairwise_hists(X_cloud, nbins=64)  # [B,15,64,64]
        centroid, sigma = beam_centroid_sigma(X_cloud)
        aux = torch.cat([sigma, centroid], dim=1)  # [B,12]

        vae_out = self.vae(x_hist, aux)
        z0 = vae_out["z"]
        z0 = z0 + self.mu_to_z(MU.to(z0.dtype))

        elem_params_norm = normalize_elem_params(elem_params, self.elem_scales)
        elem_tokens = self.tokenizer(elem_params_norm, elem_s)

        # GNO-based trajectory
        z_pred = self.tracker(z0, elem_tokens, elem_s, mu=MU)  # [B,N,latent]

        zN = z_pred[:, -1, :]
        xN_hat = self.vae.decode(zN)
        sigmaN_hat = self.vae.aux_sigma_head(zN)
        centroidN_hat = self.vae.aux_centroid_head(zN)

        return {
            "x_hist": x_hist,
            "aux": aux,
            "z0": z0,
            "vae": vae_out,
            "z_pred_traj": z_pred,
            "zN": zN,
            "xN_hat": xN_hat,
            "sigmaN_hat": sigmaN_hat,
            "centroidN_hat": centroidN_hat,
        }


# ---------------------------
# Losses
# ---------------------------
@dataclass
class LossWeights:
    beta: float = 1e-5
    gamma: float = 1e-4
    delta: float = 1e-4


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)


def vae_recon_loss_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(dim=1)


def compute_total_loss(model_out: Dict[str, torch.Tensor], Y_cloud: torch.Tensor, weights: LossWeights) -> Dict[str, torch.Tensor]:
    x_hist0 = model_out["x_hist"]
    x0_hat = model_out["vae"]["x_hat"]
    mu = model_out["vae"]["mu"]
    logvar = model_out["vae"]["logvar"]

    recon0 = vae_recon_loss_mse(x0_hat, x_hist0)
    kl = kl_divergence_standard_normal(mu, logvar)

    sigma0_hat = model_out["vae"]["sigma_hat"]
    centroid0_hat = model_out["vae"]["centroid_hat"]
    aux0 = model_out["aux"]
    sigma0 = aux0[:, :6]
    centroid0 = aux0[:, 6:]
    sigma_loss0 = F.mse_loss(sigma0_hat, sigma0, reduction="none").mean(dim=1)
    centroid_loss0 = F.mse_loss(centroid0_hat, centroid0, reduction="none").mean(dim=1)

    y_hist = cloud_to_pairwise_hists(Y_cloud, nbins=64)
    reconN = vae_recon_loss_mse(model_out["xN_hat"], y_hist)
    y_centroid, y_sigma = beam_centroid_sigma(Y_cloud)
    sigma_lossN = F.mse_loss(model_out["sigmaN_hat"], y_sigma, reduction="none").mean(dim=1)
    centroid_lossN = F.mse_loss(model_out["centroidN_hat"], y_centroid, reduction="none").mean(dim=1)

    total = (
        recon0 + reconN
        + weights.beta * kl
        + weights.gamma * (sigma_loss0 + sigma_lossN)
        + weights.delta * (centroid_loss0 + centroid_lossN)
    )

    return {
        "loss": total.mean(),
        "recon0": recon0.mean(),
        "reconN": reconN.mean(),
        "kl": kl.mean(),
        "sigma_loss0": sigma_loss0.mean(),
        "centroid_loss0": centroid_loss0.mean(),
        "sigma_lossN": sigma_lossN.mean(),
        "centroid_lossN": centroid_lossN.mean(),
    }


def compute_vae_only_loss(vae_out: Dict[str, torch.Tensor], x_hist: torch.Tensor, aux: torch.Tensor, weights: LossWeights) -> Dict[str, torch.Tensor]:
    recon0 = vae_recon_loss_mse(vae_out["x_hat"], x_hist)
    kl = kl_divergence_standard_normal(vae_out["mu"], vae_out["logvar"])
    sigma = aux[:, :6]
    centroid = aux[:, 6:]
    sigma_loss = F.mse_loss(vae_out["sigma_hat"], sigma, reduction="none").mean(dim=1)
    centroid_loss = F.mse_loss(vae_out["centroid_hat"], centroid, reduction="none").mean(dim=1)

    total = recon0 + weights.beta * kl + weights.gamma * sigma_loss + weights.delta * centroid_loss
    return {
        "loss": total.mean(),
        "recon0": recon0.mean(),
        "kl": kl.mean(),
        "sigma_loss0": sigma_loss.mean(),
        "centroid_loss0": centroid_loss.mean(),
    }


# ---------------------------
# Train
# ---------------------------
def save_ckpt(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to .npz dataset")
    ap.add_argument("--mode", required=True, choices=["vae", "dynamics"], help="Train VAE-only or internal dynamics")
    ap.add_argument("--out_ckpt", required=True, help="Output checkpoint .pt")
    ap.add_argument("--vae_ckpt", default="", help="(dynamics mode) path to pretrained vae checkpoint")
    ap.add_argument("--freeze_vae", action="store_true", help="(dynamics mode) freeze VAE weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_elem", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data = load_npz(args.data)

    train_ds = CloudDataset(data, split="train" if "train" in data else "train")
    val_ds = CloudDataset(data, split="val" if "val" in data else "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    mu_dim = int(data["MU"].shape[-1])
    weights = LossWeights(beta=args.beta, gamma=args.gamma, delta=args.delta)

    # MU preprocessing (fit on train split)
    MU_spec = [
        1e-9,                # example: Q_bunch in Coulombs -> nC
        1e-3,                # example: pipe_radius m -> mm
        {"unit": 1.0, "op": "log"},  # example: impedance scale -> log1p
    ]
    mu_cfg = MUPreprocessConfig(specs=MU_spec, standardize=True, clip=10.0)
    mu_prep = MUPreprocessor(mu_cfg).to(device)

    # If MU dimension differs from the spec length, fall back to identity+standardize.
    if len(MU_spec) != mu_dim:
        MU_spec = [1.0] * mu_dim
        mu_cfg = MUPreprocessConfig(specs=MU_spec, standardize=True, clip=10.0)
        mu_prep = MUPreprocessor(mu_cfg).to(device)

    mu_train_np = data["MU"][data["train"]] if "train" in data else data["MU"]
    mu_prep.fit(mu_train_np)

    model = BeamLatentTrackingModel(mu_dim=mu_dim, latent_dim=256, d_model=512).to(device)

    if args.mode == "dynamics" and args.vae_ckpt:
        ck = torch.load(args.vae_ckpt, map_location="cpu")
        # Supports either direct state_dict or wrapped dict.
        vae_sd = ck["vae_state_dict"] if "vae_state_dict" in ck else ck.get("state_dict", ck)
        missing, unexpected = model.vae.load_state_dict(vae_sd, strict=False)
        print("Loaded VAE:", "missing=", missing, "unexpected=", unexpected)

    if args.mode == "dynamics" and args.freeze_vae:
        for p in model.vae.parameters():
            p.requires_grad = False

    # Choose params to optimize
    if args.mode == "vae":
        params = list(model.vae.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def eval_loop() -> Dict[str, float]:
        model.eval()
        acc: Dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for X, Y, MU in val_loader:
                X, Y, MU = X.to(device), Y.to(device), MU.to(device)
                MUe = mu_prep(MU)

                if args.mode == "vae":
                    x_hist = cloud_to_pairwise_hists(X, nbins=64)
                    cent, sig = beam_centroid_sigma(X)
                    aux = torch.cat([sig, cent], dim=1)
                    vae_out = model.vae(x_hist, aux)
                    ld = compute_vae_only_loss(vae_out, x_hist, aux, weights)
                else:
                    B = X.shape[0]
                    elem_params, elem_s = make_dummy_lattice_tokens(B, args.n_elem, device)
                    out = model(X, MUe, elem_params, elem_s, mode="AR")
                    ld = compute_total_loss(out, Y, weights)

                for k, v in ld.items():
                    acc[k] = acc.get(k, 0.0) + float(v.item())
                n += 1

        for k in list(acc.keys()):
            acc[k] /= max(n, 1)
        return acc

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for step, (X, Y, MU) in enumerate(train_loader):
            X, Y, MU = X.to(device), Y.to(device), MU.to(device)
            MUe = mu_prep(MU)

            opt.zero_grad(set_to_none=True)

            if args.mode == "vae":
                x_hist = cloud_to_pairwise_hists(X, nbins=64)
                cent, sig = beam_centroid_sigma(X)
                aux = torch.cat([sig, cent], dim=1)
                vae_out = model.vae(x_hist, aux)
                ld = compute_vae_only_loss(vae_out, x_hist, aux, weights)
                loss = ld["loss"]
            else:
                B = X.shape[0]
                elem_params, elem_s = make_dummy_lattice_tokens(B, args.n_elem, device)
                out = model(X, MUe, elem_params, elem_s, mode="AR")
                ld = compute_total_loss(out, Y, weights)
                loss = ld["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                msg = f"epoch={epoch} step={step} "
                msg += " ".join([f"{k}={float(v.item()):.6f}" for k, v in ld.items() if k in ("loss", "recon0", "reconN", "kl")])
                print(msg)

        val = eval_loop()
        print(f"[epoch {epoch}] val: " + " ".join([f"{k}={val[k]:.6f}" for k in sorted(val.keys())]))

        # save best by val loss
        if val["loss"] < best_val:
            best_val = val["loss"]
            if args.mode == "vae":
                save_ckpt(args.out_ckpt, {
                    "mode": "vae",
                    "vae_state_dict": model.vae.state_dict(),
                    "mu_prep_state_dict": mu_prep.state_dict(),
                    "mu_dim": mu_dim,
                    "weights": vars(weights),
                    "val_metrics": val,
                })
            else:
                save_ckpt(args.out_ckpt, {
                    "mode": "dynamics",
                    "state_dict": model.state_dict(),
                    "mu_prep_state_dict": mu_prep.state_dict(),
                    "mu_dim": mu_dim,
                    "weights": vars(weights),
                    "val_metrics": val,
                })
            print(f"Saved best checkpoint to {args.out_ckpt} (val loss={best_val:.6f})")

    print(f"Done. total_time_sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()