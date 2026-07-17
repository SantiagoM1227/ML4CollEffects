"""
beam_global_model.py

Reusable PyTorch module for the thesis CVAE + GNO + physical-statistics pipeline.

functions: 
---------
1. Convert an initial 6D particle cloud into 15 normalized pairwise histograms.
2. Encode those histograms with a conditional VAE.
3. Propagate the latent state through a lattice with a Graph Neural Operator.
4. Decode a normalized final distribution-shape tensor.
5. Predict final physical moments with a separate moment/statistics head:
      (sigma_0, centroid_0, machine_mu) -> (sigma_N, centroid_N)
6. Attach physical bin edges and densities to the decoded shape using the predicted
   final moments, so the output histograms have physical units.

Expected coordinate order
-------------------------
    (x, px, y, py, zeta, delta)

The model returns both:
    - hist_mass: per-bin probability mass, each pairwise channel sums to 1
    - hist_density: probability density in physical units, mass divided by dx*dy

This file is intentionally notebook-friendly: copy it next to your notebook and import with
    from beam_global_model import *
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Constants and basic cloud utilities
# -----------------------------------------------------------------------------

COORD_NAMES: Tuple[str, ...] = ("x", "px", "y", "py", "zeta", "delta")

PAIR_INDEX: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]

PAIR_NAMES: Tuple[str, ...] = tuple(f"{COORD_NAMES[i]}_{COORD_NAMES[j]}" for i, j in PAIR_INDEX)


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def beam_centroid_sigma(cloud: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute physical centroids and RMS beam sizes.

    Parameters
    ----------
    cloud:
        Tensor [B, Np, 6].

    Returns
    -------
    centroid, sigma:
        Tensors [B, 6], [B, 6].
    """
    if cloud.ndim != 3 or cloud.shape[-1] != 6:
        raise ValueError(f"cloud must have shape [B,Np,6], got {tuple(cloud.shape)}")
    centroid = cloud.mean(dim=1)
    sigma = cloud.std(dim=1, unbiased=False).clamp_min(eps)
    return centroid, sigma


def moments_to_aux(sigma: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Concatenate sigma and centroid into the 12-vector used by the CVAE."""
    return torch.cat([sigma, centroid], dim=-1)


def cloud_to_aux(cloud: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centroid, sigma = beam_centroid_sigma(cloud, eps=eps)
    aux = moments_to_aux(sigma, centroid)
    return aux, sigma, centroid


def _hist2d_one_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    x_edges: torch.Tensor,
    y_edges: torch.Tensor,
) -> torch.Tensor:
    """Fast non-differentiable 2D histogram for one sample."""
    nb = x_edges.numel() - 1
    ix = torch.bucketize(x.contiguous(), x_edges.contiguous()) - 1
    iy = torch.bucketize(y.contiguous(), y_edges.contiguous()) - 1
    ix = ix.clamp(0, nb - 1)
    iy = iy.clamp(0, nb - 1)
    flat = ix * nb + iy
    hist = torch.zeros(nb * nb, device=x.device, dtype=torch.float32)
    hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return hist.view(nb, nb)


def edges_from_sigma_centroid(
    sigma: torch.Tensor,
    centroid: torch.Tensor,
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build physical bin edges for all six coordinates.

    Returns
    -------
    edges:
        Tensor [B, 6, nbins+1], where each coordinate spans
        centroid_i +/- clip_k * sigma_i.
    """
    if sigma.shape != centroid.shape or sigma.ndim != 2 or sigma.shape[1] != 6:
        raise ValueError("sigma and centroid must both have shape [B,6]")
    B = sigma.shape[0]
    sigma = sigma.clamp_min(eps)
    lo = centroid - clip_k * sigma
    hi = centroid + clip_k * sigma
    hi = torch.where((hi - lo) < eps, lo + eps, hi)
    t = torch.linspace(0.0, 1.0, nbins + 1, device=sigma.device, dtype=sigma.dtype)
    return lo[:, :, None] * (1.0 - t[None, None, :]) + hi[:, :, None] * t[None, None, :]


def pairwise_edges_from_stats(
    sigma: torch.Tensor,
    centroid: torch.Tensor,
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Build physical edges and centers for every pairwise marginal.

    Returns a dictionary with:
        coord_edges: [B,6,nbins+1]
        pair_x_edges: [B,15,nbins+1]
        pair_y_edges: [B,15,nbins+1]
        pair_x_centers: [B,15,nbins]
        pair_y_centers: [B,15,nbins]
        pair_dxdy: [B,15,nbins,nbins]
    """
    coord_edges = edges_from_sigma_centroid(sigma, centroid, nbins=nbins, clip_k=clip_k, eps=eps)
    x_edges = []
    y_edges = []
    for i, j in PAIR_INDEX:
        x_edges.append(coord_edges[:, i, :])
        y_edges.append(coord_edges[:, j, :])
    pair_x_edges = torch.stack(x_edges, dim=1)
    pair_y_edges = torch.stack(y_edges, dim=1)
    pair_x_centers = 0.5 * (pair_x_edges[:, :, :-1] + pair_x_edges[:, :, 1:])
    pair_y_centers = 0.5 * (pair_y_edges[:, :, :-1] + pair_y_edges[:, :, 1:])
    dx = (pair_x_edges[:, :, 1:] - pair_x_edges[:, :, :-1]).clamp_min(eps)
    dy = (pair_y_edges[:, :, 1:] - pair_y_edges[:, :, :-1]).clamp_min(eps)
    pair_dxdy = dx[:, :, :, None] * dy[:, :, None, :]
    return {
        "coord_edges": coord_edges,
        "pair_x_edges": pair_x_edges,
        "pair_y_edges": pair_y_edges,
        "pair_x_centers": pair_x_centers,
        "pair_y_centers": pair_y_centers,
        "pair_dxdy": pair_dxdy,
    }


def normalize_hist_channels(hist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize [B,15,H,W] so every pairwise channel sums to one."""
    return hist / (hist.sum(dim=(-1, -2), keepdim=True) + eps)


def cloud_to_pairwise_hists(
    cloud: torch.Tensor,
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
    return_edges: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Convert a 6D cloud [B,Np,6] into 15 pairwise normalized 2D histograms.

    The physical range for each coordinate is sample-adaptive:
        centroid_i +/- clip_k * sigma_i.

    The output histograms are probability masses, not physical densities.
    Each of the 15 channels sums to one.
    """
    B, _, D = cloud.shape
    if D != 6:
        raise ValueError(f"Expected last dimension 6, got {D}")
    centroid, sigma = beam_centroid_sigma(cloud, eps=eps)
    edge_info = pairwise_edges_from_stats(sigma, centroid, nbins=nbins, clip_k=clip_k, eps=eps)

    out = torch.empty(B, len(PAIR_INDEX), nbins, nbins, device=cloud.device, dtype=torch.float32)
    for p, (i, j) in enumerate(PAIR_INDEX):
        x_edges = edge_info["pair_x_edges"][:, p, :]
        y_edges = edge_info["pair_y_edges"][:, p, :]
        for b in range(B):
            out[b, p] = _hist2d_one_sample(
                cloud[b, :, i].to(torch.float32),
                cloud[b, :, j].to(torch.float32),
                x_edges[b].to(torch.float32),
                y_edges[b].to(torch.float32),
            )
    out = normalize_hist_channels(out, eps=eps)
    if return_edges:
        return out, edge_info
    return out


def mass_to_physical_density(
    hist_mass: torch.Tensor,
    sigma: torch.Tensor,
    centroid: torch.Tensor,
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convert normalized per-bin probability mass to physical density.

    density_{ij}[a,b] = mass_{ij}[a,b] / (Delta xi_i * Delta xi_j).
    Units are inverse product of the two coordinate units.
    """
    hist_mass = normalize_hist_channels(hist_mass, eps=eps)
    edge_info = pairwise_edges_from_stats(sigma, centroid, nbins=nbins, clip_k=clip_k, eps=eps)
    density = hist_mass / (edge_info["pair_dxdy"].to(hist_mass.dtype) + eps)
    return density, edge_info


# -----------------------------------------------------------------------------
# Dataset and lattice helpers
# -----------------------------------------------------------------------------

class CloudDataset(Dataset):
    """
    Dataset wrapper for npz files with X_cloud, Y_cloud, MU.

    If keys train/val/test exist, they are used as index arrays. Otherwise all samples are used.
    """
    def __init__(self, data: Dict[str, np.ndarray], split: str = "train"):
        self.X = data["X_cloud"]
        self.Y = data["Y_cloud"]
        self.MU = data["MU"]
        if split in data:
            self.idx = np.asarray(data[split], dtype=np.int64)
        else:
            self.idx = np.arange(self.X.shape[0], dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.idx))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.idx[i])
        X = torch.as_tensor(self.X[j], dtype=torch.float32)
        Y = torch.as_tensor(self.Y[j], dtype=torch.float32)
        MU = torch.as_tensor(self.MU[j], dtype=torch.float32)
        return X, Y, MU


def make_loaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(CloudDataset(data, "train"), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(CloudDataset(data, "val"), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(CloudDataset(data, "test"), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def make_dummy_lattice_tokens(
    B: int,
    N_elem: int,
    device: Union[str, torch.device],
    length: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple placeholder lattice tokens [L,K1,K2,phi,Vrf,frf,phirf]."""
    elem_params = torch.zeros(B, N_elem, 7, device=device, dtype=torch.float32)
    elem_s = torch.linspace(0.0, length, N_elem, device=device, dtype=torch.float32)[None, :].repeat(B, 1)
    elem_params[..., 0] = length / max(N_elem, 1)
    return elem_params, elem_s


def make_fodo_lattice_tokens(
    B: int,
    N_elem: int,
    device: Union[str, torch.device],
    length: float = 4.0,
    kf: float = 1.0,
    kd: float = -2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Small FODO-like token generator compatible with the ElementTokenizer.

    The 7 parameters are [L,K1,K2,phi,Vrf,frf,phirf]. Only L and K1 are populated.
    """
    elem_params, elem_s = make_dummy_lattice_tokens(B, N_elem, device, length=length)
    pattern = torch.tensor([kf, kd, kf, 0.0], device=device, dtype=torch.float32)
    elem_params[..., 1] = pattern[torch.arange(N_elem, device=device) % len(pattern)][None, :]
    return elem_params, elem_s


# -----------------------------------------------------------------------------
# Machine-parameter preprocessing
# -----------------------------------------------------------------------------

MUOp = Literal["identity", "scale", "log", "log10", "log_log", "signed_log", "signed_log_log"]
SpecItem = Union[float, int, str, dict]


def _apply_mu_op(x: torch.Tensor, op: str, eps: float = 1e-12) -> torch.Tensor:
    op = op.lower()
    if op in ("identity", "none", "", "scale"):
        return x
    if op in ("log", "log1p"):
        return torch.log1p(torch.clamp(x, min=0.0))
    if op in ("log10", "log10_1p"):
        return torch.log10(1.0 + torch.clamp(x, min=0.0))
    if op == "log_log":
        return torch.log1p(torch.log1p(torch.clamp(x, min=0.0)))
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
    """
    Notebook-friendly machine-parameter preprocessor.

    Spec rules per dimension:
      - number: unit divisor, x <- x / unit
      - string: operation, e.g. "log", "signed_log"
      - dict: {"unit": ..., "op": ..., "clip": ...}
    """
    def __init__(self, cfg: MUPreprocessConfig):
        super().__init__()
        self.cfg = cfg
        self.m = len(cfg.specs)
        self.register_buffer("mean", torch.zeros(self.m, dtype=torch.float32))
        self.register_buffer("std", torch.ones(self.m, dtype=torch.float32))
        self.fitted = False

    def _parse_spec(self, spec: SpecItem) -> Tuple[float, str, Optional[float]]:
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
            raise TypeError(f"Unsupported MU spec type: {type(spec)}")
        if unit == 0.0:
            unit = 1.0
        return unit, op, clip

    def transform(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.ndim != 2 or mu.shape[1] != self.m:
            raise ValueError(f"mu must have shape [B,{self.m}], got {tuple(mu.shape)}")
        cols = []
        for j, spec in enumerate(self.cfg.specs):
            unit, op, clip = self._parse_spec(spec)
            x = mu[:, j] / unit
            x = _apply_mu_op(x, op, eps=self.cfg.eps)
            if clip is not None:
                x = torch.clamp(x, -clip, clip)
            cols.append(x)
        return torch.stack(cols, dim=1)

    @torch.no_grad()
    def fit(self, mu_train: Union[np.ndarray, torch.Tensor]) -> "MUPreprocessor":
        x = torch.as_tensor(mu_train, dtype=torch.float32, device=self.mean.device)
        xt = self.transform(x)
        self.mean.copy_(xt.mean(dim=0))
        std = xt.std(dim=0, unbiased=False)
        self.std.copy_(torch.where(std < self.cfg.eps, torch.ones_like(std), std))
        self.fitted = True
        return self

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        xt = self.transform(mu)
        if self.cfg.standardize:
            if not self.fitted:
                raise RuntimeError("MUPreprocessor.standardize=True but fit() was not called.")
            xt = (xt - self.mean) / (self.std + self.cfg.eps)
        return xt


# -----------------------------------------------------------------------------
# Element tokenization
# -----------------------------------------------------------------------------

@dataclass
class ElementNormScales:
    L: float = 1.0
    K1: float = 10.0
    K2: float = 10.0
    phi: float = 2.0 * np.pi
    Vrf: float = 1.0
    frf: float = 1.0
    phirf: float = 2.0 * np.pi


def normalize_elem_params(p: torch.Tensor, scales: ElementNormScales) -> torch.Tensor:
    """Normalize element parameters [L,K1,K2,phi,Vrf,frf,phirf]."""
    if p.shape[-1] != 7:
        raise ValueError(f"element parameter tensor must end in 7, got {p.shape[-1]}")
    L, K1, K2, phi, Vrf, frf, phirf = torch.unbind(p, dim=-1)
    return torch.stack([
        L / scales.L,
        K1 / scales.K1,
        K2 / scales.K2,
        phi / scales.phi,
        torch.log1p(torch.clamp(Vrf / scales.Vrf, min=0.0)),
        torch.log1p(torch.clamp(frf / scales.frf, min=0.0)),
        phirf / scales.phirf,
    ], dim=-1)


class FourierPositionalEncoding(nn.Module):
    """Fourier features for longitudinal element position s."""
    def __init__(
        self,
        n_freq_pairs: int = 32,
        wl_min: float = 1e-2,
        wl_max: float = 1e3,
        frequency_scale: float = 299_792_458.0,
    ):
        super().__init__()
        wls = torch.logspace(np.log10(wl_min), np.log10(wl_max), steps=n_freq_pairs)
        freqs = frequency_scale * (2.0 * np.pi / wls)
        self.register_buffer("freqs", freqs.float())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s[..., None].float() * self.freqs[None, None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ElementTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        param_dim: int = 7,
        pos_dim: int = 64,
        frequency_scale: float = 299_792_458.0,
    ):
        super().__init__()
        self.pos = FourierPositionalEncoding(n_freq_pairs=pos_dim // 2, frequency_scale=frequency_scale)
        self.pos_proj = nn.Linear(pos_dim, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, elem_params_norm: torch.Tensor, elem_s: torch.Tensor) -> torch.Tensor:
        if elem_s.ndim == 3:
            elem_s = elem_s.squeeze(-1)
        return self.mlp(elem_params_norm) + self.pos_proj(self.pos(elem_s))


# -----------------------------------------------------------------------------
# Conditional VAE
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        groups = min(8, c_out)
        self.down = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, c_out),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        return self.act(self.conv(h) + h)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        groups = min(8, c_out)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, c_out),
        )
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return self.act(self.conv(x) + self.skip(x))


class ConditionalBeamVAE(nn.Module):
    """Conditional VAE for 15 pairwise histograms plus 12 physical descriptors."""
    def __init__(
        self,
        in_ch: int = 15,
        latent_dim: int = 512,
        aux_dim: int = 12,
        domain_dim: int = 2,
        base_ch: int = 32,
        nbins: int = 64,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.domain_dim = domain_dim
        self.base_ch = base_ch
        self.nbins = nbins

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        spatial_after = nbins // 16
        if spatial_after < 1:
            raise ValueError("nbins must be at least 16")
        self.enc_feat_dim = (base_ch * 8) * spatial_after * spatial_after

        self.fc = nn.Linear(self.enc_feat_dim + aux_dim + domain_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.dec_fc = nn.Linear(latent_dim + domain_dim, self.enc_feat_dim)
        self.dec1 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec3 = UpBlock(base_ch * 2, base_ch)
        self.dec4 = UpBlock(base_ch, base_ch)
        self.dec_out = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.aux_logsigma_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 6))
        self.aux_centroid_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 6))

    def encode(self, x_hist: torch.Tensor, aux: torch.Tensor, domain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc4(self.enc3(self.enc2(self.enc1(x_hist)))).flatten(1)
        h = torch.cat([h, aux, domain], dim=1)
        h = F.gelu(self.fc(h))
        return self.fc_mu(h), self.fc_logvar(h), h

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, domain: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        z_cond = torch.cat([z, domain], dim=1)
        side = self.nbins // 16
        h = self.dec_fc(z_cond).view(z.shape[0], self.base_ch * 8, side, side)
        h = self.dec4(self.dec3(self.dec2(self.dec1(h))))
        mass = F.relu(self.dec_out(h))
        zero = mass.sum(dim=(-1, -2), keepdim=True) < 1e-8
        uniform = torch.ones_like(mass) / (mass.shape[-1] * mass.shape[-2])
        return torch.where(zero, uniform, normalize_hist_channels(mass, eps=eps))

    def forward(
        self,
        x_hist: torch.Tensor,
        aux: torch.Tensor,
        domain: torch.Tensor,
        *,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar, h = self.encode(x_hist, aux, domain)
        z = self.reparam(mu, logvar, sample=sample)
        x_hat = self.decode(z, domain)
        log_sigma_hat = self.aux_logsigma_head(h)
        centroid_hat = self.aux_centroid_head(h)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "encoder_features": h,
            "x_hat": x_hat,
            "log_sigma_hat": log_sigma_hat,
            "sigma_hat": torch.exp(log_sigma_hat),
            "centroid_hat": centroid_hat,
        }


# -----------------------------------------------------------------------------
# Graph Neural Operator latent tracker
# -----------------------------------------------------------------------------

class GNOLayer(nn.Module):
    """Causal graph-kernel integral layer over lattice nodes."""
    def __init__(self, d_model: int, edge_dim: int = 32, kernel_width: int = 128):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.kernel_mlp = nn.Sequential(
            nn.Linear(edge_dim, kernel_width),
            nn.GELU(),
            nn.Linear(kernel_width, d_model),
        )
        self.post = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v: torch.Tensor, edge_feats: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        kappa = self.kernel_mlp(edge_feats) * causal_mask.unsqueeze(-1)
        integral = torch.einsum("bije,bje->bie", kappa, v)
        denom = causal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        update = self.W(v) + self.post(integral / denom)
        return self.norm(v + F.gelu(update))


class GraphNeuralOperatorTracker(nn.Module):
    """Map initial latent state and lattice tokens to a latent trajectory."""
    def __init__(self, latent_dim: int = 512, d_model: int = 512, n_layers: int = 4, edge_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.edge_encoder = nn.Linear(3, edge_dim)
        self.node_lifter = nn.Linear(latent_dim + d_model, d_model)
        self.gno_layers = nn.ModuleList([GNOLayer(d_model, edge_dim=edge_dim) for _ in range(n_layers)])
        self.projection_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, z0: torch.Tensor, elem_tokens: torch.Tensor, elem_s: torch.Tensor) -> torch.Tensor:
        B, N, _ = elem_tokens.shape
        if elem_s.ndim == 2:
            elem_s = elem_s.unsqueeze(-1)
        s_i = elem_s.unsqueeze(2)
        s_j = elem_s.unsqueeze(1)
        dist = s_i - s_j
        raw_edges = torch.cat([s_i.expand(B, N, N, -1), s_j.expand(B, N, N, -1), dist], dim=-1)
        edge_feats = F.gelu(self.edge_encoder(raw_edges.float()))
        causal_mask = (dist.squeeze(-1) >= 0).float()
        z0_expanded = z0.unsqueeze(1).expand(B, N, -1)
        v = self.node_lifter(torch.cat([z0_expanded, elem_tokens], dim=-1))
        for layer in self.gno_layers:
            v = layer(v, edge_feats, causal_mask)
        dz_traj = self.projection_head(v)
        return z0_expanded + dz_traj


# -----------------------------------------------------------------------------
# Distribution model: CVAE + GNO
# -----------------------------------------------------------------------------

class CVAEGNODistributionModel(nn.Module):
    """
    Main distribution-transformation branch.

    It predicts the final normalized pairwise histogram mass tensor, but does not by itself
    decide the physical bin ranges. Physical units are attached by GlobalPhysicalBeamModel
    using the separate moment/statistics branch.
    """
    def __init__(
        self,
        mu_dim: int,
        latent_dim: int = 512,
        d_model: int = 512,
        domain_dim: int = 2,
        nbins: int = 64,
        base_ch: int = 32,
        gno_layers: int = 4,
        frequency_scale: float = 299_792_458.0,
    ):
        super().__init__()
        self.mu_dim = mu_dim
        self.latent_dim = latent_dim
        self.domain_dim = domain_dim
        self.nbins = nbins
        self.vae = ConditionalBeamVAE(
            in_ch=15,
            latent_dim=latent_dim,
            aux_dim=12,
            domain_dim=domain_dim,
            base_ch=base_ch,
            nbins=nbins,
        )
        self.elem_scales = ElementNormScales()
        self.tokenizer = ElementTokenizer(d_model=d_model, param_dim=7, pos_dim=64, frequency_scale=frequency_scale)
        self.tracker = GraphNeuralOperatorTracker(latent_dim=latent_dim, d_model=d_model, n_layers=gno_layers)
        self.mu_to_z = nn.Sequential(nn.Linear(mu_dim, 256), nn.GELU(), nn.Linear(256, latent_dim))

    def encode_initial(
        self,
        X_cloud: torch.Tensor,
        MU_enc: torch.Tensor,
        domain: torch.Tensor,
        *,
        sample_vae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x_hist = cloud_to_pairwise_hists(X_cloud, nbins=self.nbins)
        aux, sigma0, centroid0 = cloud_to_aux(X_cloud)
        vae_out = self.vae(x_hist, aux, domain, sample=sample_vae)
        z0 = vae_out["z"] + self.mu_to_z(MU_enc.to(vae_out["z"].dtype))
        return {
            "x_hist": x_hist,
            "aux0": aux,
            "sigma0": sigma0,
            "centroid0": centroid0,
            "vae": vae_out,
            "z0": z0,
        }

    def forward(
        self,
        X_cloud: torch.Tensor,
        MU_enc: torch.Tensor,
        elem_params: torch.Tensor,
        elem_s: torch.Tensor,
        domain: torch.Tensor,
        *,
        sample_vae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        init = self.encode_initial(X_cloud, MU_enc, domain, sample_vae=sample_vae)
        elem_norm = normalize_elem_params(elem_params, self.elem_scales)
        elem_tokens = self.tokenizer(elem_norm, elem_s)
        z_traj = self.tracker(init["z0"], elem_tokens, elem_s)
        zN = z_traj[:, -1, :]
        histN_mass = self.vae.decode(zN, domain)
        return {
            **init,
            "elem_tokens": elem_tokens,
            "z_pred_traj": z_traj,
            "zN": zN,
            "histN_mass": histN_mass,
            "xN_hat": histN_mass,  # backward-compatible key
        }


# Backward-compatible alias for your notebook naming.
BeamLatentGNOModel = CVAEGNODistributionModel
BeamLatentTrackingModel_GNO = CVAEGNODistributionModel


# -----------------------------------------------------------------------------
# Physical statistics head
# -----------------------------------------------------------------------------

@dataclass
class MomentHeadConfig:
    mu_dim: int
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.0
    use_residual: bool = True
    include_latent: bool = False
    latent_dim: int = 512
    eps: float = 1e-12


class MomentTransportHead(nn.Module):
    """
    Predict final physical moments from initial moments and machine parameters.

    Input semantics:
        sigma0:    [B,6] physical RMS values
        centroid0: [B,6] physical coordinate means
        MU_enc:    [B,mu_dim] preprocessed machine/collective parameters
        z_context: optional [B,latent_dim] final latent state if include_latent=True

    Output semantics:
        log_sigmaN_hat, sigmaN_hat, centroidN_hat.
    """
    def __init__(self, cfg: MomentHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 12 + cfg.mu_dim + (cfg.latent_dim if cfg.include_latent else 0)
        layers: List[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden_dim), nn.GELU()]
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        for _ in range(max(0, cfg.n_layers - 1)):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(cfg.hidden_dim, 12)

    def forward(
        self,
        sigma0: torch.Tensor,
        centroid0: torch.Tensor,
        MU_enc: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        sigma0_safe = torch.clamp(sigma0, min=self.cfg.eps)

        log_sigma0 = torch.log(sigma0_safe)

        # Important change:
        # Instead of feeding raw centroids, feed centroids in beam-size units.
        centroid0_scaled = centroid0 / sigma0_safe

        features = [log_sigma0, centroid0_scaled, MU_enc]

        if self.cfg.include_latent:
            if z_context is None:
                raise ValueError(
                    "MomentTransportHead was configured with include_latent=True, "
                    "but z_context=None."
                )
            features.append(z_context)

        x = torch.cat(features, dim=-1)
        raw = self.out(self.backbone(x))

        d_log_sigma = raw[:, :6]

        # Important change:
        # This is no longer a raw physical displacement.
        # It is a dimensionless displacement in units of sigma0.
        d_centroid_scaled = raw[:, 6:]

        if self.cfg.use_residual:
            log_sigmaN = log_sigma0 + d_log_sigma
            centroidN = centroid0 + d_centroid_scaled * sigma0_safe
        else:
            log_sigmaN = d_log_sigma
            centroidN = d_centroid_scaled * sigma0_safe

        sigmaN = torch.exp(log_sigmaN).clamp_min(self.cfg.eps)

        return {
            "log_sigmaN_hat": log_sigmaN,
            "sigmaN_hat": sigmaN,
            "centroidN_hat": centroidN,

            "delta_log_sigma_hat": d_log_sigma,

            # This is now dimensionless.
            "delta_centroid_scaled_hat": d_centroid_scaled,

            # Keep old key for compatibility, but now it stores the physical displacement.
            "delta_centroid_hat": centroidN - centroid0,
        }


# -----------------------------------------------------------------------------
# Global physical model: distribution branch + statistics branch
# -----------------------------------------------------------------------------

class GlobalPhysicalBeamModel(nn.Module):
    """
    Combined model for physical final distributions.

    It separates shape from units:
        - CVAE+GNO predicts normalized pairwise histogram shape.
        - MomentTransportHead predicts final sigma/centroid.
        - The final physical density is built by placing the decoded mass on the
          moment-defined physical grid.
    """
    def __init__(
        self,
        mu_dim: int,
        latent_dim: int = 512,
        d_model: int = 512,
        domain_dim: int = 2,
        nbins: int = 64,
        base_ch: int = 32,
        gno_layers: int = 4,
        clip_k: float = 5.0,
        stats_include_latent: bool = True,
        frequency_scale: float = 299_792_458.0,
    ):
        super().__init__()
        self.nbins = nbins
        self.clip_k = clip_k
        self.distribution = CVAEGNODistributionModel(
            mu_dim=mu_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            domain_dim=domain_dim,
            nbins=nbins,
            base_ch=base_ch,
            gno_layers=gno_layers,
            frequency_scale=frequency_scale,
        )
        self.stats_head = MomentTransportHead(
            MomentHeadConfig(
                mu_dim=mu_dim,
                hidden_dim=256,
                n_layers=3,
                use_residual=True,
                include_latent=stats_include_latent,
                latent_dim=latent_dim,
            )
        )

    @property
    def vae(self) -> ConditionalBeamVAE:
        return self.distribution.vae

    @property
    def tracker(self) -> GraphNeuralOperatorTracker:
        return self.distribution.tracker

    @property
    def tokenizer(self) -> ElementTokenizer:
        return self.distribution.tokenizer

    @property
    def elem_scales(self) -> ElementNormScales:
        return self.distribution.elem_scales

    def forward(
        self,
        X_cloud: torch.Tensor,
        MU_enc: torch.Tensor,
        elem_params: torch.Tensor,
        elem_s: torch.Tensor,
        domain: torch.Tensor,
        *,
        sample_vae: bool = True,
        return_physical_density: bool = True,
    ) -> Dict[str, torch.Tensor]:
        dist_out = self.distribution(
            X_cloud,
            MU_enc,
            elem_params,
            elem_s,
            domain,
            sample_vae=sample_vae,
        )
        stats_out = self.stats_head(
            dist_out["sigma0"],
            dist_out["centroid0"],
            MU_enc,
            z_context=dist_out["zN"] if self.stats_head.cfg.include_latent else None,
        )
        hist_mass = normalize_hist_channels(dist_out["histN_mass"])
        out = {**dist_out, **stats_out, "histN_mass": hist_mass, "xN_hat": hist_mass}
        if return_physical_density:
            hist_density, edge_info = mass_to_physical_density(
                hist_mass,
                stats_out["sigmaN_hat"],
                stats_out["centroidN_hat"],
                nbins=self.nbins,
                clip_k=self.clip_k,
            )
            out.update(edge_info)
            out["histN_density"] = hist_density
        return out

    def freeze_vae(self, freeze: bool = True) -> None:
        set_requires_grad(self.distribution.vae, not freeze)

    def freeze_distribution_branch(self, freeze: bool = True) -> None:
        set_requires_grad(self.distribution, not freeze)

    def train_only_stats_head(self) -> None:
        self.freeze_distribution_branch(True)
        set_requires_grad(self.stats_head, True)

    def train_distribution_and_stats(self) -> None:
        set_requires_grad(self, True)


# -----------------------------------------------------------------------------
# Losses and training helpers
# -----------------------------------------------------------------------------

@dataclass
class GlobalLossWeights:
    hist: float = 1.0
    log_sigma: float = 1.0
    centroid: float = 20.0
    dcentroid: float = 10.0
    kl: float = 1e-2
    density: float = 0.0

def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.square() - 1.0 - logvar, dim=1)


def pairwise_hist_loss(pred_mass: torch.Tensor, target_mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_mass = normalize_hist_channels(pred_mass, eps=eps)
    target_mass = normalize_hist_channels(target_mass, eps=eps)
    return F.mse_loss(pred_mass, target_mass, reduction="none").sum(dim=(-1, -2)).mean()


def moment_target_from_cloud(Y_cloud: torch.Tensor, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    centroidY, sigmaY = beam_centroid_sigma(Y_cloud, eps=eps)
    return {
        "sigmaY": sigmaY,
        "log_sigmaY": torch.log(torch.clamp(sigmaY, min=eps)),
        "centroidY": centroidY,
    }

def compute_global_loss(
    out: Dict[str, torch.Tensor],
    Y_cloud: torch.Tensor,
    weights: GlobalLossWeights = GlobalLossWeights(),
    nbins: int = 64,
    clip_k: float = 5.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Stable loss for physical statistics.

    Main idea:
        Do NOT train on raw sigma_N and raw centroid_N.

        Train on:
            delta_log_sigma = log(sigma_N) - log(sigma_0)
            delta_centroid_scaled = (centroid_N - centroid_0) / sigma_0

    This removes most order-of-magnitude problems.
    """

    # --------------------------------------------------------
    # Histogram target
    # --------------------------------------------------------
    target_mass = cloud_to_pairwise_hists(
        Y_cloud,
        nbins=nbins,
        clip_k=clip_k,
        eps=eps,
    )

    hist = pairwise_hist_loss(
        out["histN_mass"],
        target_mass,
        eps=eps,
    )

    # --------------------------------------------------------
    # Initial and final physical moments
    # --------------------------------------------------------
    centroidY, sigmaY = beam_centroid_sigma(
        Y_cloud,
        eps=eps,
    )

    sigmaY = torch.clamp(
        sigmaY,
        min=eps,
    )

    sigma0 = torch.clamp(
        out["sigma0"],
        min=eps,
    )

    centroid0 = out["centroid0"]

    log_sigma0 = torch.log(
        sigma0,
    )

    log_sigmaY = torch.log(
        sigmaY,
    )

    # --------------------------------------------------------
    # Stable normalized targets
    # --------------------------------------------------------
    target_delta_log_sigma = (
        log_sigmaY
        -
        log_sigma0
    )

    target_delta_centroid_scaled = (
        centroidY
        -
        centroid0
    ) / sigma0.detach()

    # --------------------------------------------------------
    # Stable predicted normalized quantities
    # --------------------------------------------------------
    if "delta_log_sigma_hat" in out:
        pred_delta_log_sigma = out["delta_log_sigma_hat"]
    else:
        pred_delta_log_sigma = (
            out["log_sigmaN_hat"]
            -
            log_sigma0
        )

    if "delta_centroid_scaled_hat" in out:
        pred_delta_centroid_scaled = out["delta_centroid_scaled_hat"]
    else:
        pred_delta_centroid_scaled = (
            out["centroidN_hat"]
            -
            centroid0
        ) / sigma0.detach()

    # --------------------------------------------------------
    # Optional signed-log-like compression for heavy tails
    #
    # asinh behaves like:
    #   x near 0      -> x
    #   large |x|     -> sign(x) log(2|x|)
    #
    # This is better than mantissa/exponent for signed quantities.
    # --------------------------------------------------------
    USE_ASINH_FOR_CENTROID = True
    CENTROID_TAU = 0.25

    if USE_ASINH_FOR_CENTROID:
        pred_centroid_loss_var = torch.asinh(
            pred_delta_centroid_scaled / CENTROID_TAU
        )

        target_centroid_loss_var = torch.asinh(
            target_delta_centroid_scaled / CENTROID_TAU
        )
    else:
        pred_centroid_loss_var = pred_delta_centroid_scaled
        target_centroid_loss_var = target_delta_centroid_scaled

    # --------------------------------------------------------
    # Main stable losses
    # --------------------------------------------------------
    delta_log_sigma_loss = F.smooth_l1_loss(
        pred_delta_log_sigma,
        target_delta_log_sigma,
        beta=0.25,
    )

    delta_centroid_scaled_loss = F.smooth_l1_loss(
        pred_centroid_loss_var,
        target_centroid_loss_var,
        beta=0.25,
    )

    # --------------------------------------------------------
    # Auxiliary final-value losses
    # Keep these small. They anchor the physical reconstruction.
    # --------------------------------------------------------
    final_log_sigma_loss = F.smooth_l1_loss(
        out["log_sigmaN_hat"],
        log_sigmaY,
        beta=0.25,
    )

    final_centroid_residual = (
        out["centroidN_hat"]
        -
        centroidY
    ) / sigmaY.detach()

    final_centroid_loss = F.smooth_l1_loss(
        final_centroid_residual,
        torch.zeros_like(final_centroid_residual),
        beta=0.25,
    )

    # --------------------------------------------------------
    # Optional KL
    # --------------------------------------------------------
    if "vae" in out and "mu" in out["vae"] and "logvar" in out["vae"]:
        kl = kl_divergence_standard_normal(
            out["vae"]["mu"],
            out["vae"]["logvar"],
        ).mean()
    else:
        kl = torch.zeros(
            (),
            device=Y_cloud.device,
        )

    # --------------------------------------------------------
    # Optional physical density loss
    # Keep off while stabilizing physical moments.
    # --------------------------------------------------------
    density_loss = torch.zeros(
        (),
        device=Y_cloud.device,
    )

    if weights.density != 0.0 and "histN_density" in out:
        target_density, _ = mass_to_physical_density(
            target_mass,
            sigmaY,
            centroidY,
            nbins=nbins,
            clip_k=clip_k,
            eps=eps,
        )

        density_loss = F.smooth_l1_loss(
            out["histN_density"],
            target_density,
            beta=0.25,
        )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------
    hist_w = getattr(weights, "hist", 0.0)
    kl_w = getattr(weights, "kl", 0.0)
    density_w = getattr(weights, "density", 0.0)

    # New stable weights
    delta_log_sigma_w = getattr(weights, "delta_log_sigma", 1.0)
    delta_centroid_scaled_w = getattr(weights, "delta_centroid_scaled", 10.0)

    # Small auxiliary physical anchors
    final_log_sigma_w = getattr(weights, "log_sigma", 0.25)
    final_centroid_w = getattr(weights, "centroid", 1.0)

    loss = (
        hist_w * hist
        + delta_log_sigma_w * delta_log_sigma_loss
        + delta_centroid_scaled_w * delta_centroid_scaled_loss
        + final_log_sigma_w * final_log_sigma_loss
        + final_centroid_w * final_centroid_loss
        + kl_w * kl
        + density_w * density_loss
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    sigma_rel = torch.linalg.norm(
        out["sigmaN_hat"] - sigmaY,
        dim=1,
    ) / (
        torch.linalg.norm(sigmaY, dim=1)
        + eps
    )

    delta_log_sigma_rms = torch.sqrt(
        torch.mean(
            (
                pred_delta_log_sigma
                -
                target_delta_log_sigma
            ) ** 2,
            dim=1,
        )
    )

    delta_centroid_scaled_rms = torch.sqrt(
        torch.mean(
            (
                pred_delta_centroid_scaled
                -
                target_delta_centroid_scaled
            ) ** 2,
            dim=1,
        )
    )

    centroid_rms_sigma = torch.sqrt(
        torch.mean(
            final_centroid_residual ** 2,
            dim=1,
        )
    )

    centroid_abs = torch.mean(
        torch.abs(
            out["centroidN_hat"]
            -
            centroidY
        )
    )

    return {
        "loss": loss,

        "hist_loss": hist,

        "delta_log_sigma_loss": delta_log_sigma_loss,
        "delta_centroid_scaled_loss": delta_centroid_scaled_loss,

        "log_sigma_loss": final_log_sigma_loss,
        "centroid_loss": final_centroid_loss,

        "kl": kl,
        "density_loss": density_loss,

        "sigma_rel": sigma_rel.mean(),
        "delta_log_sigma_rms": delta_log_sigma_rms.mean(),
        "delta_centroid_scaled_rms": delta_centroid_scaled_rms.mean(),
        "centroid_rms_sigma": centroid_rms_sigma.mean(),
        "centroid_abs": centroid_abs,
    }


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_module(module: nn.Module) -> None:
    set_requires_grad(module, False)


def unfreeze_module(module: nn.Module) -> None:
    set_requires_grad(module, True)


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


def load_state_dict_flexible(
    model: nn.Module,
    ckpt_path: str,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load a checkpoint that may be either a raw state_dict or a dict containing state_dict/model.
    Returns (missing_keys, unexpected_keys).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
    result = model.load_state_dict(ckpt, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def _default_domain(MU_raw: torch.Tensor, domain_dim: int = 2) -> torch.Tensor:
    if MU_raw.shape[1] >= domain_dim:
        return MU_raw[:, :domain_dim]
    return torch.zeros(MU_raw.shape[0], domain_dim, device=MU_raw.device, dtype=MU_raw.dtype)


def train_one_epoch(
    model: GlobalPhysicalBeamModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mu_prep: Optional[MUPreprocessor],
    N_elem: int,
    device: Union[str, torch.device],
    *,
    weights: GlobalLossWeights = GlobalLossWeights(),
    lattice: str = "fodo",
    sample_vae: bool = True,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0
    for X, Y, MU in loader:
        X, Y, MU = X.to(device), Y.to(device), MU.to(device)
        MU_enc = mu_prep(MU) if mu_prep is not None else MU
        domain = _default_domain(MU, model.distribution.domain_dim)
        elem_params, elem_s = make_fodo_lattice_tokens(X.shape[0], N_elem, device) if lattice == "fodo" else make_dummy_lattice_tokens(X.shape[0], N_elem, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(X, MU_enc, elem_params, elem_s, domain, sample_vae=sample_vae)
        ld = compute_global_loss(out, Y, weights=weights, nbins=model.nbins, clip_k=model.clip_k)
        ld["loss"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), grad_clip)
        optimizer.step()

        for k, v in ld.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: GlobalPhysicalBeamModel,
    loader: DataLoader,
    mu_prep: Optional[MUPreprocessor],
    N_elem: int,
    device: Union[str, torch.device],
    *,
    weights: GlobalLossWeights = GlobalLossWeights(),
    lattice: str = "fodo",
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n_batches = 0
    for X, Y, MU in loader:
        X, Y, MU = X.to(device), Y.to(device), MU.to(device)
        MU_enc = mu_prep(MU) if mu_prep is not None else MU
        domain = _default_domain(MU, model.distribution.domain_dim)
        elem_params, elem_s = make_fodo_lattice_tokens(X.shape[0], N_elem, device) if lattice == "fodo" else make_dummy_lattice_tokens(X.shape[0], N_elem, device)
        out = model(X, MU_enc, elem_params, elem_s, domain, sample_vae=False)
        ld = compute_global_loss(out, Y, weights=weights, nbins=model.nbins, clip_k=model.clip_k)
        for k, v in ld.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


__all__ = [
    "COORD_NAMES", "PAIR_INDEX", "PAIR_NAMES",
    "get_device", "load_npz", "beam_centroid_sigma", "cloud_to_aux", "cloud_to_pairwise_hists",
    "edges_from_sigma_centroid", "pairwise_edges_from_stats", "mass_to_physical_density",
    "CloudDataset", "make_loaders", "make_dummy_lattice_tokens", "make_fodo_lattice_tokens",
    "MUPreprocessConfig", "MUPreprocessor", "ElementNormScales", "normalize_elem_params", "ElementTokenizer",
    "ConditionalBeamVAE", "GNOLayer", "GraphNeuralOperatorTracker", "CVAEGNODistributionModel",
    "BeamLatentGNOModel", "BeamLatentTrackingModel_GNO",
    "MomentHeadConfig", "MomentTransportHead", "GlobalPhysicalBeamModel",
    "GlobalLossWeights", "compute_global_loss", "set_requires_grad", "freeze_module", "unfreeze_module",
    "trainable_parameters", "load_state_dict_flexible", "train_one_epoch", "evaluate",
]
