from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    dataset_path: str = "./data/haissinski_forward_dataset.npz"
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    width: int = 64
    modes: int = 16
    depth: int = 4
    hidden_proj: int = 128
    deriv_weight: float = 0.05
    moment_weight: float = 0.05
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_model: str = "./checkpoints/haissinski_fno_best.pt"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def finite_diff_1d(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    x:    [B, N]
    grid: [N]
    returns dx/dq with shape [B, N]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [B, N]")
    if grid.ndim != 1:
        raise ValueError("grid must be 1D")

    dx = torch.zeros_like(x)
    dq = grid[1:] - grid[:-1]

    # centered interior
    dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (grid[2:] - grid[:-2]).unsqueeze(0)
    # one-sided boundaries
    dx[:, 0] = (x[:, 1] - x[:, 0]) / dq[0]
    dx[:, -1] = (x[:, -1] - x[:, -2]) / dq[-1]
    return dx


def trapz_batch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: [B, N]
    x: [N]
    returns [B]
    """
    return torch.trapz(y, x, dim=-1)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1) + eps
    return torch.mean(torch.sqrt(num / den))


def line_moments(lam: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lam: [B, N], assumed nonnegative but not necessarily normalized.
    q:   [N]
    returns centroid and rms with shape [B]
    """
    mass = trapz_batch(lam, q).clamp_min(eps)
    mean = trapz_batch(lam * q.unsqueeze(0), q) / mass
    var = trapz_batch(lam * (q.unsqueeze(0) - mean.unsqueeze(-1)) ** 2, q) / mass
    return mean, torch.sqrt(var.clamp_min(eps))


# ============================================================
# Dataset
# ============================================================

class HaissinskiForwardDataset(Dataset):
    """
    Expected NPZ keys:
      - q_grid:        [Nq]
      - W:             [Ns, Nq]          sampled wake functions on the same grid
      - I:             [Ns] or [Ns, 1]   intensity / normalized current
      - lambda_target: [Ns, Nq]          target stationary line densities

    Optional keys:
      - machine_params: [Ns, P]
      - train_idx, val_idx, test_idx     explicit split indices

    If indices are absent, the loader expects boolean masks named
      train, val, test
    or will create a default 80/10/10 split.
    """

    def __init__(self, data: Dict[str, np.ndarray], split: str):
        self.q_grid = torch.from_numpy(np.asarray(data["q_grid"])).float()
        self.W = np.asarray(data["W"], dtype=np.float32)
        self.I = np.asarray(data["I"], dtype=np.float32)
        self.lam = np.asarray(data["lambda_target"], dtype=np.float32)

        if self.I.ndim == 1:
            self.I = self.I[:, None]

        self.machine_params: Optional[np.ndarray] = None
        if "machine_params" in data:
            self.machine_params = np.asarray(data["machine_params"], dtype=np.float32)

        self.indices = self._get_indices(data, split)

    def _get_indices(self, data: Dict[str, np.ndarray], split: str) -> np.ndarray:
        idx_name = f"{split}_idx"
        if idx_name in data:
            return np.asarray(data[idx_name], dtype=np.int64)

        if split in data:
            mask = np.asarray(data[split])
            if mask.dtype == bool:
                return np.where(mask)[0].astype(np.int64)
            return mask.astype(np.int64)

        # default 80/10/10 split if not present
        n = self.W.shape[0]
        perm = np.arange(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            return perm[:n_train]
        if split == "val":
            return perm[n_train:n_train + n_val]
        if split == "test":
            return perm[n_train + n_val:]
        raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        q = self.q_grid                               # [Nq]
        w = torch.from_numpy(self.W[idx]).float()    # [Nq]
        I = torch.from_numpy(self.I[idx]).float()    # [1] or [P_I]
        y = torch.from_numpy(self.lam[idx]).float()  # [Nq]

        features = [
            w,
            torch.full_like(w, I[0]),
            q,
        ]

        if self.machine_params is not None:
            mu = torch.from_numpy(self.machine_params[idx]).float()
            for j in range(mu.numel()):
                features.append(torch.full_like(w, mu[j]))

        x = torch.stack(features, dim=-1)            # [Nq, C]
        return x, y


# ============================================================
# Model
# ============================================================

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
        # x: [B, C, N]
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.compl_weight()[:, :, :n_modes],
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        y = self.norm(y)
        return self.act(y)


class HaissinskiFNO1d(nn.Module):
    """
    Forward surrogate for the operator
        (W(q), I, q [, machine params]) -> lambda(q; I)

    The output is passed through softplus and then normalized with trapezoidal
    integration so the prediction is always nonnegative and has unit mass.
    """

    def __init__(
        self,
        in_channels: int,
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
        hidden_proj: int = 128,
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, hidden_proj, kernel_size=1)
        self.proj2 = nn.Conv1d(hidden_proj, 1, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], with q-grid in channel index 2 by construction
        q = x[:, :, 2]
        z = self.lift(x)          # [B, N, W]
        z = z.permute(0, 2, 1)    # [B, W, N]
        for block in self.blocks:
            z = block(z)
        z = self.act(self.proj1(z))
        z = self.proj2(z)[:, 0, :]  # [B, N]

        lam = F.softplus(z) + 1e-10
        mass = trapz_batch(lam, q[0]).unsqueeze(-1).clamp_min(1e-12)
        lam = lam / mass
        return lam


# ============================================================
# Losses and evaluation
# ============================================================

def loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    q: torch.Tensor,
    deriv_weight: float,
    moment_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    fit = torch.mean((pred - target) ** 2)
    rel = relative_l2(pred, target)

    d_pred = finite_diff_1d(pred, q)
    d_true = finite_diff_1d(target, q)
    deriv = torch.mean((d_pred - d_true) ** 2)

    m_pred, s_pred = line_moments(pred, q)
    m_true, s_true = line_moments(target, q)
    moment = torch.mean((m_pred - m_true) ** 2 + (s_pred - s_true) ** 2)

    loss = fit + 0.1 * rel + deriv_weight * deriv + moment_weight * moment
    stats = {
        "fit": float(fit.detach().cpu()),
        "rel": float(rel.detach().cpu()),
        "deriv": float(deriv.detach().cpu()),
        "moment": float(moment.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, stats


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, deriv_weight: float, moment_weight: float) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "fit": 0.0, "rel": 0.0, "deriv": 0.0, "moment": 0.0}
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        q = x[0, :, 2]

        pred = model(x)
        _, stats = loss_fn(pred, y, q, deriv_weight, moment_weight)

        b = x.shape[0]
        for k in totals:
            totals[k] += stats[k] * b
        count += b

    for k in totals:
        totals[k] /= max(1, count)
    return totals


# ============================================================
# Training
# ============================================================

def build_dataloaders(cfg: TrainConfig):
    data = load_npz(cfg.dataset_path)
    train_ds = HaissinskiForwardDataset(data, "train")
    val_ds = HaissinskiForwardDataset(data, "val")
    test_ds = HaissinskiForwardDataset(data, "test")

    in_channels = train_ds[0][0].shape[-1]

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, in_channels


def train(cfg: TrainConfig) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, in_channels = build_dataloaders(cfg)

    model = HaissinskiFNO1d(
        in_channels=in_channels,
        width=cfg.width,
        modes=cfg.modes,
        depth=cfg.depth,
        hidden_proj=cfg.hidden_proj,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        verbose=True,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            q = x[0, :, 2]

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = loss_fn(pred, y, q, cfg.deriv_weight, cfg.moment_weight)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            b = x.shape[0]
            running += loss.item() * b
            n_seen += b

        train_loss = running / max(1, n_seen)
        val_metrics = evaluate(model, val_loader, cfg.device, cfg.deriv_weight, cfg.moment_weight)
        scheduler.step(val_metrics["rel"])

        if val_metrics["rel"] < best_val:
            best_val = val_metrics["rel"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:04d} | lr={lr:.3e} | "
                f"train_loss={train_loss:.6e} | "
                f"val_rel={val_metrics['rel']:.6e} | "
                f"val_fit={val_metrics['fit']:.6e} | "
                f"val_deriv={val_metrics['deriv']:.6e} | "
                f"val_moment={val_metrics['moment']:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg.device, cfg.deriv_weight, cfg.moment_weight)
    print("\nBest test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6e}")

    return model, test_metrics


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 1D FNO forward surrogate for Haissinski profile prediction."
    )
    parser.add_argument("--dataset-path", type=str, default="./data/haissinski_forward_dataset.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden-proj", type=int, default=128)
    parser.add_argument("--deriv-weight", type=float, default=0.05)
    parser.add_argument("--moment-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default="./checkpoints/haissinski_fno_best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        modes=args.modes,
        depth=args.depth,
        hidden_proj=args.hidden_proj,
        deriv_weight=args.deriv_weight,
        moment_weight=args.moment_weight,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        save_model=args.save_model,
    )

    model, metrics = train(cfg)

    out_path = Path(cfg.save_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
        out_path,
    )
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
