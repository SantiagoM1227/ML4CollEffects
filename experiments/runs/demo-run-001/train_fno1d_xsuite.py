from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    dataset_path: str

    out_dir: str = "./models"
    run_name: str = "fno1d_lambda"

    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-6

    width: int = 64
    modes: int = 16
    depth: int = 4
    hidden_proj: int = 128

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # data
    in_channels: int = 5  # [lambda_in, kf1, kd1, kf2, zeta]


# ----------------------------
# Data
# ----------------------------
class LambdaDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], split: str) -> None:
        self.x = data["X_lambda"][data[split]]          # [Ns, Nz]
        self.y = data["Y_lambda"][data[split]]          # [Ns, Nz]
        self.mu = data["MU"][data[split]]               # [Ns, 3]
        self.zeta_grid = data["zeta_grid"]              # [Nz]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float()          # [Nz]
        y = torch.from_numpy(self.y[idx]).float()          # [Nz]
        mu = torch.from_numpy(self.mu[idx]).float()        # [3]
        grid = torch.from_numpy(self.zeta_grid).float()    # [Nz]

        # features: [Nz, 5]
        features = torch.stack(
            [
                x,
                torch.full_like(x, mu[0]),
                torch.full_like(x, mu[1]),
                torch.full_like(x, mu[2]),
                grid,
            ],
            dim=-1,
        )
        return features, y


def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


# ----------------------------
# FNO model (same as your notebook)
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
        # x: [B, C, N]
        B, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.compl_weight()[:, :, :n_modes],
        )
        x = torch.fft.irfft(out_ft, n=n, dim=-1)
        return x


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
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
        # x: [B, N, C]
        x = self.lift(x)            # [B, N, W]
        x = x.permute(0, 2, 1)      # [B, W, N]
        for block in self.blocks:
            x = block(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x[:, 0, :]           # [B, N]


# ----------------------------
# Metrics
# ----------------------------
def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1) + eps
    return torch.mean(torch.sqrt(num / den))


def line_density_mass(pred: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # pred: [B, Nz], grid: [Nz]
    dz = (grid[-1] - grid[0]) / max(1, grid.numel() - 1)
    return torch.sum(pred, dim=-1) * dz


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    rel_sum = 0.0
    mass_err_sum = 0.0
    count = 0

    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        pred = model(features)

        mse = torch.mean((pred - target) ** 2)
        rel = relative_l2(pred, target)

        grid = features[0, :, -1]  # zeta grid from features (same for whole batch)
        mass_pred = line_density_mass(pred, grid)
        mass_true = line_density_mass(target, grid)
        mass_err = torch.mean(torch.abs(mass_pred - mass_true))

        b = target.size(0)
        mse_sum += mse.item() * b
        rel_sum += rel.item() * b
        mass_err_sum += mass_err.item() * b
        count += b

    return {
        "mse": mse_sum / max(1, count),
        "rel_l2": rel_sum / max(1, count),
        "mass_abs_err": mass_err_sum / max(1, count),
    }


# ----------------------------
# Train
# ----------------------------
def train(cfg: TrainConfig) -> None:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = load_npz(cfg.dataset_path)
    train_ds = LambdaDataset(data, "train")
    val_ds = LambdaDataset(data, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = FNO1d(
        in_channels=cfg.in_channels,
        width=cfg.width,
        modes=cfg.modes,
        depth=cfg.depth,
        hidden_proj=cfg.hidden_proj,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_best.pt")
    last_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_last.pt")

    # save config once
    Path(os.path.join(cfg.out_dir, f"{cfg.run_name}_config.json")).write_text(
        json.dumps(asdict(cfg), indent=2)
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        s = 0.0
        n = 0

        for features, target in train_loader:
            features = features.to(cfg.device)
            target = target.to(cfg.device)

            pred = model(features)
            loss = torch.mean((pred - target) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = target.size(0)
            s += loss.item() * b
            n += b

        train_mse = s / max(1, n)
        val_metrics = evaluate(model, val_loader, cfg.device)

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_mse={train_mse:.3e} "
                f"| val_mse={val_metrics['mse']:.3e} "
                f"val_rel_l2={val_metrics['rel_l2']:.3e} "
                f"val_mass_abs_err={val_metrics['mass_abs_err']:.3e}"
            )

    # save last
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "epoch": cfg.epochs,
        },
        last_path,
    )
    print("[OK] saved:", best_path)
    print("[OK] saved:", last_path)


def main():
    dataset_path = os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("Set DATASET_PATH env var to your dataset .npz path")

    cfg = TrainConfig(dataset_path=dataset_path)
    print("[INFO] device:", cfg.device)
    train(cfg)


if __name__ == "__main__":
    main()