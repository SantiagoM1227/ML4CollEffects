
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    dataset_path: str = "./data/neural/xsuite_neural_dataset.npz"
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


class LambdaDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], split: str) -> None:
        self.x = data["X_lambda"][data[split]]
        self.y = data["Y_lambda"][data[split]]
        self.mu = data["MU"][data[split]]
        self.zeta_grid = data["zeta_grid"]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float()          # [Nz]
        y = torch.from_numpy(self.y[idx]).float()          # [Nz]
        mu = torch.from_numpy(self.mu[idx]).float()        # [3]
        grid = torch.from_numpy(self.zeta_grid).float()    # [Nz]
        # channels: lambda_in, kf1, kd1, kf2, zeta
        features = torch.stack(
            [
                x,
                torch.full_like(x, mu[0]),
                torch.full_like(x, mu[1]),
                torch.full_like(x, mu[2]),
                grid,
            ],
            dim=-1,
        )  # [Nz, 5]
        return features, y


def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

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


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1) + eps
    return torch.mean(torch.sqrt(num / den))


def line_density_mass(pred: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 1:
        raise ValueError("grid must be 1D")
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
        grid = features[0, :, -1]
        mass_pred = line_density_mass(pred, grid)
        mass_true = line_density_mass(target, grid)
        mass_err = torch.mean(torch.abs(mass_pred - mass_true))

        b = features.shape[0]
        mse_sum += mse.item() * b
        rel_sum += rel.item() * b
        mass_err_sum += mass_err.item() * b
        count += b

    return {
        "mse": mse_sum / max(1, count),
        "rel_l2": rel_sum / max(1, count),
        "mass_abs_err": mass_err_sum / max(1, count),
    }


def train(cfg: TrainConfig) -> Tuple[FNO1d, Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = load_npz(cfg.dataset_path)
    train_ds = LambdaDataset(data, "train")
    val_ds = LambdaDataset(data, "val")
    test_ds = LambdaDataset(data, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = FNO1d(
        in_channels=5,
        width=cfg.width,
        modes=cfg.modes,
        depth=cfg.depth,
        hidden_proj=cfg.hidden_proj,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for features, target in train_loader:
            features = features.to(cfg.device)
            target = target.to(cfg.device)

            optimizer.zero_grad()
            pred = model(features)
            loss_fit = torch.mean((pred - target) ** 2)
            loss_rel = relative_l2(pred, target)
            grid = features[0, :, -1]
            mass_pred = line_density_mass(pred, grid)
            mass_true = line_density_mass(target, grid)
            loss_mass = torch.mean((mass_pred - mass_true) ** 2)
            loss = loss_fit + 0.1 * loss_rel + 0.01 * loss_mass
            loss.backward()
            optimizer.step()

            b = features.shape[0]
            epoch_loss += loss.item() * b
            n_seen += b

        scheduler.step()
        train_loss = epoch_loss / max(1, n_seen)
        val_metrics = evaluate(model, val_loader, cfg.device)

        if val_metrics["rel_l2"] < best_val:
            best_val = val_metrics["rel_l2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.6e} | "
                f"val_rel_l2={val_metrics['rel_l2']:.6e} | "
                f"val_mse={val_metrics['mse']:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg.device)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6e}")

    return model, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal 1D FNO on Xsuite density data.")
    parser.add_argument("--dataset-path", type=str, default="./data/neural/xsuite_neural_dataset.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden-proj", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default="")
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
        device=args.device,
        seed=args.seed,
    )
    model, metrics = train(cfg)

    if args.save_model:
        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "metrics": metrics,
            },
            out_path,
        )
        print("Saved model to", out_path)


if __name__ == "__main__":
    main()
