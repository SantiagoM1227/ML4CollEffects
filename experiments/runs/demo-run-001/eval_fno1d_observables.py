from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
@dataclass
class EvalConfig:
    dataset_path: str
    checkpoint_path: str
    out_dir: str = "./output"

    split: str = "test"           # "train" | "val" | "test"
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FNO architecture (must match training)
    in_channels: int = 5  # [lambda_in, mu1, mu2, mu3, zeta]
    width: int = 128
    modes: int = 32
    depth: int = 5
    hidden_proj: int = 128

    eps: float = 1e-12

# ----------------------------
# Helpers
# ----------------------------

def save_mean_density(z, Yt, Yp, out_path, title="Mean line density"):
    mt = Yt.mean(axis=0)
    mp = Yp.mean(axis=0)
    plt.figure(figsize=(7,4))
    plt.plot(z, mt, label="true mean")
    plt.plot(z, mp, label="pred mean", linestyle="--")
    plt.title(title)
    plt.xlabel("zeta")
    plt.ylabel("lambda(zeta)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_mean_residual(z, R, out_path, title="Mean residual"):
    mr = R.mean(axis=0)
    plt.figure(figsize=(7,4))
    plt.plot(z, mr)
    plt.axhline(0.0, color="k", lw=1)
    plt.title(title)
    plt.xlabel("zeta")
    plt.ylabel("pred - true")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_rmse_vs_z(z, R, out_path, title="RMSE vs zeta"):
    rmse = np.sqrt((R**2).mean(axis=0))
    plt.figure(figsize=(7,4))
    plt.plot(z, rmse)
    plt.title(title)
    plt.xlabel("zeta")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residual_hist(R, out_path, bins=100, title="Residual histogram (all z, all samples)"):
    r = R.ravel()
    plt.figure(figsize=(7,4))
    plt.hist(r, bins=bins, density=True, alpha=0.8)
    plt.axvline(0.0, color="k", lw=1)
    plt.title(title)
    plt.xlabel("pred - true")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residual_heatmap(z, R, out_path, title="Residual heatmap (samples x zeta)"):
    # sort samples by mean_zeta of truth (optional; makes structure visible)
    # here: no sorting, just raw order
    plt.figure(figsize=(9,4))
    plt.imshow(R, aspect="auto", origin="lower",
               extent=[z[0], z[-1], 0, R.shape[0]],
               cmap="RdBu_r")
    plt.colorbar(label="pred - true")
    plt.title(title)
    plt.xlabel("zeta")
    plt.ylabel("sample index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



# ----------------------------
# Dataset (same as your notebook)
# ----------------------------
class LambdaDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], split: str) -> None:
        self.x = data["X_lambda"][data[split]]      # [Ns, Nz]
        self.y = data["Y_lambda"][data[split]]      # [Ns, Nz]
        self.mu = data["MU"][data[split]]           # [Ns, 3]
        self.zeta_grid = data["zeta_grid"]          # [Nz]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float()          # [Nz]
        y = torch.from_numpy(self.y[idx]).float()          # [Nz]
        mu = torch.from_numpy(self.mu[idx]).float()        # [3]
        grid = torch.from_numpy(self.zeta_grid).float()    # [Nz]
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
        return features, y, mu, grid


def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


# ----------------------------
# Model (same as your notebook)
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
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], self.compl_weight()[:, :, :n_modes])
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
    def __init__(self, in_channels: int, width: int, modes: int, depth: int, hidden_proj: int) -> None:
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

        out = x[:, 0, :]               # [B, Nz]
        out = torch.nn.functional.softplus(out)  # enforce positivity
        # normalize mass to 1 using zeta grid spacing (approx constant)
        return out


# ----------------------------
# Observables from lambda(zeta)
# ----------------------------
def dz_from_grid(z: np.ndarray) -> float:
    return float((z[-1] - z[0]) / max(1, z.size - 1))


def normalize_lambda(lam: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    dz = dz_from_grid(z)
    mass = float(np.sum(lam) * dz)
    lamn = lam / (mass + eps)
    return lamn, mass


def moments_from_lambda(lam: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    lamn, mass = normalize_lambda(lam, z, eps)
    dz = dz_from_grid(z)

    mean = float(np.sum(z * lamn) * dz)
    var = float(np.sum((z - mean) ** 2 * lamn) * dz)
    rms = float(np.sqrt(max(var, 0.0)))

    m3 = float(np.sum((z - mean) ** 3 * lamn) * dz)
    m4 = float(np.sum((z - mean) ** 4 * lamn) * dz)

    skew = float(m3 / (rms**3 + eps))
    kurt = float(m4 / (rms**4 + eps))  # (Gaussian ~ 3)

    # Shannon entropy of discretized density: -∫ p log p dz
    p = np.clip(lamn, eps, None)
    entropy = float(-np.sum(p * np.log(p)) * dz)

    # "emittance-like" proxy (units of zeta): just sigma_z (cannot get true eps_z without delta)
    # We'll report sigma_z = rms
    return {
        "mass": mass,
        "mean_zeta": mean,
        "sigma_zeta": rms,
        "skew": skew,
        "kurt": kurt,
        "entropy": entropy,
    }


def compare_obs(true_obs: Dict[str, float], pred_obs: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    out = {}
    for k in true_obs:
        t = true_obs[k]
        p = pred_obs[k]
        out[f"{k}_true"] = t
        out[f"{k}_pred"] = p
        # relative error for scale-like quantities; abs error for mean/skew/kurt/entropy also useful
        out[f"{k}_abs_err"] = float(p - t)
        out[f"{k}_rel_err"] = float((p - t) / (abs(t) + eps))
    return out


def save_curve_and_residual(z, y_true, y_pred, out_path, title=""):
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(z, y_true, label="true")
    plt.plot(z, y_pred, label="pred", linestyle="--")
    plt.title(title or "lambda(zeta)")
    plt.xlabel("zeta")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    r = y_pred - y_true
    plt.plot(z, r)
    plt.axhline(0.0, color="k", lw=1)
    plt.title("residual (pred-true)")
    plt.xlabel("zeta")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------------
# Evaluation + plotting
# ----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_hist(true_vals, pred_vals, name: str, out_dir: str, bins: int = 50):
    plt.figure(figsize=(7, 4))
    plt.hist(true_vals, bins=bins, alpha=0.6, density=True, label="true")
    plt.hist(pred_vals, bins=bins, alpha=0.6, density=True, label="pred")
    plt.title(name)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{name}.png"), dpi=150)
    plt.close()


def save_scatter(x, y, name: str, out_dir: str):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.6)
    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(name)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"scatter_{name}.png"), dpi=150)
    plt.close()


def main():
    
    y_true_all = []
    y_pred_all = []
    cfg = EvalConfig(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        checkpoint_path=os.environ.get("CKPT_PATH", ""),
    )
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH env var to dataset .npz")
    if not cfg.checkpoint_path:
        raise ValueError("Set CKPT_PATH env var to trained FNO checkpoint .pt")

    ensure_dir(cfg.out_dir)

    data = load_npz(cfg.dataset_path)
    ds = LambdaDataset(data, cfg.split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = FNO1d(
        in_channels=cfg.in_channels,
        width=cfg.width,
        modes=cfg.modes,
        depth=cfg.depth,
        hidden_proj=cfg.hidden_proj,
    ).to(cfg.device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    # adapt depending on how you saved it
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    all_rows = []
    zeta_grid = data["zeta_grid"].astype(np.float64)

    # store arrays for plotting
    obs_keys = ["mass", "mean_zeta", "sigma_zeta", "skew", "kurt", "entropy"]
    true_vals = {k: [] for k in obs_keys}
    pred_vals = {k: [] for k in obs_keys}

    with torch.no_grad():
        for features, y_true, mu, grid in loader:
            features = features.to(cfg.device)   # [B,Nz,5]
            y_true_t = y_true.to(cfg.device)     # [B,Nz]
            y_pred_t = model(features)           # [B,Nz]

            y_true_np = y_true_t.cpu().numpy().astype(np.float64)
            y_pred_np = y_pred_t.cpu().numpy().astype(np.float64)
            mu_np = mu.numpy().astype(np.float64)

            for i in range(y_true_np.shape[0]):
                obs_t = moments_from_lambda(y_true_np[i], zeta_grid, cfg.eps)
                obs_p = moments_from_lambda(y_pred_np[i], zeta_grid, cfg.eps)
                lam_t, _ = normalize_lambda(y_true_np[i], zeta_grid, cfg.eps)
                lam_p, _ = normalize_lambda(y_pred_np[i], zeta_grid, cfg.eps)
                y_true_all.append(lam_t)
                y_pred_all.append(lam_p)

                for k in obs_keys:
                    true_vals[k].append(obs_t[k])
                    pred_vals[k].append(obs_p[k])

                row = {
                    "mu1": mu_np[i, 0],
                    "mu1": mu_np[i, 1],
                    "mu2": mu_np[i, 2],
                }
                row.update(compare_obs(obs_t, obs_p, cfg.eps))
                all_rows.append(row)
                ex_dir = os.path.join(cfg.out_dir, "examples")
                ensure_dir(ex_dir)

                if len(all_rows) < 20:  # first 20 samples
                    save_curve_and_residual(
                        zeta_grid,
                        y_true_np[i],
                        y_pred_np[i],
                        out_path=os.path.join(ex_dir, f"sample_{len(all_rows):04d}.png"),
                        title=f"{cfg.split} sample {len(all_rows)}"
                    )
    
    Yt = np.stack(y_true_all, axis=0)   # [Ns, Nz]
    Yp = np.stack(y_pred_all, axis=0)   # [Ns, Nz]
    R = Yp - Yt                         # residuals

    # --- Line density aggregate plots (truth vs pred)
    save_mean_density(zeta_grid, Yt, Yp, os.path.join(cfg.out_dir, f"lambda_mean_{cfg.split}.png"))
    save_mean_residual(zeta_grid, R, os.path.join(cfg.out_dir, f"lambda_mean_residual_{cfg.split}.png"))
    save_rmse_vs_z(zeta_grid, R, os.path.join(cfg.out_dir, f"lambda_rmse_vs_z_{cfg.split}.png"))
    save_residual_hist(R, os.path.join(cfg.out_dir, f"lambda_residual_hist_{cfg.split}.png"))
    save_residual_heatmap(zeta_grid, R, os.path.join(cfg.out_dir, f"lambda_residual_heatmap_{cfg.split}.png"))

    # Save CSV
    csv_path = os.path.join(cfg.out_dir, f"observables_{cfg.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print("[OK] wrote", csv_path)

    # Save summary JSON
    summary = {}
    for k in obs_keys:
        t = np.array(true_vals[k])
        p = np.array(pred_vals[k])
        summary[k] = {
            "true_mean": float(np.mean(t)),
            "pred_mean": float(np.mean(p)),
            "true_std": float(np.std(t)),
            "pred_std": float(np.std(p)),
            "mae": float(np.mean(np.abs(p - t))),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        }
    json_path = os.path.join(cfg.out_dir, f"summary_{cfg.split}.json")
    Path(json_path).write_text(json.dumps(summary, indent=2))
    print("[OK] wrote", json_path)

    # Plots
    for k in obs_keys:
        save_hist(true_vals[k], pred_vals[k], k, cfg.out_dir)
        save_scatter(np.array(true_vals[k]), np.array(pred_vals[k]), k, cfg.out_dir)

    print("[OK] plots saved to", cfg.out_dir)


if __name__ == "__main__":
    main()