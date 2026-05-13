# experiments/runs/demo-run-002/eval_cp6d_neuralop.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import the training components (same file/folder as train_cp6d_neuralop.py)
from train_cp6d_neuralop import (
    Config,
    CP6DBuilder,
    CP6DNeuralOp,
    CP6DDataset,
    DIM_NAMES,
    DIM_IDX,
)

# --- NeuralOperator FNO import (version-tolerant) ---
try:
    from neuralop.models import FNO
except Exception:
    from neuralop.models.fno import FNO


@dataclass
class EvalCfg:
    dataset_path: str
    ckpt_path: str
    meta_path: str
    out_dir: str = "./experiments/runs/demo-run-002/output_eval"
    split: str = "test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    n_eval_examples: int = 6  # how many example factor plots


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_hist(true_vals: List[float], pred_vals: List[float], out_path: str, title: str):
    plt.figure(figsize=(7, 4))
    plt.hist(true_vals, bins=50, alpha=0.6, density=True, label="true")
    plt.hist(pred_vals, bins=50, alpha=0.6, density=True, label="pred")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter(x: List[float], y: List[float], out_path: str, title: str):
    x = np.array(x)
    y = np.array(y)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.6)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_factor_grid_plot(
    factors_true: np.ndarray,  # [6,R,B]
    factors_pred: np.ndarray,  # [6,R,B]
    grids: Dict[str, np.ndarray],
    out_path: str,
    title: str = "",
    max_rank_plots: int = 4,
):
    """
    Plot a few rank components for each dimension.
    """
    D, R, B = factors_true.shape
    K = min(max_rank_plots, R)

    fig, axes = plt.subplots(D, K, figsize=(3.5 * K, 2.5 * D), constrained_layout=True)
    if D == 1:
        axes = np.expand_dims(axes, axis=0)
    if K == 1:
        axes = np.expand_dims(axes, axis=1)

    for di, name in enumerate(DIM_NAMES):
        x = grids[name]
        for r in range(K):
            ax = axes[di, r]
            ax.plot(x, factors_true[di, r], label="true", lw=1.5)
            ax.plot(x, factors_pred[di, r], label="pred", lw=1.2, alpha=0.9)
            ax.set_title(f"{name} | r={r}")
            ax.grid(alpha=0.25)
            if di == 0 and r == 0:
                ax.legend()

    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cp_marginals_from_factors(factors: torch.Tensor) -> torch.Tensor:
    """
    Since this demo predicts factors only (no amplitudes),
    we approximate the marginal for each dim by averaging over rank.
    factors: [B,6,R,Bins] -> marginals: [B,6,Bins]
    """
    # average across rank components, then normalize
    m = factors.mean(dim=2)  # [B,6,Bins]
    m = m / (m.sum(dim=-1, keepdim=True) + 1e-12)
    return m


def moments_1d_batch(p: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    p: [B,Bins] discrete prob (assume sums ~1)
    grid: [Bins]
    returns mean: [B], sigma: [B]
    """
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    mu = (p * grid.view(1, -1)).sum(dim=-1)
    var = (p * (grid.view(1, -1) - mu[:, None]) ** 2).sum(dim=-1)
    return mu, torch.sqrt(var.clamp_min(0.0))


def main():
    cfg = EvalCfg(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        ckpt_path=os.environ.get("CP6D_CKPT", ""),
        meta_path=os.environ.get("CP6D_META", ""),
        out_dir=os.environ.get("OUT_DIR", "./experiments/runs/demo-run-002/output_eval"),
        split=os.environ.get("SPLIT", "test"),
    )
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")
    if not cfg.ckpt_path:
        raise ValueError("Set CP6D_CKPT")
    if not cfg.meta_path:
        raise ValueError("Set CP6D_META")

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "examples"))

    raw = np.load(cfg.dataset_path, allow_pickle=True)
    X = raw["X_cloud"]
    Y = raw["Y_cloud"]
    MU = raw["MU"]
    idx = raw[cfg.split]

    meta = json.loads(Path(cfg.meta_path).read_text())
    tr_cfg = meta["config"]

    # Rebuild builder and overwrite grids EXACTLY from meta
    # (Important: you want evaluation fields/factors consistent with training.)
    sub = np.concatenate([X[raw["train"][:50]], Y[raw["train"][:50]]], axis=0)
    builder = CP6DBuilder(
        sub,
        n_bins=tr_cfg["n_bins"],
        pct_lo=tr_cfg["pct_lo"],
        pct_hi=tr_cfg["pct_hi"],
        sigma_steps=tr_cfg["sigma_steps"],
        rank=tr_cfg["rank"],
        seed=tr_cfg["seed"],
    ).to(cfg.device)

    builder.grids = {k: np.array(v, dtype=np.float64) for k, v in meta["grids"].items()}
    builder.deltas = {k: float(v) for k, v in meta["deltas"].items()}
    builder.tgrids = {k: torch.tensor(builder.grids[k], dtype=torch.float32, device=cfg.device) for k in builder.grids.keys()}

    ds = CP6DDataset(X, Y, MU, idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Build model and load ckpt
    model = CP6DNeuralOp(
        rank=tr_cfg["rank"],
        n_bins=tr_cfg["n_bins"],
        mu_dim=tr_cfg.get("mu_dim", 3),
        width=tr_cfg["fno_width"],
        modes=tr_cfg["fno_modes"],
        layers=tr_cfg["fno_layers"],
    ).to(cfg.device)

    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Collect moment metrics
    true_mu: Dict[str, List[float]] = {k: [] for k in DIM_NAMES}
    pred_mu: Dict[str, List[float]] = {k: [] for k in DIM_NAMES}
    true_sig: Dict[str, List[float]] = {k: [] for k in DIM_NAMES}
    pred_sig: Dict[str, List[float]] = {k: [] for k in DIM_NAMES}

    # Save a few example factor plots
    ex_saved = 0

    with torch.no_grad():
        for xb, yb, mub in loader:
            xb = xb.to(cfg.device)   # [B,Np,6]
            yb = yb.to(cfg.device)
            mub = mub.to(cfg.device)

            # Build CP factors for batch
            fx_list = []
            fy_list = []
            for b in range(xb.shape[0]):
                fx, _ = builder.cloud_to_cp(xb[b])  # [6,R,Bins]
                fy, _ = builder.cloud_to_cp(yb[b])
                fx_list.append(fx)
                fy_list.append(fy)
            fx = torch.stack(fx_list, dim=0)  # [B,6,R,Bins]
            fy = torch.stack(fy_list, dim=0)

            fhat = model(fx, mub)  # [B,6,R,Bins]

            # Compute approximate 1D marginals and moments
            m_true = cp_marginals_from_factors(fy)    # [B,6,Bins]
            m_pred = cp_marginals_from_factors(fhat)

            for di, name in enumerate(DIM_NAMES):
                grid = builder.tgrids[name]  # [Bins]
                mu_t, sig_t = moments_1d_batch(m_true[:, di], grid)
                mu_p, sig_p = moments_1d_batch(m_pred[:, di], grid)

                true_mu[name].extend(mu_t.detach().cpu().numpy().tolist())
                pred_mu[name].extend(mu_p.detach().cpu().numpy().tolist())
                true_sig[name].extend(sig_t.detach().cpu().numpy().tolist())
                pred_sig[name].extend(sig_p.detach().cpu().numpy().tolist())

            # Example plots (true vs pred factors)
            if ex_saved < cfg.n_eval_examples:
                B = fx.shape[0]
                take = min(B, cfg.n_eval_examples - ex_saved)
                for j in range(take):
                    ft = fy[j].detach().cpu().numpy()
                    fp = fhat[j].detach().cpu().numpy()
                    out_path = os.path.join(cfg.out_dir, "examples", f"factors_{ex_saved:03d}.png")
                    save_factor_grid_plot(
                        ft, fp,
                        grids={k: builder.grids[k] for k in builder.grids.keys()},
                        out_path=out_path,
                        title=f"example={ex_saved}",
                        max_rank_plots=4,
                    )
                    ex_saved += 1

    # Write plots
    for name in DIM_NAMES:
        save_hist(true_mu[name], pred_mu[name], os.path.join(cfg.out_dir, f"hist_mean_{name}.png"), f"mean {name}")
        save_scatter(true_mu[name], pred_mu[name], os.path.join(cfg.out_dir, f"scatter_mean_{name}.png"), f"mean {name}")

        save_hist(true_sig[name], pred_sig[name], os.path.join(cfg.out_dir, f"hist_sigma_{name}.png"), f"sigma {name}")
        save_scatter(true_sig[name], pred_sig[name], os.path.join(cfg.out_dir, f"scatter_sigma_{name}.png"), f"sigma {name}")

    metrics = {
        "split": cfg.split,
        "n_samples": len(ds),
    }
    Path(os.path.join(cfg.out_dir, f"metrics_{cfg.split}.json")).write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote eval outputs to:", cfg.out_dir)


if __name__ == "__main__":
    main()