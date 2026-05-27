# experiments/runs/demo-run-002/eval_cp6d_neuralop.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_cp6d_neuralop import (
    FieldBuilder,
    OperatorFieldDataset,
    PlaneFNO2d,
    field_loss,
    high_freq_penalty,
)


@dataclass
class EvalCfg:
    dataset_path: str
    ckpt_path: str
    meta_path: str
    out_dir: str = "./experiments/runs/demo-run-002/output_eval"
    split: str = "test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    n_eval_examples: int = 6


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_plane_triplet(true_planes: np.ndarray, pred_planes: np.ndarray, out_path: str, title: str = ""):
    """
    true_planes/pred_planes: [3,H,W] with planes (xpx, ypy, zd)
    """
    names = ["x-px", "y-py", "zeta-delta"]
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=True)

    for i in range(3):
        ax_t = axes[i, 0]
        ax_p = axes[i, 1]

        im0 = ax_t.imshow(true_planes[i], origin="lower", aspect="auto")
        ax_t.set_title(f"true {names[i]}")
        fig.colorbar(im0, ax=ax_t, fraction=0.046, pad=0.04)

        im1 = ax_p.imshow(pred_planes[i], origin="lower", aspect="auto")
        ax_p.set_title(f"pred {names[i]}")
        fig.colorbar(im1, ax=ax_p, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_residual_triplet(true_planes: np.ndarray, pred_planes: np.ndarray, out_path: str, title: str = ""):
    """
    Plot residuals (pred - true) for the 3 planes.
    """
    names = ["x-px", "y-py", "zeta-delta"]
    resid = pred_planes - true_planes
    vmax = float(np.quantile(np.abs(resid), 0.99) + 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), constrained_layout=True)
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(resid[i], origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"residual {names[i]} (pred-true)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_scatter(true_vals, pred_vals, out_path: str, title: str, xlabel="true", ylabel="pred"):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)

    plt.figure(figsize=(5, 5))
    plt.scatter(true_vals, pred_vals, s=8, alpha=0.6)
    mn = float(min(true_vals.min(), pred_vals.min()))
    mx = float(max(true_vals.max(), pred_vals.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def moments_2d_density(rho: torch.Tensor, u_grid: torch.Tensor, v_grid: torch.Tensor):
    """
    rho:   [B,H,W] density-like, assumed nonnegative (we renormalize internally)
    u_grid:[H]
    v_grid:[W]
    returns dict with tensors [B]: mu_u, mu_v, sig_u, sig_v, corr
    """
    B, H, W = rho.shape
    rho = rho / (rho.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    u = u_grid.view(1, H, 1).expand(B, H, W)
    v = v_grid.view(1, 1, W).expand(B, H, W)

    mu_u = (rho * u).sum(dim=(-2, -1))
    mu_v = (rho * v).sum(dim=(-2, -1))

    du = u - mu_u.view(B, 1, 1)
    dv = v - mu_v.view(B, 1, 1)

    var_u = (rho * du * du).sum(dim=(-2, -1)).clamp_min(0.0)
    var_v = (rho * dv * dv).sum(dim=(-2, -1)).clamp_min(0.0)

    sig_u = torch.sqrt(var_u + 1e-12)
    sig_v = torch.sqrt(var_v + 1e-12)

    cov = (rho * du * dv).sum(dim=(-2, -1))
    corr = cov / (sig_u * sig_v + 1e-12)

    return {
        "mu_u": mu_u,
        "mu_v": mu_v,
        "sig_u": sig_u,
        "sig_v": sig_v,
        "corr": corr,
    }

def main():
    cfg = EvalCfg(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        ckpt_path=os.environ.get("CP6D_CKPT", ""),
        meta_path=os.environ.get("CP6D_META", ""),
        out_dir=os.environ.get("OUT_DIR", "./experiments/runs/demo-run-002/output_eval"),
        split=os.environ.get("SPLIT", "test"),
    )

    print("[EVAL] meta_path =", cfg.meta_path)
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")
    if not cfg.ckpt_path:
        raise ValueError("Set CP6D_CKPT")
    if not cfg.meta_path:
        raise ValueError("Set CP6D_META")

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "examples"))
    ensure_dir(os.path.join(cfg.out_dir, "examples_resid"))
    ensure_dir(os.path.join(cfg.out_dir, "scatter"))

    raw = np.load(cfg.dataset_path, allow_pickle=True)
    X = raw["X_cloud"]
    Y = raw["Y_cloud"]
    MU = raw["MU"]
    idx = raw[cfg.split]

    meta = json.loads(Path(cfg.meta_path).read_text())
    tr_cfg = meta["config"]

    # Rebuild FieldBuilder + overwrite grids EXACTLY from meta
    sub = np.concatenate([X[raw["train"][:50]], Y[raw["train"][:50]]], axis=0)
    fb = FieldBuilder(
        X_cloud_train_subset=sub,
        grid_n=tr_cfg["grid_n"],
        lo=tr_cfg["percentile_lo"],
        hi=tr_cfg["percentile_hi"],
        sigma_steps=tr_cfg["sigma_steps"],
    ).to(cfg.device)

    g = meta["grids"]
    fb.x_grid = torch.tensor(g["x_grid"], dtype=torch.float32, device=cfg.device)
    fb.y_grid = torch.tensor(g["y_grid"], dtype=torch.float32, device=cfg.device)
    fb.z_grid = torch.tensor(g["z_grid"], dtype=torch.float32, device=cfg.device)
    fb.px_grid = torch.tensor(g["px_grid"], dtype=torch.float32, device=cfg.device)
    fb.py_grid = torch.tensor(g["py_grid"], dtype=torch.float32, device=cfg.device)
    fb.d_grid = torch.tensor(g["d_grid"], dtype=torch.float32, device=cfg.device)

    ds = OperatorFieldDataset(X, Y, MU, idx, fb)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = PlaneFNO2d(
        mu_dim=tr_cfg.get("mu_dim", 3),
        width=tr_cfg["fno_width"],
        modes=tr_cfg["fno_modes"],
        layers=tr_cfg["fno_layers"],
    ).to(cfg.device)

    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if isinstance(sd, dict):
        sd.pop("_metadata", None)

    model.load_state_dict(sd, strict=False)
    model.eval()

    metrics = {
        "split": cfg.split,
        "n_samples": len(ds),
        "field_loss_mean": 0.0,
        "mse_mean": 0.0,
        "hf_penalty_mean": 0.0,
    }

    sv_field = 0.0
    sv_mse = 0.0
    sv_hf = 0.0
    nv = 0

    # scalar summaries for scatter plots
    plane_names = ["xpx", "ypy", "zd"]
    true_stats = {k: [] for k in [
        "mu_u_xpx", "mu_v_xpx", "sig_u_xpx", "sig_v_xpx", "corr_xpx",
        "mu_u_ypy", "mu_v_ypy", "sig_u_ypy", "sig_v_ypy", "corr_ypy",
        "mu_u_zd",  "mu_v_zd",  "sig_u_zd",  "sig_v_zd",  "corr_zd",
    ]}
    pred_stats = {k: [] for k in true_stats.keys()}

    ex_saved = 0

    with torch.no_grad():
        for Fin, Fout, mu in loader:
            Fin = Fin.to(cfg.device)     # [B,3,H,W]
            Fout = Fout.to(cfg.device)   # [B,3,H,W]
            mu = mu.to(cfg.device)       # [B,3]

            Fhat = model(Fin, mu)
            # per-plane moments for scatter (use the grids from FieldBuilder)
            # channel 0: (x,px) uses (x_grid, px_grid)
            # channel 1: (y,py) uses (y_grid, py_grid)
            # channel 2: (zeta,delta) uses (z_grid, d_grid)
            mom_t0 = moments_2d_density(Fout[:, 0], fb.x_grid, fb.px_grid)
            mom_p0 = moments_2d_density(Fhat[:, 0], fb.x_grid, fb.px_grid)

            mom_t1 = moments_2d_density(Fout[:, 1], fb.y_grid, fb.py_grid)
            mom_p1 = moments_2d_density(Fhat[:, 1], fb.y_grid, fb.py_grid)

            mom_t2 = moments_2d_density(Fout[:, 2], fb.z_grid, fb.d_grid)
            mom_p2 = moments_2d_density(Fhat[:, 2], fb.z_grid, fb.d_grid)

            def _extend(prefix, mt, mp):
                true_stats[f"mu_u_{prefix}"].extend(mt["mu_u"].detach().cpu().numpy().tolist())
                true_stats[f"mu_v_{prefix}"].extend(mt["mu_v"].detach().cpu().numpy().tolist())
                true_stats[f"sig_u_{prefix}"].extend(mt["sig_u"].detach().cpu().numpy().tolist())
                true_stats[f"sig_v_{prefix}"].extend(mt["sig_v"].detach().cpu().numpy().tolist())
                true_stats[f"corr_{prefix}"].extend(mt["corr"].detach().cpu().numpy().tolist())

                pred_stats[f"mu_u_{prefix}"].extend(mp["mu_u"].detach().cpu().numpy().tolist())
                pred_stats[f"mu_v_{prefix}"].extend(mp["mu_v"].detach().cpu().numpy().tolist())
                pred_stats[f"sig_u_{prefix}"].extend(mp["sig_u"].detach().cpu().numpy().tolist())
                pred_stats[f"sig_v_{prefix}"].extend(mp["sig_v"].detach().cpu().numpy().tolist())
                pred_stats[f"corr_{prefix}"].extend(mp["corr"].detach().cpu().numpy().tolist())

            _extend("xpx", mom_t0, mom_p0)
            _extend("ypy", mom_t1, mom_p1)
            _extend("zd",  mom_t2, mom_p2)

            v_field = field_loss(Fhat, Fout)
            v_mse = torch.mean((Fhat - Fout) ** 2)
            v_hf = high_freq_penalty(Fhat, keep=tr_cfg["fno_modes"])

            bsz = Fin.size(0)
            sv_field += float(v_field.item()) * bsz
            sv_mse += float(v_mse.item()) * bsz
            sv_hf += float(v_hf.item()) * bsz
            nv += bsz

            if ex_saved < cfg.n_eval_examples:
                take = min(bsz, cfg.n_eval_examples - ex_saved)
                for j in range(take):
                    t = Fout[j].detach().cpu().numpy()
                    p = Fhat[j].detach().cpu().numpy()
                    out_path = os.path.join(cfg.out_dir, "examples", f"planes_{ex_saved:03d}.png")
                    save_plane_triplet(t, p, out_path, title=f"example={ex_saved}")
                    out_resid = os.path.join(cfg.out_dir, "examples_resid", f"resid_{ex_saved:03d}.png")
                    save_residual_triplet(t, p, out_resid, title=f"example={ex_saved}")
                    ex_saved += 1

    # Scatter plots: true vs pred for each scalar summary
    for k in true_stats.keys():
        out_path = os.path.join(cfg.out_dir, "scatter", f"{k}.png")
        save_scatter(true_stats[k], pred_stats[k], out_path, title=k)
        
    metrics["field_loss_mean"] = sv_field / max(1, nv)
    metrics["mse_mean"] = sv_mse / max(1, nv)
    metrics["hf_penalty_mean"] = sv_hf / max(1, nv)

    Path(os.path.join(cfg.out_dir, f"metrics_{cfg.split}.json")).write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote eval outputs to:", cfg.out_dir)
    print("[OK] metrics:", metrics)


if __name__ == "__main__":
    main()