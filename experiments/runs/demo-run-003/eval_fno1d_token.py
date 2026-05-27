from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_demo003_latent1d_tokens_neuralop import (
    Config,
    load_npz,
    CloudDataset,
    Latent1DTokenAE,
    LatentTokenNeuralOp,
    chamfer_l2,
)


@dataclass
class EvalCfg:
    dataset_path: str
    ae_ckpt: str
    op_ckpt: str
    meta_path: str

    out_dir: str = "./experiments/runs/demo-run-003/output_eval"
    split: str = "test"
    batch_size: int = 8
    n_examples: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_scatter(x, y, out_path: str, title: str):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.6)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_cloud_overlay(true_cloud: np.ndarray, pred_cloud: np.ndarray, out_path: str, title: str):
    # overlay in (zeta,x) just as a sanity view
    plt.figure(figsize=(7, 4))
    plt.scatter(true_cloud[:, 2], true_cloud[:, 0], s=1, alpha=0.35, label="true")
    plt.scatter(pred_cloud[:, 2], pred_cloud[:, 0], s=1, alpha=0.35, label="pred")
    plt.xlabel("zeta")
    plt.ylabel("x")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def main():
    cfg = EvalCfg(
        dataset_path=os.environ.get("DATASET_PATH", ""),
        ae_ckpt=os.environ.get("DEMO003_AE", ""),
        op_ckpt=os.environ.get("DEMO003_OP", ""),
        meta_path=os.environ.get("DEMO003_META", ""),
        out_dir=os.environ.get("OUT_DIR", "./experiments/runs/demo-run-003/output_eval"),
        split=os.environ.get("SPLIT", "test"),
    )
    if not cfg.dataset_path:
        raise ValueError("Set DATASET_PATH")
    if not cfg.ae_ckpt:
        raise ValueError("Set DEMO003_AE")
    if not cfg.op_ckpt:
        raise ValueError("Set DEMO003_OP")
    if not cfg.meta_path:
        raise ValueError("Set DEMO003_META")

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "examples"))
    ensure_dir(os.path.join(cfg.out_dir, "scatter"))

    meta = json.loads(Path(cfg.meta_path).read_text())
    tr_cfg = meta["config"]

    data = load_npz(cfg.dataset_path)
    ds = CloudDataset(data, cfg.split, Np=int(tr_cfg.get("Np", 4096)))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # build models
    ae = Latent1DTokenAE(
        token_dim=int(tr_cfg["token_dim"]),
        particle_hidden=int(tr_cfg["particle_hidden"]),
        token_hidden=int(tr_cfg["token_hidden"]),
    ).to(cfg.device)

    op = LatentTokenNeuralOp(
        token_dim=int(tr_cfg["token_dim"]),
        mu_dim=int(tr_cfg.get("mu_dim", 3)),
        width=int(tr_cfg["fno_width"]),
        modes=int(tr_cfg["fno_modes"]),
        layers=int(tr_cfg["fno_layers"]),
    ).to(cfg.device)

    # load ckpts
    ae_ckpt = torch.load(cfg.ae_ckpt, map_location=cfg.device, weights_only=False)
    op_ckpt = torch.load(cfg.op_ckpt, map_location=cfg.device, weights_only=False)

    ae.load_state_dict(ae_ckpt["state_dict"], strict=False)
    op.load_state_dict(op_ckpt["state_dict"], strict=False)
    ae.eval()
    op.eval()

    chamfers: List[float] = []
    token_mses: List[float] = []

    # scatter: global mean x, delta from point clouds
    true_mean_x: List[float] = []
    pred_mean_x: List[float] = []
    true_mean_delta: List[float] = []
    pred_mean_delta: List[float] = []

    ex_saved = 0

    for Xb, Yb, mu in loader:
        Xb = Xb.to(cfg.device)
        Yb = Yb.to(cfg.device)
        mu = mu.to(cfg.device)

        Zx, pos, _ = ae(Xb, M=int(tr_cfg["M"]), Np=Yb.shape[1])
        Zy, _posy, _ = ae(Yb, M=int(tr_cfg["M"]), Np=Yb.shape[1])

        Zhat = op(Zx, mu, pos)

        token_mse = torch.mean((Zhat - Zy) ** 2)
        token_mses.append(float(token_mse.item()))

        Yhat = ae.decode_sample(Zhat, pos, Np=Yb.shape[1])
        ch = chamfer_l2(Yhat, Yb)
        chamfers.append(float(ch.item()))

        # scatter stats
        true_mean_x.extend(Yb[:, :, 0].mean(dim=1).detach().cpu().numpy().tolist())
        pred_mean_x.extend(Yhat[:, :, 0].mean(dim=1).detach().cpu().numpy().tolist())
        true_mean_delta.extend(Yb[:, :, 5].mean(dim=1).detach().cpu().numpy().tolist())
        pred_mean_delta.extend(Yhat[:, :, 5].mean(dim=1).detach().cpu().numpy().tolist())

        if ex_saved < cfg.n_examples:
            take = min(Xb.size(0), cfg.n_examples - ex_saved)
            for j in range(take):
                yt = Yb[j].detach().cpu().numpy()
                yp = Yhat[j].detach().cpu().numpy()
                save_cloud_overlay(
                    yt, yp,
                    os.path.join(cfg.out_dir, "examples", f"overlay_zeta_x_{ex_saved:03d}.png"),
                    title=f"example={ex_saved} (zeta-x overlay)"
                )
                ex_saved += 1

    # scatter plots
    save_scatter(true_mean_x, pred_mean_x, os.path.join(cfg.out_dir, "scatter", "mean_x.png"), "mean_x (cloud)")
    save_scatter(true_mean_delta, pred_mean_delta, os.path.join(cfg.out_dir, "scatter", "mean_delta.png"), "mean_delta (cloud)")

    metrics = {
        "split": cfg.split,
        "n_samples": len(ds),
        "token_mse_mean": float(np.mean(token_mses)),
        "chamfer_mean": float(np.mean(chamfers)),
    }
    Path(os.path.join(cfg.out_dir, f"metrics_{cfg.split}.json")).write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote eval outputs to:", cfg.out_dir)
    print("[OK] metrics:", metrics)


if __name__ == "__main__":
    main()