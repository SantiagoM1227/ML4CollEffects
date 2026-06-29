from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.data.latent_dataset import LatentNPZDataset
from scripts.models.latent_dynamics import latent_dynamics_loss, temporal_correlation, wasserstein_1d
from scripts.plotting.plot_dynamics import save_attention_map, save_hist_triplet
from scripts.tokenizer import ElementTokenizer, MuNormalizer
from scripts.tracking_transformer import TrackingTransformer
from scripts.vae_15x2d import ConvVAE2D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-npz", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--dyn-ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--n-plots", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    plot_dir = outdir / "plots"
    metrics_dir = outdir / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = LatentNPZDataset(args.latent_npz, split=args.split)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    dyn_ckpt = torch.load(args.dyn_ckpt, map_location="cpu")
    dargs = dyn_ckpt.get("args", {})

    model = TrackingTransformer(
        z_dim=256,
        h_dim=512,
        d_model=int(dargs.get("d_model", 512)),
        n_layers=int(dargs.get("n_layers", 4)),
        n_heads=int(dargs.get("n_heads", 8)),
        dropout=float(dargs.get("dropout", 0.0)),
    )
    model.load_state_dict(dyn_ckpt["state_dict"], strict=True)
    model.to(device).eval()

    mu_norm = MuNormalizer.from_state_dict(dyn_ckpt["mu_normalizer"]).to(device)
    tok = ElementTokenizer(mu_dim=3, d_token=512, n_pos_freqs=16, mu_normalizer=mu_norm)
    tok.load_state_dict(dyn_ckpt["tokenizer_state"], strict=True)
    tok.to(device).eval()

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(vae_ckpt.get("bins", 64))
    latent_dim = int(vae_ckpt.get("latent_dim", 256))

    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=latent_dim)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    loss_sum = mse_sum = mae_sum = rel_sum = 0.0
    n = 0
    plotted = 0
    dz_true_all = []
    dz_pred_all = []
    w1_vals = []
    rollout_drifts = []
    rollout_stability = []

    for batch in dl:
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        mu = batch["mu"].to(device)
        s = batch["s"].to(device)

        h = tok(mu[:, None, :], s[:, None])
        z_seq, attn = model(z0, h, return_attention=True)
        z1p = z_seq[:, 1, :]

        loss, logs = latent_dynamics_loss(z1p, z1)

        B = z0.shape[0]
        loss_sum += float(loss.item()) * B
        mse_sum += float(logs["mse"].item()) * B
        mae_sum += float(logs["mae"].item()) * B
        rel_sum += float(logs["rel_l2"].item()) * B
        n += B

        dz_true = (z1 - z0).detach().cpu().numpy()
        dz_pred = (z1p - z0).detach().cpu().numpy()
        dz_true_all.append(dz_true)
        dz_pred_all.append(dz_pred)
        w1_vals.append(wasserstein_1d(dz_true, dz_pred))

        zt = z0.clone()
        norms = []
        for _ in range(args.rollout_steps):
            zt = model(zt, h)[:, 1, :]
            norms.append(torch.norm(zt - z0, dim=-1).mean().item())
        rollout_drifts.append(float(np.mean(norms)))
        rollout_stability.append(float(np.std(norms)))

        if plotted == 0:
            save_attention_map(attn.detach().cpu().numpy(), plot_dir / "attention_eval.png")

        if plotted < args.n_plots:
            k = min(args.n_plots - plotted, B)
            y_truth = vae.decode(z1[:k]).detach().cpu().numpy()
            y_pred = vae.decode(z1p[:k]).detach().cpu().numpy()
            for i in range(k):
                save_hist_triplet(y_truth[i], y_pred[i], plot_dir / f"truth_vs_pred_{plotted:03d}.png", n_channels=6)
                plotted += 1
                if plotted >= args.n_plots:
                    break

    dz_true_np = np.concatenate(dz_true_all, axis=0) if dz_true_all else np.zeros((0, 1), dtype=np.float32)
    dz_pred_np = np.concatenate(dz_pred_all, axis=0) if dz_pred_all else np.zeros((0, 1), dtype=np.float32)

    metrics = {
        "split": args.split,
        "n_samples": int(n),
        "loss": float(loss_sum / max(n, 1)),
        "latent_mse": float(mse_sum / max(n, 1)),
        "latent_mae": float(mae_sum / max(n, 1)),
        "latent_rel_l2": float(rel_sum / max(n, 1)),
        "temporal_corr": temporal_correlation(dz_true_np, dz_pred_np),
        "trajectory_wasserstein": float(np.mean(w1_vals)) if w1_vals else 0.0,
        "rollout_drift": float(np.mean(rollout_drifts)) if rollout_drifts else 0.0,
        "rollout_stability": float(np.mean(rollout_stability)) if rollout_stability else 0.0,
        "latent_npz": args.latent_npz,
        "vae_ckpt": args.vae_ckpt,
        "dyn_ckpt": args.dyn_ckpt,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", metrics_dir / "metrics.json")


if __name__ == "__main__":
    main()
