from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.npz_dataset import XsuiteNPZDataset, collate_cloud
from scripts.plotting.plot_vae import (
    save_latent_covariance_heatmap,
    save_latent_histograms,
    save_latent_pca_umap,
    save_recon_grid_15ch,
)
from scripts.trainers.train_vae import compute_recon_metrics
from scripts.vae_tf import ConvVAE2D, cloud6d_to_15x2d_hist, vae_loss


@torch.no_grad()
def batch_cloud_to_hist(batch_cloud: torch.Tensor, bins: int, ranges) -> torch.Tensor:
    out = []
    for b in range(batch_cloud.shape[0]):
        out.append(torch.from_numpy(cloud6d_to_15x2d_hist(batch_cloud[b].cpu().numpy(), bins=bins, ranges=ranges)))
    return torch.stack(out, dim=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--beta", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    plot_dir = outdir / "plots"
    metrics_dir = outdir / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(ckpt.get("bins", args.bins))
    latent_dim = int(ckpt.get("latent_dim", 256))
    global_ranges = tuple((float(a), float(b)) for (a, b) in ckpt["global_ranges"])

    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=latent_dim)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    base = XsuiteNPZDataset(args.data)
    raw = np.load(args.data, allow_pickle=True)

    if args.split == "all":
        ds = base
    else:
        if args.split in raw.files:
            idx = raw[args.split].astype(np.int64)
        else:
            n = len(base)
            n_train = int(0.8 * n)
            if args.split == "train":
                idx = np.arange(0, n_train, dtype=np.int64)
            elif args.split == "val":
                idx = np.arange(n_train, n, dtype=np.int64)
            else:
                idx = np.array([], dtype=np.int64)
        ds = Subset(base, idx.tolist())

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_cloud)

    loss_sum = recon_sum = kl_sum = 0.0
    n = 0
    metric_keys = ["mse", "mae", "rel_l2", "ssim", "kl_hist", "js_hist", "wasserstein", "spectral_error"]
    metric_acc = {k: 0.0 for k in metric_keys}

    mu_list = []
    logvar_list = []
    first_x = None
    first_xhat = None

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= args.max_batches:
                break
            x = batch_cloud_to_hist(batch["cloud"].to(device), bins, global_ranges).to(device)
            xhat, mu, logvar, _ = vae(x)
            loss, logs = vae_loss(xhat, x, mu, logvar, beta=args.beta)

            B = x.shape[0]
            loss_sum += float(loss.item()) * B
            recon_sum += float(logs["recon"].item()) * B
            kl_sum += float(logs["kl"].item()) * B
            n += B

            met = compute_recon_metrics(x, xhat)
            for k in metric_keys:
                metric_acc[k] += float(met[k]) * B

            mu_list.append(mu.detach().cpu().numpy())
            logvar_list.append(logvar.detach().cpu().numpy())

            if first_x is None:
                first_x = x[0].detach().cpu().numpy()
                first_xhat = xhat[0].detach().cpu().numpy()

    for k in metric_keys:
        metric_acc[k] /= max(n, 1)

    metrics = {
        "split": args.split,
        "n_samples": int(n),
        "loss_mean": float(loss_sum / max(n, 1)),
        "recon_mean": float(recon_sum / max(n, 1)),
        "kl_mean": float(kl_sum / max(n, 1)),
        "vae_ckpt": args.vae_ckpt,
        **metric_acc,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    mu_np = np.concatenate(mu_list, axis=0) if mu_list else np.zeros((0, latent_dim), dtype=np.float32)
    logvar_np = np.concatenate(logvar_list, axis=0) if logvar_list else np.zeros((0, latent_dim), dtype=np.float32)

    save_latent_histograms(mu_np, logvar_np, plot_dir / "latent_hist_eval.png")
    save_latent_covariance_heatmap(mu_np, plot_dir / "latent_cov_eval.png")
    save_latent_pca_umap(mu_np, plot_dir / "latent_embed_eval.png")
    if first_x is not None:
        save_recon_grid_15ch(first_x, first_xhat, plot_dir / "recon_eval.png", n_channels=6)

    print("Saved:", metrics_dir / "metrics.json")


if __name__ == "__main__":
    main()
