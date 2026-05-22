from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.data.latent_dataset import LatentNPZDataset
from scripts.models.latent_dynamics import LatentDynamicsMLP, latent_dynamics_loss
from scripts.vae_15x2d import ConvVAE2D
from scripts.plotting.plot_dynamics import save_hist_triplet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-npz", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--dyn-ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--n-plots", type=int, default=8)
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
    model = LatentDynamicsMLP()
    model.load_state_dict(dyn_ckpt["state_dict"], strict=True)
    model.to(device).eval()

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(vae_ckpt.get("bins", 64))
    latent_dim = int(vae_ckpt.get("latent_dim", 256))

    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=latent_dim)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    mse_sum = 0.0
    n = 0
    plotted = 0

    for batch in dl:
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        mu = batch["mu"].to(device)

        z1p = model(z0, mu)
        loss, _ = latent_dynamics_loss(z1p, z1)

        B = z0.shape[0]
        mse_sum += float(loss.item()) * B
        n += B

        if plotted < args.n_plots:
            k = min(args.n_plots - plotted, B)
            y_truth = vae.decode(z1[:k]).detach().cpu().numpy()
            y_pred = vae.decode(z1p[:k]).detach().cpu().numpy()
            for i in range(k):
                save_hist_triplet(y_truth[i], y_pred[i], plot_dir / f"truth_vs_pred_{plotted:03d}.png", n_channels=6)
                plotted += 1
                if plotted >= args.n_plots:
                    break

    metrics = {
        "split": args.split,
        "n_samples": int(n),
        "latent_mse": float(mse_sum / max(n, 1)),
        "latent_npz": args.latent_npz,
        "vae_ckpt": args.vae_ckpt,
        "dyn_ckpt": args.dyn_ckpt,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", metrics_dir / "metrics.json")


if __name__ == "__main__":
    main()