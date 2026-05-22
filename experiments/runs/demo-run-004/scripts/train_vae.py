from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from npz_dataset import XsuiteNPZDataset, collate_cloud
from vae_15x2d import ConvVAE2D, cloud6d_to_15x2d_hist, vae_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    p.add_argument("--out", type=str, default="./runs/vae")
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def batch_cloud_to_hist(batch_cloud: torch.Tensor, bins: int) -> torch.Tensor:
    # batch_cloud: (B,Np,6)
    B = batch_cloud.shape[0]
    out = []
    for b in range(B):
        hist = cloud6d_to_15x2d_hist(batch_cloud[b].cpu().numpy(), bins=bins)
        out.append(torch.from_numpy(hist))
    x = torch.stack(out, dim=0)  # (B,15,bins,bins)
    return x


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = XsuiteNPZDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_cloud)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae = ConvVAE2D(in_channels=15, bins=args.bins, latent_dim=256).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    vae.train()
    step = 0
    for ep in range(args.epochs):
        for batch in dl:
            cloud = batch["cloud"].to(device)
            x = batch_cloud_to_hist(cloud, args.bins).to(device)

            xhat, mu, logvar, z = vae(x)
            loss, logs = vae_loss(xhat, x, mu, logvar, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 50 == 0:
                print(f"[vae] ep={ep} step={step} loss={loss.item():.4e} recon={logs['recon'].item():.4e} kl={logs['kl'].item():.4e}")
            step += 1

        ckpt = {
            "state_dict": vae.state_dict(),
            "bins": args.bins,
            "latent_dim": 256,
        }
        torch.save(ckpt, outdir / f"vae_ep{ep:03d}.pt")


if __name__ == "__main__":
    main()
