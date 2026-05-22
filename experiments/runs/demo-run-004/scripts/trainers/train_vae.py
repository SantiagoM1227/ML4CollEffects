from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from scripts.npz_dataset import XsuiteNPZDataset, collate_cloud
from scripts.vae_15x2d import ConvVAE2D, cloud6d_to_15x2d_hist, vae_loss
from scripts.plotting.plot_vae import save_latent_histograms, save_loss_curves, save_recon_grid_15ch


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def batch_cloud_to_hist(batch_cloud: torch.Tensor, bins: int) -> torch.Tensor:
    out = []
    for b in range(batch_cloud.shape[0]):
        out.append(torch.from_numpy(cloud6d_to_15x2d_hist(batch_cloud[b].cpu().numpy(), bins=bins)))
    return torch.stack(out, dim=0)


def split_subsets(npz_path: str, base_ds: XsuiteNPZDataset) -> Tuple[Subset, Subset]:
    raw = np.load(npz_path, allow_pickle=True)
    if "train" in raw.files and "val" in raw.files:
        train_idx = raw["train"].astype(np.int64)
        val_idx = raw["val"].astype(np.int64)
    else:
        n = len(base_ds)
        idx = np.arange(n)
        n_train = int(0.8 * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
    return Subset(base_ds, train_idx.tolist()), Subset(base_ds, val_idx.tolist())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    plot_dir = outdir / "plots"
    metrics_dir = outdir / "metrics"
    tb_dir = outdir / "tensorboard"
    latent_dir = outdir / "latent"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = XsuiteNPZDataset(args.data)
    train_ds, val_ds = split_subsets(args.data, base)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_cloud)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cloud)

    vae = ConvVAE2D(in_channels=15, bins=args.bins, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else _NoOpWriter()

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [],
    }

    global_step = 0

    for ep in range(args.epochs):
        vae.train()
        tr_loss = tr_recon = tr_kl = 0.0
        tr_n = 0

        for batch in train_dl:
            x = batch_cloud_to_hist(batch["cloud"].to(device), args.bins).to(device)
            xhat, mu, logvar, z = vae(x)
            loss, logs = vae_loss(xhat, x, mu, logvar, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            B = x.shape[0]
            tr_loss += float(loss.item()) * B
            tr_recon += float(logs["recon"].item()) * B
            tr_kl += float(logs["kl"].item()) * B
            tr_n += B

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/recon_step", logs["recon"].item(), global_step)
                writer.add_scalar("train/kl_step", logs["kl"].item(), global_step)
            global_step += 1

        tr_loss /= max(tr_n, 1)
        tr_recon /= max(tr_n, 1)
        tr_kl /= max(tr_n, 1)

        vae.eval()
        va_loss = va_recon = va_kl = 0.0
        va_n = 0

        latent_mu_list = []
        latent_logvar_list = []
        first_x = None
        first_xhat = None

        with torch.no_grad():
            for batch in val_dl:
                x = batch_cloud_to_hist(batch["cloud"].to(device), args.bins).to(device)
                xhat, mu, logvar, z = vae(x)
                loss, logs = vae_loss(xhat, x, mu, logvar, beta=args.beta)

                B = x.shape[0]
                va_loss += float(loss.item()) * B
                va_recon += float(logs["recon"].item()) * B
                va_kl += float(logs["kl"].item()) * B
                va_n += B

                latent_mu_list.append(mu.detach().cpu().numpy())
                latent_logvar_list.append(logvar.detach().cpu().numpy())

                if first_x is None:
                    first_x = x[0].detach().cpu().numpy()
                    first_xhat = xhat[0].detach().cpu().numpy()

        va_loss /= max(va_n, 1)
        va_recon /= max(va_n, 1)
        va_kl /= max(va_n, 1)

        history["train_loss"].append(tr_loss)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["val_loss"].append(va_loss)
        history["val_recon"].append(va_recon)
        history["val_kl"].append(va_kl)

        writer.add_scalar("train/loss_epoch", tr_loss, ep)
        writer.add_scalar("train/recon_epoch", tr_recon, ep)
        writer.add_scalar("train/kl_epoch", tr_kl, ep)
        writer.add_scalar("val/loss_epoch", va_loss, ep)
        writer.add_scalar("val/recon_epoch", va_recon, ep)
        writer.add_scalar("val/kl_epoch", va_kl, ep)

        print(
            f"[stage1][vae] ep={ep:03d} "
            f"train_loss={tr_loss:.4e} val_loss={va_loss:.4e} "
            f"train_recon={tr_recon:.4e} val_recon={va_recon:.4e} "
            f"train_kl={tr_kl:.4e} val_kl={va_kl:.4e}"
        )

        torch.save(
            {"state_dict": vae.state_dict(), "bins": args.bins, "latent_dim": args.latent_dim, "epoch": ep, "args": vars(args)},
            ckpt_dir / f"vae_ep{ep:03d}.pt",
        )

        (metrics_dir / "history.json").write_text(json.dumps(history, indent=2))

        save_loss_curves(
            {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "train_recon": history["train_recon"],
                "val_recon": history["val_recon"],
                "train_kl": history["train_kl"],
                "val_kl": history["val_kl"],
            },
            plot_dir / "loss_curves.png",
            title="Stage 1: VAE losses",
        )

        if first_x is not None:
            save_recon_grid_15ch(first_x, first_xhat, plot_dir / f"recon_ep{ep:03d}.png", n_channels=6)

        mu_np = np.concatenate(latent_mu_list, axis=0) if latent_mu_list else np.zeros((0, args.latent_dim), dtype=np.float32)
        logvar_np = np.concatenate(latent_logvar_list, axis=0) if latent_logvar_list else np.zeros((0, args.latent_dim), dtype=np.float32)
        save_latent_histograms(mu_np, logvar_np, plot_dir / f"latent_hist_ep{ep:03d}.png")

        stats = {
            "epoch": ep,
            "mu_mean": float(mu_np.mean()) if mu_np.size else 0.0,
            "mu_std": float(mu_np.std()) if mu_np.size else 0.0,
            "logvar_mean": float(logvar_np.mean()) if logvar_np.size else 0.0,
            "logvar_std": float(logvar_np.std()) if logvar_np.size else 0.0,
        }
        (metrics_dir / f"latent_stats_ep{ep:03d}.json").write_text(json.dumps(stats, indent=2))

    writer.close()


if __name__ == "__main__":
    main()