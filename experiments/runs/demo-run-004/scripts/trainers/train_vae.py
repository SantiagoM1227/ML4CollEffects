from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from scripts.npz_dataset import XsuiteNPZDataset, collate_cloud
from scripts.plotting.plot_vae import (
    save_gradient_norms,
    save_kl_recon_curve,
    save_latent_covariance_heatmap,
    save_latent_histograms,
    save_latent_pca_umap,
    save_loss_curves,
    save_mu_logvar_evolution,
    save_recon_grid_15ch,
)
from scripts.vae_15x2d import ConvVAE2D, cloud6d_to_15x2d_hist, compute_global_ranges_6d, vae_loss


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


@torch.no_grad()
def batch_cloud_to_hist(batch_cloud: torch.Tensor, bins: int, ranges) -> torch.Tensor:
    out = []
    for b in range(batch_cloud.shape[0]):
        h = cloud6d_to_15x2d_hist(batch_cloud[b].cpu().numpy(), bins=bins, ranges=ranges)
        out.append(torch.from_numpy(h))
    return torch.stack(out, dim=0)


def _ssim_global(x: np.ndarray, y: np.ndarray) -> float:
    # x,y: (B,C,H,W)
    c1 = 1e-4
    c2 = 9e-4
    vals = []
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            a = x[b, c]
            z = y[b, c]
            mu_a = float(a.mean())
            mu_z = float(z.mean())
            var_a = float(a.var())
            var_z = float(z.var())
            cov = float(((a - mu_a) * (z - mu_z)).mean())
            num = (2 * mu_a * mu_z + c1) * (2 * cov + c2)
            den = (mu_a**2 + mu_z**2 + c1) * (var_a + var_z + c2)
            vals.append(num / (den + 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def _kl_js_hist(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    p = np.clip(x, eps, None)
    q = np.clip(y, eps, None)
    m = 0.5 * (p + q)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=(-1, -2))
    js = 0.5 * np.sum(p * (np.log(p) - np.log(m)), axis=(-1, -2)) + 0.5 * np.sum(q * (np.log(q) - np.log(m)), axis=(-1, -2))
    return float(np.mean(kl)), float(np.mean(js))


def _wasserstein_flat(x: np.ndarray, y: np.ndarray) -> float:
    # Approximate 1D Wasserstein over flattened pixel index CDF per channel
    x1 = np.sort(x.reshape(x.shape[0], x.shape[1], -1), axis=-1)
    y1 = np.sort(y.reshape(y.shape[0], y.shape[1], -1), axis=-1)
    return float(np.mean(np.abs(x1 - y1)))


def _spectral_error(x: np.ndarray, y: np.ndarray) -> float:
    fx = np.fft.rfft2(x, axes=(-2, -1))
    fy = np.fft.rfft2(y, axes=(-2, -1))
    num = np.linalg.norm(np.abs(fx - fy))
    den = np.linalg.norm(np.abs(fx)) + 1e-12
    return float(num / den)


def compute_recon_metrics(x: torch.Tensor, xhat: torch.Tensor) -> Dict[str, float]:
    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = xhat.detach().cpu().numpy().astype(np.float64)
    err = y_np - x_np
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(x_np) + 1e-12))
    ssim = _ssim_global(x_np, y_np)
    kl_hist, js_hist = _kl_js_hist(x_np, y_np)
    wass = _wasserstein_flat(x_np, y_np)
    spec = _spectral_error(x_np, y_np)
    return {
        "mse": mse,
        "mae": mae,
        "rel_l2": rel_l2,
        "ssim": ssim,
        "kl_hist": kl_hist,
        "js_hist": js_hist,
        "wasserstein": wass,
        "spectral_error": spec,
    }


def _grad_norm(model: torch.nn.Module) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            tot += float(torch.sum(g * g).item())
    return float(np.sqrt(max(tot, 0.0)))


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
    p.add_argument("--beta-warmup-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--range-lo-pct", type=float, default=0.5)
    p.add_argument("--range-hi-pct", type=float, default=99.5)
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
    global_ranges = compute_global_ranges_6d(base.X_cloud, lo_pct=args.range_lo_pct, hi_pct=args.range_hi_pct)
    (metrics_dir / "global_ranges.json").write_text(json.dumps([[float(a), float(b)] for (a, b) in global_ranges], indent=2))

    train_ds, val_ds = split_subsets(args.data, base)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_cloud)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cloud)

    vae = ConvVAE2D(in_channels=15, bins=args.bins, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else _NoOpWriter()

    history: Dict[str, list] = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [],
        "val_mse": [], "val_mae": [], "val_rel_l2": [], "val_ssim": [],
        "val_kl_hist": [], "val_js_hist": [], "val_wasserstein": [], "val_spectral_error": [],
        "beta_eff": [],
    }
    mu_means, logvar_means, grad_norms = [], [], []

    global_step = 0

    for ep in range(args.epochs):
        vae.train()
        tr_loss = tr_recon = tr_kl = 0.0
        tr_n = 0

        beta_eff = args.beta * min(1.0, float(ep + 1) / max(args.beta_warmup_epochs, 1))

        for batch in train_dl:
            x = batch_cloud_to_hist(batch["cloud"].to(device), args.bins, global_ranges).to(device)
            xhat, mu, logvar, _ = vae(x)
            loss, logs = vae_loss(xhat, x, mu, logvar, beta=beta_eff)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = _grad_norm(vae)
            grad_norms.append(gn)
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
                writer.add_scalar("train/grad_norm_step", gn, global_step)
            global_step += 1

        tr_loss /= max(tr_n, 1)
        tr_recon /= max(tr_n, 1)
        tr_kl /= max(tr_n, 1)

        vae.eval()
        va_loss = va_recon = va_kl = 0.0
        va_n = 0
        val_metrics_acc = {k: 0.0 for k in ["mse", "mae", "rel_l2", "ssim", "kl_hist", "js_hist", "wasserstein", "spectral_error"]}

        latent_mu_list = []
        latent_logvar_list = []
        first_x = None
        first_xhat = None

        with torch.no_grad():
            for batch in val_dl:
                x = batch_cloud_to_hist(batch["cloud"].to(device), args.bins, global_ranges).to(device)
                xhat, mu, logvar, _ = vae(x)
                loss, logs = vae_loss(xhat, x, mu, logvar, beta=beta_eff)

                B = x.shape[0]
                va_loss += float(loss.item()) * B
                va_recon += float(logs["recon"].item()) * B
                va_kl += float(logs["kl"].item()) * B
                va_n += B

                m = compute_recon_metrics(x, xhat)
                for k in val_metrics_acc:
                    val_metrics_acc[k] += float(m[k]) * B

                latent_mu_list.append(mu.detach().cpu().numpy())
                latent_logvar_list.append(logvar.detach().cpu().numpy())

                if first_x is None:
                    first_x = x[0].detach().cpu().numpy()
                    first_xhat = xhat[0].detach().cpu().numpy()

        va_loss /= max(va_n, 1)
        va_recon /= max(va_n, 1)
        va_kl /= max(va_n, 1)

        for k in val_metrics_acc:
            val_metrics_acc[k] /= max(va_n, 1)

        history["train_loss"].append(tr_loss)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["val_loss"].append(va_loss)
        history["val_recon"].append(va_recon)
        history["val_kl"].append(va_kl)
        history["val_mse"].append(val_metrics_acc["mse"])
        history["val_mae"].append(val_metrics_acc["mae"])
        history["val_rel_l2"].append(val_metrics_acc["rel_l2"])
        history["val_ssim"].append(val_metrics_acc["ssim"])
        history["val_kl_hist"].append(val_metrics_acc["kl_hist"])
        history["val_js_hist"].append(val_metrics_acc["js_hist"])
        history["val_wasserstein"].append(val_metrics_acc["wasserstein"])
        history["val_spectral_error"].append(val_metrics_acc["spectral_error"])
        history["beta_eff"].append(beta_eff)

        mu_np = np.concatenate(latent_mu_list, axis=0) if latent_mu_list else np.zeros((0, args.latent_dim), dtype=np.float32)
        logvar_np = np.concatenate(latent_logvar_list, axis=0) if latent_logvar_list else np.zeros((0, args.latent_dim), dtype=np.float32)
        mu_mean = float(mu_np.mean()) if mu_np.size else 0.0
        logvar_mean = float(logvar_np.mean()) if logvar_np.size else 0.0
        mu_means.append(mu_mean)
        logvar_means.append(logvar_mean)

        writer.add_scalar("train/loss_epoch", tr_loss, ep)
        writer.add_scalar("train/recon_epoch", tr_recon, ep)
        writer.add_scalar("train/kl_epoch", tr_kl, ep)
        writer.add_scalar("val/loss_epoch", va_loss, ep)
        writer.add_scalar("val/recon_epoch", va_recon, ep)
        writer.add_scalar("val/kl_epoch", va_kl, ep)
        for k, v in val_metrics_acc.items():
            writer.add_scalar(f"val/{k}", v, ep)
        writer.add_scalar("train/beta_eff", beta_eff, ep)

        print(
            f"[stage1][vae] ep={ep:03d} train_loss={tr_loss:.4e} val_loss={va_loss:.4e} "
            f"val_mse={val_metrics_acc['mse']:.4e} val_ssim={val_metrics_acc['ssim']:.4f} "
            f"val_js={val_metrics_acc['js_hist']:.4e} beta={beta_eff:.3e}"
        )

        torch.save(
            {
                "state_dict": vae.state_dict(),
                "bins": args.bins,
                "latent_dim": args.latent_dim,
                "epoch": ep,
                "args": vars(args),
                "global_ranges": [[float(a), float(b)] for (a, b) in global_ranges],
            },
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
        save_kl_recon_curve(history, plot_dir / "kl_vs_recon.png")
        save_gradient_norms(grad_norms, plot_dir / "grad_norms.png")
        save_mu_logvar_evolution(mu_means, logvar_means, plot_dir / "mu_logvar_evolution.png")

        if first_x is not None:
            save_recon_grid_15ch(first_x, first_xhat, plot_dir / f"recon_ep{ep:03d}.png", n_channels=6)

        save_latent_histograms(mu_np, logvar_np, plot_dir / f"latent_hist_ep{ep:03d}.png")
        save_latent_covariance_heatmap(mu_np, plot_dir / f"latent_cov_ep{ep:03d}.png")
        save_latent_pca_umap(mu_np, plot_dir / f"latent_embed_ep{ep:03d}.png")

        stats = {
            "epoch": ep,
            "mu_mean": mu_mean,
            "mu_std": float(mu_np.std()) if mu_np.size else 0.0,
            "logvar_mean": logvar_mean,
            "logvar_std": float(logvar_np.std()) if logvar_np.size else 0.0,
            "val_metrics": val_metrics_acc,
            "beta_eff": beta_eff,
        }
        (metrics_dir / f"latent_stats_ep{ep:03d}.json").write_text(json.dumps(stats, indent=2))

    writer.close()


if __name__ == "__main__":
    main()
