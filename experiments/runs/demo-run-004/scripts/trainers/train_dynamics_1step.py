from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from scripts.data.latent_dataset import LatentNPZDataset
from scripts.models.latent_dynamics import latent_dynamics_loss, temporal_correlation, wasserstein_1d
from scripts.plotting.plot_dynamics import save_attention_map, save_dynamics_curves
from scripts.tokenizer import ElementTokenizer, MuNormalizer
from scripts.tracking_transformer import TrackingTransformer


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-npz", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=8)
    return p.parse_args()


def _eval_epoch(model, tok, dl, device, rollout_steps: int) -> Dict[str, float]:
    model.eval()
    tok.eval()

    loss_sum = mse_sum = mae_sum = rel_sum = 0.0
    n = 0

    dz_true_all = []
    dz_pred_all = []
    w1_all = []
    drift_all = []

    with torch.no_grad():
        for batch in dl:
            z0 = batch["z0"].to(device)
            z1 = batch["z1"].to(device)
            mu = batch["mu"].to(device)
            s = batch["s"].to(device)

            h = tok(mu[:, None, :], s[:, None])
            z_seq = model(z0, h)
            z1p = z_seq[:, 1, :]

            loss, logs = latent_dynamics_loss(z1p, z1)

            B = z0.shape[0]
            n += B
            loss_sum += float(loss.item()) * B
            mse_sum += float(logs["mse"].item()) * B
            mae_sum += float(logs["mae"].item()) * B
            rel_sum += float(logs["rel_l2"].item()) * B

            dz_true = (z1 - z0).detach().cpu().numpy()
            dz_pred = (z1p - z0).detach().cpu().numpy()
            dz_true_all.append(dz_true)
            dz_pred_all.append(dz_pred)
            w1_all.append(wasserstein_1d(dz_true, dz_pred))

            zt = z0
            drift_sample = []
            for _ in range(rollout_steps):
                zt = model(zt, h)[:, 1, :]
                drift_sample.append(torch.norm(zt - z0, dim=-1).mean().item())
            drift_all.append(float(np.mean(drift_sample)))

    dz_true_np = np.concatenate(dz_true_all, axis=0) if dz_true_all else np.zeros((0, 1), dtype=np.float32)
    dz_pred_np = np.concatenate(dz_pred_all, axis=0) if dz_pred_all else np.zeros((0, 1), dtype=np.float32)

    return {
        "loss": float(loss_sum / max(n, 1)),
        "mse": float(mse_sum / max(n, 1)),
        "mae": float(mae_sum / max(n, 1)),
        "rel_l2": float(rel_sum / max(n, 1)),
        "temporal_corr": temporal_correlation(dz_true_np, dz_pred_np),
        "wasserstein": float(np.mean(w1_all)) if w1_all else 0.0,
        "drift": float(np.mean(drift_all)) if drift_all else 0.0,
        "n": int(n),
    }


def main():
    args = parse_args()
    seed_all(args.seed)

    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    plot_dir = outdir / "plots"
    metrics_dir = outdir / "metrics"
    tb_dir = outdir / "tensorboard"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = LatentNPZDataset(args.latent_npz, split="train")
    val_ds = LatentNPZDataset(args.latent_npz, split="val")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    lat_raw = np.load(args.latent_npz, allow_pickle=True)
    mu_mean = lat_raw["mu_mean"].astype(np.float32)
    mu_std = lat_raw["mu_std"].astype(np.float32)

    mu_norm = MuNormalizer(mean=torch.from_numpy(mu_mean), std=torch.from_numpy(mu_std)).to(device)
    tok = ElementTokenizer(mu_dim=3, d_token=512, n_pos_freqs=16, mu_normalizer=mu_norm).to(device)

    model = TrackingTransformer(
        z_dim=256,
        h_dim=512,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(tok.parameters()), lr=args.lr)
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else _NoOpWriter()

    history = {
        "train_loss": [], "train_mse": [], "train_mae": [], "train_rel_l2": [],
        "val_loss": [], "val_mse": [], "val_mae": [], "val_rel_l2": [],
        "val_temporal_corr": [], "val_wasserstein": [], "val_drift": [],
    }

    global_step = 0

    for ep in range(args.epochs):
        model.train()
        tok.train()

        loss_sum = mse_sum = mae_sum = rel_sum = 0.0
        n = 0

        for batch in train_dl:
            z0 = batch["z0"].to(device)
            z1 = batch["z1"].to(device)
            mu = batch["mu"].to(device)
            s = batch["s"].to(device)

            # one-step target alignment: input z0 + token(mu,s_t) predicts z1
            h = tok(mu[:, None, :], s[:, None])
            z_seq = model(z0, h)
            z1p = z_seq[:, 1, :]

            loss, logs = latent_dynamics_loss(z1p, z1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            B = z0.shape[0]
            n += B
            loss_sum += float(loss.item()) * B
            mse_sum += float(logs["mse"].item()) * B
            mae_sum += float(logs["mae"].item()) * B
            rel_sum += float(logs["rel_l2"].item()) * B

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                writer.add_scalar("train/mse_step", float(logs["mse"].item()), global_step)
            global_step += 1

        tr = {
            "loss": float(loss_sum / max(n, 1)),
            "mse": float(mse_sum / max(n, 1)),
            "mae": float(mae_sum / max(n, 1)),
            "rel_l2": float(rel_sum / max(n, 1)),
        }

        va = _eval_epoch(model, tok, val_dl, device, rollout_steps=args.rollout_steps)

        history["train_loss"].append(tr["loss"])
        history["train_mse"].append(tr["mse"])
        history["train_mae"].append(tr["mae"])
        history["train_rel_l2"].append(tr["rel_l2"])
        history["val_loss"].append(va["loss"])
        history["val_mse"].append(va["mse"])
        history["val_mae"].append(va["mae"])
        history["val_rel_l2"].append(va["rel_l2"])
        history["val_temporal_corr"].append(va["temporal_corr"])
        history["val_wasserstein"].append(va["wasserstein"])
        history["val_drift"].append(va["drift"])

        writer.add_scalar("train/loss_epoch", tr["loss"], ep)
        writer.add_scalar("val/loss_epoch", va["loss"], ep)
        writer.add_scalar("val/temporal_corr", va["temporal_corr"], ep)
        writer.add_scalar("val/wasserstein", va["wasserstein"], ep)
        writer.add_scalar("val/drift", va["drift"], ep)

        print(
            f"[stage2][dyn] ep={ep:03d} train_mse={tr['mse']:.4e} val_mse={va['mse']:.4e} "
            f"val_corr={va['temporal_corr']:.4f} val_w1={va['wasserstein']:.4e}"
        )

        # save one attention map from first val batch
        with torch.no_grad():
            for batch in val_dl:
                z0 = batch["z0"].to(device)
                mu = batch["mu"].to(device)
                s = batch["s"].to(device)
                h = tok(mu[:, None, :], s[:, None])
                _, attn = model(z0[:1], h[:1], return_attention=True)
                save_attention_map(attn.detach().cpu().numpy(), plot_dir / f"attention_ep{ep:03d}.png")
                break

        torch.save(
            {
                "state_dict": model.state_dict(),
                "tokenizer_state": tok.state_dict(),
                "mu_normalizer": mu_norm.state_dict(),
                "epoch": ep,
                "args": vars(args),
            },
            ckpt_dir / f"dyn_ep{ep:03d}.pt",
        )

        (metrics_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_dynamics_curves(history, plot_dir / "loss_curves.png")

    writer.close()


if __name__ == "__main__":
    main()
