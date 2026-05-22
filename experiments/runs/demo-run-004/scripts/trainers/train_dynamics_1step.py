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
from scripts.models.latent_dynamics import LatentDynamicsMLP, latent_dynamics_loss
from scripts.plotting.plot_dynamics import save_loss_curves


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
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def eval_epoch(model, dl, device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in dl:
            z0 = batch["z0"].to(device)
            z1 = batch["z1"].to(device)
            mu = batch["mu"].to(device)
            z1p = model(z0, mu)
            loss, _ = latent_dynamics_loss(z1p, z1)
            B = z0.shape[0]
            loss_sum += float(loss.item()) * B
            n += B
    return {"mse": float(loss_sum / max(n, 1)), "n": int(n)}


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

    model = LatentDynamicsMLP(hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else _NoOpWriter()

    history = {"train_mse": [], "val_mse": []}
    global_step = 0

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n = 0

        for batch in train_dl:
            z0 = batch["z0"].to(device)
            z1 = batch["z1"].to(device)
            mu = batch["mu"].to(device)

            z1p = model(z0, mu)
            loss, _ = latent_dynamics_loss(z1p, z1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            B = z0.shape[0]
            loss_sum += float(loss.item()) * B
            n += B

            if global_step % args.log_every == 0:
                writer.add_scalar("train/mse_step", float(loss.item()), global_step)
            global_step += 1

        train_mse = float(loss_sum / max(n, 1))
        val_mse = float(eval_epoch(model, val_dl, device)["mse"])

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        writer.add_scalar("train/mse_epoch", train_mse, ep)
        writer.add_scalar("val/mse_epoch", val_mse, ep)

        print(f"[stage2][dyn] ep={ep:03d} train_mse={train_mse:.4e} val_mse={val_mse:.4e}")

        torch.save({"state_dict": model.state_dict(), "epoch": ep, "args": vars(args)}, ckpt_dir / f"dyn_ep{ep:03d}.pt")
        (metrics_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_loss_curves(history, plot_dir / "loss_curves.png", title="Stage 2: Latent dynamics MSE")

    writer.close()


if __name__ == "__main__":
    main()