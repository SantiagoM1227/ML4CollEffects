from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.npz_dataset import XsuiteNPZDataset, collate_cloud
from scripts.vae_tf import ConvVAE2D, cloud6d_to_15x2d_hist


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
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raw = np.load(args.data, allow_pickle=True)
    if "Y_cloud" not in raw.files:
        raise KeyError("Dataset missing Y_cloud; cannot export z1 for dynamics training.")

    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(ckpt.get("bins", args.bins))
    latent_dim = int(ckpt.get("latent_dim", 256))
    global_ranges = tuple((float(a), float(b)) for (a, b) in ckpt["global_ranges"])

    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=latent_dim)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    ds = XsuiteNPZDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cloud)

    z0_all, z1_all, mu_all, s_all = [], [], [], []

    offset = 0
    with torch.no_grad():
        for batch in dl:
            cloud_x = batch["cloud"].to(device)
            x15 = batch_cloud_to_hist(cloud_x, bins, global_ranges).to(device)
            mu_x, _ = vae.encode(x15)
            z0 = mu_x

            B = cloud_x.shape[0]
            y_cloud = raw["Y_cloud"][offset : offset + B].astype(np.float32)
            y15_list = [torch.from_numpy(cloud6d_to_15x2d_hist(y_cloud[b], bins=bins, ranges=global_ranges)) for b in range(B)]
            y15 = torch.stack(y15_list, dim=0).to(device)
            mu_y, _ = vae.encode(y15)
            z1 = mu_y

            z0_all.append(z0.detach().cpu().numpy().astype(np.float32))
            z1_all.append(z1.detach().cpu().numpy().astype(np.float32))
            mu_all.append(batch["mu"].cpu().numpy().astype(np.float32))
            s_all.append(np.zeros((B,), dtype=np.float32))

            offset += B

    z0_all = np.concatenate(z0_all, axis=0)
    z1_all = np.concatenate(z1_all, axis=0)
    mu_all = np.concatenate(mu_all, axis=0)
    s_all = np.concatenate(s_all, axis=0)

    train_idx = raw["train"] if "train" in raw.files else np.arange(z0_all.shape[0], dtype=np.int64)
    val_idx = raw["val"] if "val" in raw.files else np.array([], dtype=np.int64)
    test_idx = raw["test"] if "test" in raw.files else np.array([], dtype=np.int64)

    mu_mean = mu_all[train_idx].mean(axis=0) if train_idx.size else mu_all.mean(axis=0)
    mu_std = mu_all[train_idx].std(axis=0) + 1e-12 if train_idx.size else (mu_all.std(axis=0) + 1e-12)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        z0=z0_all,
        z1=z1_all,
        MU=mu_all,
        s=s_all,
        train=train_idx,
        val=val_idx,
        test=test_idx,
        vae_ckpt=str(args.vae_ckpt),
        bins=bins,
        global_ranges=np.asarray(global_ranges, dtype=np.float32),
        mu_mean=mu_mean.astype(np.float32),
        mu_std=mu_std.astype(np.float32),
    )

    print("Saved latent dataset:", out_path)
    print("z0:", z0_all.shape, "z1:", z1_all.shape, "MU:", mu_all.shape)


if __name__ == "__main__":
    main()
