from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.tokenizer import ElementTokenizer, MuNormalizer
from scripts.tracking_transformer import TrackingTransformer
from scripts.vae_15x2d import ConvVAE2D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="./runs/infer")

    # lattice token inputs (npz)
    p.add_argument("--lattice-npz", type=str, required=True, help="npz with MU_seq (T,3) and L_seq (T,) or s_seq (T,)")

    # beam input
    p.add_argument("--cloud-npz", type=str, required=True, help="npz with key cloud (Np,6)")

    # model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--max-T", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load VAE
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(ckpt.get("bins", 64))
    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=256)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    # Load lattice token inputs
    lat = np.load(args.lattice_npz)
    MU_seq = lat["MU_seq"].astype(np.float32)  # (T,3)
    if "s_seq" in lat:
        s_seq = lat["s_seq"].astype(np.float32)
    elif "L_seq" in lat:
        L_seq = lat["L_seq"].astype(np.float32)
        s_seq = np.cumsum(np.concatenate([[0.0], L_seq[:-1]], axis=0))
    else:
        raise KeyError("lattice-npz must contain s_seq or L_seq")

    T = int(min(args.max_T, MU_seq.shape[0]))
    MU_seq = MU_seq[:T]
    s_seq = s_seq[:T]

    # Build tokenizer
    mu_mean = torch.tensor(MU_seq.mean(axis=0))
    mu_std = torch.tensor(MU_seq.std(axis=0) + 1e-12)
    normalizer = MuNormalizer(mean=mu_mean, std=mu_std)
    tok = ElementTokenizer(mu_dim=3, d_token=512, n_pos_freqs=16, mu_normalizer=normalizer).to(device).eval()

    # Tracking model (random init here; load your trained weights in practice)
    model = TrackingTransformer(z_dim=256, h_dim=512, d_model=args.d_model, n_layers=6, n_heads=8).to(device).eval()

    # Load cloud
    cloud = np.load(args.cloud_npz)["cloud"].astype(np.float32)  # (Np,6)
    # Convert to 15x2d hist
    from ml4coll.models.vae_15x2d import cloud6d_to_15x2d_hist
    x = cloud6d_to_15x2d_hist(cloud, bins=bins)
    x = torch.from_numpy(x)[None, ...].to(device)  # (1,15,bins,bins)

    # Encode -> z0
    mu, logvar = vae.encode(x)
    z0 = mu  # use mean for deterministic inference

    # Tokenize lattice
    MU_t = torch.from_numpy(MU_seq)[None, ...].to(device)  # (1,T,3)
    s_t = torch.from_numpy(s_seq)[None, ...].to(device)    # (1,T)
    h = tok(MU_t, s_t)  # (1,T,512)

    # Rollout z_seq
    z_seq = model(z0, h)  # (1,T+1,256)

    # Decode all
    xhat_seq = []
    for t in range(z_seq.shape[1]):
        xhat = vae.decode(z_seq[:, t, :])
        xhat_seq.append(xhat.detach().cpu().numpy())
    xhat_seq = np.concatenate(xhat_seq, axis=0)  # (T+1,15,bins,bins)

    np.savez(outdir / "predicted_distributions.npz", Xhat=xhat_seq, MU_seq=MU_seq, s_seq=s_seq)
    print("Saved", outdir / "predicted_distributions.npz")


if __name__ == "__main__":
    main()
