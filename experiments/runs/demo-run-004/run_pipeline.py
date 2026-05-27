from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.tokenizer import ElementTokenizer, MuNormalizer
from scripts.tracking_transformer import TrackingTransformer
from scripts.vae_15x2d import ConvVAE2D, cloud6d_to_15x2d_hist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--dyn-ckpt", type=str, required=True, help="trained dynamics checkpoint")
    p.add_argument("--out", type=str, default="./runs/infer")

    p.add_argument("--lattice-npz", type=str, required=True, help="npz with MU_seq (T,3) and L_seq (T,) or s_seq (T,)")
    p.add_argument("--cloud-npz", type=str, required=True, help="npz with key cloud (Np,6)")

    p.add_argument("--max-T", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    bins = int(vae_ckpt.get("bins", 64))
    latent_dim = int(vae_ckpt.get("latent_dim", 256))
    global_ranges = tuple((float(a), float(b)) for (a, b) in vae_ckpt["global_ranges"])

    vae = ConvVAE2D(in_channels=15, bins=bins, latent_dim=latent_dim)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    vae.to(device).eval()

    dyn_ckpt = torch.load(args.dyn_ckpt, map_location="cpu")
    dargs = dyn_ckpt.get("args", {})

    mu_norm = MuNormalizer.from_state_dict(dyn_ckpt["mu_normalizer"]).to(device)
    tok = ElementTokenizer(mu_dim=3, d_token=512, n_pos_freqs=16, mu_normalizer=mu_norm)
    tok.load_state_dict(dyn_ckpt["tokenizer_state"], strict=True)
    tok.to(device).eval()

    model = TrackingTransformer(
        z_dim=latent_dim,
        h_dim=512,
        d_model=int(dargs.get("d_model", 512)),
        n_layers=int(dargs.get("n_layers", 4)),
        n_heads=int(dargs.get("n_heads", 8)),
        dropout=float(dargs.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(dyn_ckpt["state_dict"], strict=True)
    model.eval()

    lat = np.load(args.lattice_npz)
    MU_seq = lat["MU_seq"].astype(np.float32)
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

    cloud = np.load(args.cloud_npz)["cloud"].astype(np.float32)
    x = cloud6d_to_15x2d_hist(cloud, bins=bins, ranges=global_ranges)
    x = torch.from_numpy(x)[None, ...].to(device)

    mu0, _ = vae.encode(x)
    z0 = mu0

    MU_t = torch.from_numpy(MU_seq)[None, ...].to(device)
    s_t = torch.from_numpy(s_seq)[None, ...].to(device)
    h = tok(MU_t, s_t)

    z_seq = model(z0, h)

    xhat_seq = []
    for t in range(z_seq.shape[1]):
        xhat = vae.decode(z_seq[:, t, :])
        xhat_seq.append(xhat.detach().cpu().numpy())
    xhat_seq = np.concatenate(xhat_seq, axis=0)

    np.savez(outdir / "predicted_distributions.npz", Xhat=xhat_seq, MU_seq=MU_seq, s_seq=s_seq)
    print("Saved", outdir / "predicted_distributions.npz")


if __name__ == "__main__":
    main()
