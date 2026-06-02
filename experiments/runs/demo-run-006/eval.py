# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless for sbatch
import matplotlib.pyplot as plt


# ---------------------------
# Shared utilities (keep consistent with train.py)
# ---------------------------
def load_npz(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def beam_centroid_sigma(cloud: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    centroid = cloud.mean(dim=1)
    var = (cloud - centroid[:, None, :]).pow(2).mean(dim=1)
    sigma = torch.sqrt(var + eps)
    return centroid, sigma


PAIR_INDEX = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]
PHASE_SPACE_LABELS = ("x", "y", "zeta", "px", "py", "delta")


def _hist2d_bucketize(x, y, x_edges, y_edges) -> torch.Tensor:
    B, Np = x.shape
    nb = x_edges.numel() - 1
    ix = torch.bucketize(x, x_edges) - 1
    iy = torch.bucketize(y, y_edges) - 1
    ix = ix.clamp(0, nb - 1)
    iy = iy.clamp(0, nb - 1)
    flat = ix * nb + iy
    hist = torch.zeros(B, nb * nb, device=x.device, dtype=torch.float32)
    ones = torch.ones_like(flat, dtype=torch.float32)
    hist.scatter_add_(1, flat, ones)
    return hist.view(B, nb, nb)


def cloud_to_pairwise_hists(cloud: torch.Tensor, nbins: int = 64, clip_k: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
    B, Np, D = cloud.shape
    assert D == 6

    centroid, sigma = beam_centroid_sigma(cloud, eps=eps)
    t = torch.linspace(0.0, 1.0, nbins + 1, device=cloud.device, dtype=torch.float32)[None, :]
    edges = []
    for d in range(6):
        lo = centroid[:, d] - clip_k * sigma[:, d]
        hi = centroid[:, d] + clip_k * sigma[:, d]
        hi = torch.where((hi - lo) < 1e-9, lo + 1e-9, hi)
        ed = lo[:, None] * (1 - t) + hi[:, None] * t
        edges.append(ed)

    out = []
    for (i, j) in PAIR_INDEX:
        xi = cloud[:, :, i].to(torch.float32)
        yj = cloud[:, :, j].to(torch.float32)
        h = torch.zeros(B, nbins, nbins, device=cloud.device, dtype=torch.float32)
        for b in range(B):
            hb = _hist2d_bucketize(xi[b:b+1], yj[b:b+1], edges[i][b], edges[j][b])
            h[b] = hb[0]
        h = h / (float(Np) + eps)
        out.append(h)
    return torch.stack(out, dim=1)


def make_dummy_lattice_tokens(B: int, N: int, device: torch.device):
    elem_params = torch.zeros(B, N, 7, device=device, dtype=torch.float32)
    elem_params[..., 0] = 1.0
    elem_s = torch.cumsum(elem_params[..., 0], dim=1) - elem_params[..., 0]
    return elem_params, elem_s


def vae_recon_loss_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(dim=1)


def mse_per_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="none").flatten(1).mean(dim=1)


def mae_per_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b).flatten(1).mean(dim=1)


# ---------------------------
# Plotting
# ---------------------------
def plot_sigma_centroid_scatter(sigma_true, sigma_pred, cent_true, cent_pred, out_path: str, title_prefix=""):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)

    for d in range(6):
        ax = axes[0, d]
        x = sigma_true[:, d]
        y = sigma_pred[:, d]
        ax.scatter(x, y, s=8, alpha=0.6)
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{title_prefix} sigma({PHASE_SPACE_LABELS[d]})")
        ax.set_xlabel("truth"); ax.set_ylabel("pred"); ax.grid(alpha=0.2)

    for d in range(6):
        ax = axes[1, d]
        x = cent_true[:, d]
        y = cent_pred[:, d]
        ax.scatter(x, y, s=8, alpha=0.6)
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{title_prefix} <{PHASE_SPACE_LABELS[d]}>")
        ax.set_xlabel("truth"); ax.set_ylabel("pred"); ax.grid(alpha=0.2)

    fig.suptitle(f"{title_prefix} sigma/centroid scatter")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hist_triplet(truth_map, pred_map, out_path: str, title="", vmax=None):
    diff = pred_map - truth_map
    vmax = vmax if vmax is not None else float(np.percentile(truth_map, 99.5) + 1e-12)
    vmax = max(vmax, float(np.percentile(pred_map, 99.5) + 1e-12))
    dv = float(np.percentile(np.abs(diff), 99.5) + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    im0 = axes[0].imshow(truth_map, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
    axes[0].set_title("Truth"); plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_map, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
    axes[1].set_title("Pred"); plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, origin="lower", aspect="auto", vmin=-dv, vmax=dv, cmap="RdBu_r")
    axes[2].set_title("Pred - Truth"); plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_triplets_for_sample(truth_15, pred_15, sample_idx: int, pairs_to_plot, plots_dir: str, prefix: str):
    """
    truth_15, pred_15: [B,15,64,64] numpy arrays
    prefix: e.g. "vaeX" or "dynY"
    """
    for pidx in pairs_to_plot:
        i, j = PAIR_INDEX[pidx]
        title = f"{prefix} sample{sample_idx} pair ({PHASE_SPACE_LABELS[i]}, {PHASE_SPACE_LABELS[j]})"
        out_path = os.path.join(plots_dir, f"{prefix}_s{sample_idx}_p{pidx}_{PHASE_SPACE_LABELS[i]}_{PHASE_SPACE_LABELS[j]}.png")
        plot_hist_triplet(
            truth_map=truth_15[sample_idx, pidx],
            pred_map=pred_15[sample_idx, pidx],
            out_path=out_path,
            title=title,
        )
# ---------------------------
# Load model definitions from train.py
# ---------------------------
# Easiest robust approach: import the classes directly from train.py
# (so you don't duplicate code).
from train_vae_SSNO import BeamLatentTrackingModel, MUPreprocessor, MUPreprocessConfig  # noqa: E402 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vae_ckpt", required=True, help="Path to VAE checkpoint saved by train.py --mode vae")
    ap.add_argument("--dyn_ckpt", required=True, help="Path to dynamics checkpoint saved by train.py --mode dynamics")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_elem", type=int, default=32)
    ap.add_argument("--pairs_to_plot", nargs="+", type=int, default=[0, 1, 10])
    ap.add_argument("--num_plot_batches", type=int, default=1, help="How many val batches to plot from")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data = load_npz(args.data)

    # Build val loader (no torch Dataset class needed; keep simple)
    if "val" in data:
        val_idx = data["val"].astype(np.int64)
    else:
        val_idx = np.arange(data["X_cloud"].shape[0], dtype=np.int64)

    # Prepare MU preprocessor and dimensions based on checkpoint
    vae_ck = torch.load(args.vae_ckpt, map_location="cpu")
    dyn_ck = torch.load(args.dyn_ckpt, map_location="cpu")

    mu_dim = int(data["MU"].shape[-1])

    # restore MU preprocessor state from dynamics ckpt if present, else from vae ckpt
    if "mu_prep_state_dict" in dyn_ck:
        # Create a matching preprocessor config. We only need correct m (mu_dim).
        cfg = MUPreprocessConfig(specs=[1.0] * mu_dim, standardize=True, clip=10.0)
        mu_prep = MUPreprocessor(cfg).to(device)
        mu_prep.load_state_dict(dyn_ck["mu_prep_state_dict"], strict=False)
    else:
        cfg = MUPreprocessConfig(specs=[1.0] * mu_dim, standardize=True, clip=10.0)
        mu_prep = MUPreprocessor(cfg).to(device)

    mu_prep.eval()

    # If the checkpoint didn't contain a fitted preprocessor, fit it now from the dataset.
    if not getattr(mu_prep, "fitted", False):
        if "train" in data:
            mu_train_np = data["MU"][data["train"]]
        else:
            mu_train_np = data["MU"]
        mu_prep.fit(mu_train_np)

    # Instantiate model and load dynamics weights (full state_dict)
    model = BeamLatentTrackingModel(mu_dim=mu_dim, latent_dim=256, d_model=512).to(device)
    model.eval()

    if "state_dict" in dyn_ck:
        model.load_state_dict(dyn_ck["state_dict"], strict=False)
    else:
        # if user saved differently
        model.load_state_dict(dyn_ck, strict=False)

    # Ensure VAE weights match the VAE-only checkpoint (optional, but helps if dynamics ckpt didn't include latest VAE)
    if "vae_state_dict" in vae_ck:
        model.vae.load_state_dict(vae_ck["vae_state_dict"], strict=False)

    # Metrics accumulators
    metrics = {
        "vae": {"recon0": [], "sigma_mse": [], "centroid_mse": [], "sigma_mae": [], "centroid_mae": []},
        "dyn": {"reconN": [], "sigma_mse": [], "centroid_mse": [], "sigma_mae": [], "centroid_mae": []},
    }

    # Iterate val in batches
    bs = args.batch_size
    n_batches = int(np.ceil(len(val_idx) / bs))

    plot_batches_left = args.num_plot_batches

    for bi in range(n_batches):
        sl = val_idx[bi * bs : (bi + 1) * bs]
        X = torch.from_numpy(data["X_cloud"][sl]).float().to(device)  # [B,Np,6]
        Y = torch.from_numpy(data["Y_cloud"][sl]).float().to(device)
        MU = torch.from_numpy(data["MU"][sl]).float().to(device)

        MUe = mu_prep(MU)

        B = X.shape[0]
        elem_params, elem_s = make_dummy_lattice_tokens(B, args.n_elem, device)

        with torch.no_grad():
            out = model(X, MUe, elem_params, elem_s, mode="AR")

        # ---- VAE metrics (input snapshot X) ----
        x_hist = out["x_hist"]
        x0_hat = out["vae"]["x_hat"]

        recon0_b = vae_recon_loss_mse(x0_hat, x_hist)
        cent_x, sig_x = beam_centroid_sigma(X)
        sig0_hat = out["vae"]["sigma_hat"]
        cent0_hat = out["vae"]["centroid_hat"]

        metrics["vae"]["recon0"].extend(recon0_b.detach().cpu().tolist())
        metrics["vae"]["sigma_mse"].extend(mse_per_batch(sig0_hat, sig_x).detach().cpu().tolist())
        metrics["vae"]["centroid_mse"].extend(mse_per_batch(cent0_hat, cent_x).detach().cpu().tolist())
        metrics["vae"]["sigma_mae"].extend(mae_per_batch(sig0_hat, sig_x).detach().cpu().tolist())
        metrics["vae"]["centroid_mae"].extend(mae_per_batch(cent0_hat, cent_x).detach().cpu().tolist())

        # ---- Dynamics metrics (final snapshot Y) ----
        y_hist = cloud_to_pairwise_hists(Y, nbins=64)
        reconN_b = vae_recon_loss_mse(out["xN_hat"], y_hist)

        cent_y, sig_y = beam_centroid_sigma(Y)
        sigN_hat = out["sigmaN_hat"]
        centN_hat = out["centroidN_hat"]

        metrics["dyn"]["reconN"].extend(reconN_b.detach().cpu().tolist())
        metrics["dyn"]["sigma_mse"].extend(mse_per_batch(sigN_hat, sig_y).detach().cpu().tolist())
        metrics["dyn"]["centroid_mse"].extend(mse_per_batch(centN_hat, cent_y).detach().cpu().tolist())
        metrics["dyn"]["sigma_mae"].extend(mae_per_batch(sigN_hat, sig_y).detach().cpu().tolist())
        metrics["dyn"]["centroid_mae"].extend(mae_per_batch(centN_hat, cent_y).detach().cpu().tolist())

        # ---- Plots (only a few batches) ----
        if plot_batches_left > 0:



            plot_batches_left -= 1

            # ----- VAE recon plots on X (stage1 visual) -----
            x_hist_np = out["x_hist"].detach().cpu().numpy()        # [B,15,64,64] truth for VAE recon
            x_hat_np  = out["vae"]["x_hat"].detach().cpu().numpy()  # [B,15,64,64] pred by VAE

            # choose which sample(s) in the batch to plot
            sample_idx = 0
            save_triplets_for_sample(
                truth_15=x_hist_np,
                pred_15=x_hat_np,
                sample_idx=sample_idx,
                pairs_to_plot=args.pairs_to_plot,
                plots_dir=plots_dir,
                prefix=f"vaeX_b{bi}",
            )

            # optional: scatter for sigma/centroid at X (VAE aux heads)
            cent_x_np, sig_x_np = cent_x.detach().cpu().numpy(), sig_x.detach().cpu().numpy()
            plot_sigma_centroid_scatter(
                sigma_true=sig_x_np,
                sigma_pred=sig0_hat.detach().cpu().numpy(),
                cent_true=cent_x_np,
                cent_pred=cent0_hat.detach().cpu().numpy(),
                out_path=os.path.join(plots_dir, f"vaeX_scatter_b{bi}.png"),
                title_prefix=f"VAE@X batch {bi}",
            )

            # scatter for final stats
            plot_sigma_centroid_scatter(
                sigma_true=sig_y.detach().cpu().numpy(),
                sigma_pred=sigN_hat.detach().cpu().numpy(),
                cent_true=cent_y.detach().cpu().numpy(),
                cent_pred=centN_hat.detach().cpu().numpy(),
                out_path=os.path.join(plots_dir, f"final_scatter_batch{bi}.png"),
                title_prefix=f"Final batch {bi}",
            )

            # hist triplets for selected pairs, first sample in batch
            # ----- Dynamics (final) recon plots on Y (stage2 visual) -----
            y_hist_np = y_hist.detach().cpu().numpy()              # [B,15,64,64] truth at Y
            y_hat_np  = out["xN_hat"].detach().cpu().numpy()       # [B,15,64,64] pred at Y

            sample_idx = 0
            save_triplets_for_sample(
                truth_15=y_hist_np,
                pred_15=y_hat_np,
                sample_idx=sample_idx,
                pairs_to_plot=args.pairs_to_plot,
                plots_dir=plots_dir,
                prefix=f"dynY_b{bi}",
            )

            plot_sigma_centroid_scatter(
                sigma_true=sig_y.detach().cpu().numpy(),
                sigma_pred=sigN_hat.detach().cpu().numpy(),
                cent_true=cent_y.detach().cpu().numpy(),
                cent_pred=centN_hat.detach().cpu().numpy(),
                out_path=os.path.join(plots_dir, f"dynY_scatter_b{bi}.png"),
                title_prefix=f"Dyn@Y batch {bi}",
            )

            for pidx in args.pairs_to_plot:
                i, j = PAIR_INDEX[pidx]
                title = f"batch{bi} sample{sample_idx} pair ({PHASE_SPACE_LABELS[i]}, {PHASE_SPACE_LABELS[j]})"
                plot_hist_triplet(
                    truth_map=y_hist_np[sample_idx, pidx],
                    pred_map=y_hat_np[sample_idx, pidx],
                    out_path=os.path.join(plots_dir, f"triplet_b{bi}_s{sample_idx}_p{pidx}.png"),
                    title=title,
                )

    # Summarize metrics
    def summarize(arr: List[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std()) if a.size else float("nan"),
            "p50": float(np.percentile(a, 50)) if a.size else float("nan"),
            "p90": float(np.percentile(a, 90)) if a.size else float("nan"),
            "p99": float(np.percentile(a, 99)) if a.size else float("nan"),
        }

    report = {
        "vae": {
            "recon0": summarize(metrics["vae"]["recon0"]),
            "sigma_mse": summarize(metrics["vae"]["sigma_mse"]),
            "centroid_mse": summarize(metrics["vae"]["centroid_mse"]),
            "sigma_mae": summarize(metrics["vae"]["sigma_mae"]),
            "centroid_mae": summarize(metrics["vae"]["centroid_mae"]),
        },
        "dyn": {
            "reconN": summarize(metrics["dyn"]["reconN"]),
            "sigma_mse": summarize(metrics["dyn"]["sigma_mse"]),
            "centroid_mse": summarize(metrics["dyn"]["centroid_mse"]),
            "sigma_mae": summarize(metrics["dyn"]["sigma_mae"]),
            "centroid_mae": summarize(metrics["dyn"]["centroid_mae"]),
        },
        "args": vars(args),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Wrote:", os.path.join(args.out_dir, "metrics.json"))
    print("Plots in:", plots_dir)


if __name__ == "__main__":
    main()