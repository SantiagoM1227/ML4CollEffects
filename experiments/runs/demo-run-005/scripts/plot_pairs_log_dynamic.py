from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


PHASE_LABELS = ["x", "y", "zeta", "px", "py", "delta"]
PAIRS = [(0, 3), (1, 4), (2, 5), (0, 1), (3, 4), (0, 2)]


def prange(x: np.ndarray, lo=0.5, hi=99.5, pad_frac: float = 0.05):
    r0, r1 = np.percentile(x, [lo, hi])
    r0 = float(r0); r1 = float(r1)
    if not np.isfinite(r0) or not np.isfinite(r1) or r0 == r1:
        r0 = float(np.min(x)); r1 = float(np.max(x))
    if r0 == r1:
        r0 -= 1.0
        r1 += 1.0
    # small padding so points don't stick to border
    pad = pad_frac * (r1 - r0)
    return r0 - pad, r1 + pad


def hist2d(z, a, b, bins, ra, rb, eps=1e-12):
    H, xedges, yedges = np.histogram2d(
        z[:, a], z[:, b],
        bins=bins,
        range=[ra, rb],
        density=False,
    )
    H = H.astype(np.float64)
    H = H / (H.sum() + eps)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return H, extent


def robust_lognorm_limits(H_list, q_low=5.0, q_high=99.5, floor=1e-12):
    # gather positive values
    pos = np.concatenate([H[H > 0] for H in H_list], axis=0) if H_list else np.array([])
    if pos.size == 0:
        return floor, 1.0
    vmin = float(np.percentile(pos, q_low))
    vmax = float(np.percentile(pos, q_high))
    vmin = max(vmin, floor)
    vmax = max(vmax, vmin * 10.0)  # ensure separation
    return vmin, vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--outdir", type=str, default="./output")
    ap.add_argument("--qlo", type=float, default=0.5, help="percentile low for x/y ranges")
    ap.add_argument("--qhi", type=float, default=99.5, help="percentile high for x/y ranges")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = np.load(args.dataset, allow_pickle=True)
    X = ds["X_cloud"][args.index].astype(np.float64)
    Y = ds["Y_cloud"][args.index].astype(np.float64)

    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(11, 3.2 * len(PAIRS)))

    for i, (a, b) in enumerate(PAIRS):
        # Dynamic x/y limits per pair (using both before+after)
        xa = np.concatenate([X[:, a], Y[:, a]])
        yb = np.concatenate([X[:, b], Y[:, b]])
        ra = prange(xa, lo=args.qlo, hi=args.qhi)
        rb = prange(yb, lo=args.qlo, hi=args.qhi)

        Hx, ext = hist2d(X, a, b, args.bins, ra, rb)
        Hy, _ = hist2d(Y, a, b, args.bins, ra, rb)
        Hd = np.abs(Hy - Hx)

        # Dynamic color scaling per pair (log on density)
        vmin, vmax = robust_lognorm_limits([Hx, Hy], q_low=5.0, q_high=99.5)

        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]
        im0 = ax0.imshow(Hx + 1e-18, origin="lower", aspect="auto", extent=ext,
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        im1 = ax1.imshow(Hy + 1e-18, origin="lower", aspect="auto", extent=ext,
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        im2 = ax2.imshow(Hd, origin="lower", aspect="auto", extent=ext)

        ax0.set_title(f"Before: {PHASE_LABELS[a]} vs {PHASE_LABELS[b]}")
        ax1.set_title(f"After (Truth): {PHASE_LABELS[a]} vs {PHASE_LABELS[b]}")
        ax2.set_title("|diff|")

        ax0.set_xlabel(PHASE_LABELS[a]); ax1.set_xlabel(PHASE_LABELS[a]); ax2.set_xlabel(PHASE_LABELS[a])
        ax0.set_ylabel(PHASE_LABELS[b]); ax1.set_ylabel(PHASE_LABELS[b]); ax2.set_ylabel(PHASE_LABELS[b])

        # colorbars per row (optional). If too many, comment these out.
        c0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)
        c0.set_label("density (log)")
        c2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        c2.set_label("|diff|")

    fig.tight_layout()
    out_png = outdir / f"pairs_log_dynamic_idx{args.index:05d}.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("saved:", out_png)


if __name__ == "__main__":
    main()