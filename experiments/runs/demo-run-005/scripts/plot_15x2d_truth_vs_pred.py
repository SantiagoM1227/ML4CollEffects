from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


PAIR_IDX_6D = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
)

PHASE_LABELS = ["x", "y", "zeta", "px", "py", "delta"]


def prange(x: np.ndarray, lo=0.5, hi=99.5, pad_frac: float = 0.05) -> tuple[float, float]:
    r0, r1 = np.percentile(x, [lo, hi])
    r0 = float(r0); r1 = float(r1)
    if not np.isfinite(r0) or not np.isfinite(r1) or r0 == r1:
        r0 = float(np.min(x)); r1 = float(np.max(x))
    if r0 == r1:
        r0 -= 1.0
        r1 += 1.0
    pad = pad_frac * (r1 - r0)
    return r0 - pad, r1 + pad


def hist2d_pair(
    z: np.ndarray,
    a: int,
    b: int,
    *,
    bins: int,
    ra: tuple[float, float],
    rb: tuple[float, float],
    normalize: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, list[float]]:
    H, xedges, yedges = np.histogram2d(
        z[:, a], z[:, b],
        bins=bins,
        range=[ra, rb],
        density=False,
    )
    H = H.astype(np.float64)
    if normalize:
        H = H / (H.sum() + eps)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return H.astype(np.float32), extent


def robust_lognorm_limits(A: np.ndarray, B: np.ndarray, q_low=5.0, q_high=99.5, floor=1e-12) -> tuple[float, float]:
    pos = np.concatenate([A[A > 0], B[B > 0]], axis=0)
    if pos.size == 0:
        return floor, 1.0
    vmin = float(np.percentile(pos, q_low))
    vmax = float(np.percentile(pos, q_high))
    vmin = max(vmin, floor)
    vmax = max(vmax, vmin * 10.0)
    return vmin, vmax


def plot_triplet_grid_dynamic(
    X_cloud: np.ndarray,
    Y_cloud: np.ndarray,
    *,
    bins: int,
    out_png: Path,
    title_left: str,
    title_right: str,
    suptitle: str,
    qlo_xy: float = 0.5,
    qhi_xy: float = 99.5,
    pad_frac_xy: float = 0.05,
    diff_abs: bool = True,
):
    """
    Uses per-pair dynamic (x,y) ranges + per-channel LogNorm.
    """
    fig, axes = plt.subplots(15, 3, figsize=(11, 30))
    fig.suptitle(suptitle, y=0.995)

    im0_last = None
    im2_last = None

    for i, (a, b) in enumerate(PAIR_IDX_6D):
        # Dynamic x/y limits PER PAIR based on both clouds
        xa = np.concatenate([X_cloud[:, a], Y_cloud[:, a]])
        yb = np.concatenate([X_cloud[:, b], Y_cloud[:, b]])
        ra = prange(xa, lo=qlo_xy, hi=qhi_xy, pad_frac=pad_frac_xy)
        rb = prange(yb, lo=qlo_xy, hi=qhi_xy, pad_frac=pad_frac_xy)

        Hx, extent = hist2d_pair(X_cloud, a, b, bins=bins, ra=ra, rb=rb)
        Hy, _ = hist2d_pair(Y_cloud, a, b, bins=bins, ra=ra, rb=rb)

        D = np.abs(Hy - Hx) if diff_abs else (Hy - Hx)

        vmin, vmax = robust_lognorm_limits(Hx, Hy)

        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]

        im0 = ax0.imshow(
            Hx + 1e-18,
            origin="lower",
            aspect="auto",
            extent=extent,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        im1 = ax1.imshow(
            Hy + 1e-18,
            origin="lower",
            aspect="auto",
            extent=extent,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        im2 = ax2.imshow(
            D,
            origin="lower",
            aspect="auto",
            extent=extent,
        )

        im0_last = im0
        im2_last = im2

        ax0.set_ylabel(f"{PHASE_LABELS[b]}")
        ax0.set_xlabel(f"{PHASE_LABELS[a]}")
        ax1.set_xlabel(f"{PHASE_LABELS[a]}")
        ax2.set_xlabel(f"{PHASE_LABELS[a]}")

        if i == 0:
            ax0.set_title(title_left)
            ax1.set_title(title_right)
            ax2.set_title("|diff|")

    # Global colorbars
    cax1 = fig.add_axes([0.92, 0.67, 0.015, 0.25])
    fig.colorbar(im0_last, cax=cax1, label="density (log, normalized)")
    cax2 = fig.add_axes([0.92, 0.35, 0.015, 0.25])
    fig.colorbar(im2_last, cax=cax2, label="|diff|")

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.99])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("saved:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="xsuite dataset .npz with X_cloud and Y_cloud")
    ap.add_argument("--pred", type=str, default="", help="optional predictions npz (must contain Xhat)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--outdir", type=str, default="./output")

    # dynamic range controls
    ap.add_argument("--qlo-xy", type=float, default=0.5)
    ap.add_argument("--qhi-xy", type=float, default=99.5)
    ap.add_argument("--pad-frac-xy", type=float, default=0.05)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = np.load(args.dataset, allow_pickle=True)
    X_cloud = ds["X_cloud"][args.index].astype(np.float32)

    if "Y_cloud" not in ds:
        raise KeyError("Dataset has no Y_cloud; cannot do before/after truth comparisons.")
    Y_cloud = ds["Y_cloud"][args.index].astype(np.float32)

    if args.pred:
        pred = np.load(args.pred, allow_pickle=True)
        Xhat = pred["Xhat"]

        # expected (T+1,15,bins,bins) OR (15,bins,bins)
        if Xhat.ndim == 4:
            Xhat15 = Xhat[-1].astype(np.float32)
        elif Xhat.ndim == 3:
            Xhat15 = Xhat.astype(np.float32)
        else:
            raise ValueError(f"Unexpected Xhat shape: {Xhat.shape}")

        # For fair plotting, we compare truth-after vs pred-after in histogram-space.
        # Convert Y_cloud to 15x2D using the SAME dynamic ranges inside the plotting routine,
        # but that routine expects clouds. So we do: cloud->hist inside routine.
        #
        # Workaround: we can’t pass hist directly; easiest is to plot truth-before vs truth-after
        # with clouds, OR write a separate hist-vs-hist plotter.
        #
        # For now, we plot before vs truth-after dynamically (cloud based) and ALSO
        # a separate truth-after vs pred-after plot in histogram space (see note below).
        raise NotImplementedError(
            "This dynamic version plots cloud-vs-cloud (before/after). "
            "For truth-vs-pred where pred is already 15x2D, tell me and I’ll add a "
            "hist-vs-hist dynamic plotter (LogNorm + per-channel vmin/vmax) in the same file."
        )

    out_png = outdir / f"before_vs_truth_after_dynamic_idx{args.index:05d}.png"
    plot_triplet_grid_dynamic(
        X_cloud,
        Y_cloud,
        bins=args.bins,
        out_png=out_png,
        title_left="Input (before)",
        title_right="Truth (after)",
        suptitle=f"Before vs Truth-after (idx={args.index})",
        qlo_xy=args.qlo_xy,
        qhi_xy=args.qhi_xy,
        pad_frac_xy=args.pad_frac_xy,
    )


if __name__ == "__main__":
    main()