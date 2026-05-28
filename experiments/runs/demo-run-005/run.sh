#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN_ROOT/../../.." && pwd)"

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/processed/neural_dataset.npz}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/output}"
DEVICE="${DEVICE:-cuda}"
BINS="${BINS:-64}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
BATCH_STAGE1="${BATCH_STAGE1:-16}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-30}"
BATCH_STAGE2="${BATCH_STAGE2:-128}"

export PYTHONPATH="$RUN_ROOT:${PYTHONPATH:-}"
cd "$RUN_ROOT"

python -u -m scripts.trainers.train_vae \
  --data "$DATASET_PATH" \
  --outdir "$OUT_ROOT/stage1_vae" \
  --bins "$BINS" \
  --epochs "$EPOCHS_STAGE1" \
  --batch-size "$BATCH_STAGE1" \
  --device "$DEVICE"

VAE_CKPT="$OUT_ROOT/stage1_vae/checkpoints/vae_ep$(printf '%03d' "$((EPOCHS_STAGE1-1))").pt"

python -u -m scripts.evaluation.evaluate_vae \
  --data "$DATASET_PATH" \
  --vae-ckpt "$VAE_CKPT" \
  --outdir "$OUT_ROOT/stage1_vae" \
  --device "$DEVICE" \
  --split val

python -u -m scripts.export_latents \
  --data "$DATASET_PATH" \
  --vae-ckpt "$VAE_CKPT" \
  --out-npz "$OUT_ROOT/stage1_vae/latent/latent_dataset.npz" \
  --bins "$BINS" \
  --batch-size 64 \
  --device "$DEVICE"

python -u -m scripts.trainers.train_dynamics_1step \
  --latent-npz "$OUT_ROOT/stage1_vae/latent/latent_dataset.npz" \
  --outdir "$OUT_ROOT/stage2_dynamics" \
  --epochs "$EPOCHS_STAGE2" \
  --batch-size "$BATCH_STAGE2" \
  --device "$DEVICE"

DYN_CKPT="$OUT_ROOT/stage2_dynamics/checkpoints/dyn_ep$(printf '%03d' "$((EPOCHS_STAGE2-1))").pt"

python -u -m scripts.evaluation.evaluate_dynamics_1step \
  --latent-npz "$OUT_ROOT/stage1_vae/latent/latent_dataset.npz" \
  --vae-ckpt "$VAE_CKPT" \
  --dyn-ckpt "$DYN_CKPT" \
  --outdir "$OUT_ROOT/stage2_dynamics" \
  --device "$DEVICE" \
  --split val \
  --n-plots 8
