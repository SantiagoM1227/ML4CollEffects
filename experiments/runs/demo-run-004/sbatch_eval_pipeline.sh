#!/bin/bash
set -euo pipefail

# --- Paths (edit if needed)
DEMO004_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-004"
DATASET="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T10:45:01.npz"

# Everything will be written here (relative to DEMO004_ROOT)
OUTDIR="./output"

# Optional: predictions npz (leave empty if you only want before-vs-truth)
PRED=""   # e.g. "$DEMO004_ROOT/models/infer/predicted_distributions.npz"

# --- Settings
BINS=64
IDX=0

cd "$DEMO004_ROOT"

if [ -z "$PRED" ]; then
  echo "[RUN] Before vs Truth-after (requires Y_cloud in dataset)"
  python -u scripts/plot_15x2d_truth_vs_pred.py \
    --dataset "$DATASET" \
    --index "$IDX" \
    --bins "$BINS" \
    --outdir "$OUTDIR"
else
  echo "[RUN] Truth-after vs Pred-after (requires Y_cloud in dataset + Xhat in pred npz)"
  python -u scripts/plot_15x2d_truth_vs_pred.py \
    --dataset "$DATASET" \
    --pred "$PRED" \
    --index "$IDX" \
    --bins "$BINS" \
    --outdir "$OUTDIR"
fi

echo "[DONE] check output in: $DEMO004_ROOT/$OUTDIR"