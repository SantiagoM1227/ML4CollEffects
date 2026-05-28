#!/bin/bash
#SBATCH --job-name=demo005-two-stage
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-two-stage-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-two-stage-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

set -euo pipefail

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

export DEMO005_ROOT="${DEMO005_ROOT:-/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-005}"
export DATASET_PATH="${DATASET_PATH:-$DEMO005_ROOT/../../../data/processed/neural_dataset.npz}"
export DEVICE="${DEVICE:-cuda}"
export EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
export BATCH_STAGE1="${BATCH_STAGE1:-16}"
export EPOCHS_STAGE2="${EPOCHS_STAGE2:-30}"
export BATCH_STAGE2="${BATCH_STAGE2:-128}"

export PYTHONPATH="$DEMO005_ROOT:${PYTHONPATH:-}"
cd "$DEMO005_ROOT"

# ----------------------
# Stage 1: Train VAE
# ----------------------
srun python -u -m scripts.train_vae \
  --data "$DATASET_PATH" \
  --outdir "$DEMO005_ROOT/output/stage1_vae" \
  --bins 64 \
  --epochs "$EPOCHS_STAGE1" \
  --batch-size "$BATCH_STAGE1" \
  --device "$DEVICE"

VAE_CKPT="$DEMO005_ROOT/output/stage1_vae/checkpoints/vae_ep$(printf '%03d' "$((EPOCHS_STAGE1-1))").pt"

srun python -u -m scripts.eval_vae \
  --data "$DATASET_PATH" \
  --vae-ckpt "$VAE_CKPT" \
  --outdir "$DEMO005_ROOT/output/stage1_vae" \
  --device "$DEVICE" \
  --split val

# ----------------------
# Export latents for stage 2
# ----------------------
srun python -u -m scripts.export_latents \
  --data "$DATASET_PATH" \
  --vae-ckpt "$VAE_CKPT" \
  --out-npz "$DEMO005_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --bins 64 \
  --batch-size 64 \
  --device "$DEVICE"

# ----------------------
# Stage 2: Train latent dynamics model
# ----------------------
srun python -u -m scripts.train_dynamics \
  --latent-npz "$DEMO005_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --outdir "$DEMO005_ROOT/output/stage2_dynamics" \
  --epochs "$EPOCHS_STAGE2" \
  --batch-size "$BATCH_STAGE2" \
  --device "$DEVICE"

DYN_CKPT="$DEMO005_ROOT/output/stage2_dynamics/checkpoints/dyn_ep$(printf '%03d' "$((EPOCHS_STAGE2-1))").pt"

srun python -u -m scripts.eval_dynamics \
  --latent-npz "$DEMO005_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --vae-ckpt "$VAE_CKPT" \
  --dyn-ckpt "$DYN_CKPT" \
  --outdir "$DEMO005_ROOT/output/stage2_dynamics" \
  --device "$DEVICE" \
  --split val \
  --n-plots 8
