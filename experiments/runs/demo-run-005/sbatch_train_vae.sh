#!/bin/bash
#SBATCH --job-name=demo005-train-vae
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-train-vae-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-train-vae-%j.err
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

# ---- paths ----
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:16:56.npz"
export DEMO005_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-005"
cd "$DEMO005_ROOT"

# ---- hyperparams (override at submit time) ----
export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export DEVICE="${DEVICE:-cuda}"

# where to save ckpt
export OUTDIR="$DEMO005_ROOT/output/stage1_vae"
mkdir -p "$OUTDIR"

# IMPORTANT:
# Your train_vae.py currently expects: --out_ckpt (NOT --outdir)
# (If your local train_vae.py differs, adjust here.)
export VAE_CKPT="$OUTDIR/vae_best.pt"

echo "=== demo-run-005 VAE train ==="
echo "DATASET_PATH=$DATASET_PATH"
echo "DEMO005_ROOT=$DEMO005_ROOT"
echo "OUTDIR=$OUTDIR"
echo "VAE_CKPT=$VAE_CKPT"
echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE DEVICE=$DEVICE"
echo "=============================="

# -------------------------
# Train VAE (your script)
# -------------------------
srun python -u train_vae.py \
  --data "$DATASET_PATH" \
  --mode vae \
  --out_ckpt "$VAE_CKPT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"
