#!/bin/bash
#SBATCH --job-name=demo004-latent-track
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo004-latent-track-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo004-latent-track-%j.err
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

# Thread control (avoid oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Optional: reduce PyTorch CPU threads
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Dataset (fixed)
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:37:06.npz"

# Demo root
export DEMO004_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-004"

# Train VAE (saves into demo-run-004/models/vae)
srun python -u "$DEMO004_ROOT/scripts/train_vae.py" \
  --data "$DATASET_PATH" \
  --out "$DEMO004_ROOT/models/vae" \
  --bins 64 \
  --epochs 20 \
  --batch-size 16 \
  --device cuda