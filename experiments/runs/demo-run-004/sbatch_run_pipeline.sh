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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:37:06.npz"
export DEMO004_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-004"

export PYTHONPATH="$DEMO004_ROOT:${PYTHONPATH:-}"
cd "$DEMO004_ROOT"

# Stage 1
srun python -u -m scripts.trainers.train_vae \
  --data "$DATASET_PATH" \
  --outdir "$DEMO004_ROOT/output/stage1_vae" \
  --bins 64 \
  --epochs 20 \
  --batch-size 16 \
  --device cuda

srun python -u -m scripts.evaluation.evaluate_vae \
  --data "$DATASET_PATH" \
  --vae-ckpt "$DEMO004_ROOT/output/stage1_vae/checkpoints/vae_ep019.pt" \
  --outdir "$DEMO004_ROOT/output/stage1_vae" \
  --device cuda \
  --split val

# Export latents
srun python -u -m scripts.export_latents \
  --data "$DATASET_PATH" \
  --vae-ckpt "$DEMO004_ROOT/output/stage1_vae/checkpoints/vae_ep019.pt" \
  --out-npz "$DEMO004_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --bins 64 \
  --batch-size 64 \
  --device cuda

# Stage 2
srun python -u -m scripts.trainers.train_dynamics_1step \
  --latent-npz "$DEMO004_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --outdir "$DEMO004_ROOT/output/stage2_dynamics" \
  --epochs 30 \
  --batch-size 128 \
  --device cuda

srun python -u -m scripts.evaluation.evaluate_dynamics_1step \
  --latent-npz "$DEMO004_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
  --vae-ckpt "$DEMO004_ROOT/output/stage1_vae/checkpoints/vae_ep019.pt" \
  --dyn-ckpt "$DEMO004_ROOT/output/stage2_dynamics/checkpoints/dyn_ep029.pt" \
  --outdir "$DEMO004_ROOT/output/stage2_dynamics" \
  --device cuda \
  --split val \
  --n-plots 8
