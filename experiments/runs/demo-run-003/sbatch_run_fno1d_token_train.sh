#!/bin/bash
#SBATCH --job-name=demo003-latent1d-train
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-latent1d-train-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-latent1d-train-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

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

# Dataset
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:37:06.npz"

#Export training
export STAGE="B"
export DEMO003_AE=/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/demo003_latent1d_ae.pt

# Run training (Demo003: Latent-1D tokens + NeuralOperator FNO1d)
srun python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/train_fno1d_token.py