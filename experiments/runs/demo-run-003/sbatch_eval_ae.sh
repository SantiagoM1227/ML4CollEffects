#!/bin/bash
#SBATCH --job-name=demo003-ae-eval
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-ae-eval-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-ae-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=flash
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Dataset
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:37:06.npz"

# AE checkpoint + meta
export DEMO003_AE="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/demo003_latent1d_ae.pt"
export DEMO003_META="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/demo003_latent1d_meta.json"

# output
export OUT_DIR="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/output_eval_ae"
export SPLIT="test"

srun python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/eval_ae.py