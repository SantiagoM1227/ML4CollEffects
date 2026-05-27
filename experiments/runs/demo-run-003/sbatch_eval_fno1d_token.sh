#!/bin/bash
#SBATCH --job-name=demo003-latent1d-eval
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-latent1d-eval-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo003-latent1d-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=flash
#SBATCH --time=06:00:00
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
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T10:45:01.npz"

# Demo003 outputs
export DEMO003_AE="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/latent1d_ae.pt"
export DEMO003_OP="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/latent1d_fno.pt"
export DEMO003_META="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/models/latent1d_meta.json"

export OUT_DIR="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/output_eval"
export SPLIT="test"

srun python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-003/eval_fno1d_token.py