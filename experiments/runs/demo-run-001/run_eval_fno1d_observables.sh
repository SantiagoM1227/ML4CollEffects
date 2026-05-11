#!/bin/bash
#SBATCH --job-name=eval-fno1d
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/eval-fno1d-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/eval-fno1d-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=flash
#SBATCH --time=01:00:00
# If your cluster requires a specific partition, replace the line below with:
# Otherwise Slurm will use the default partition (omit the --partition directive).
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Let wrapper know how many CPUs are allocated
export JOBLIB_CPUS=${SLURM_CPUS_PER_TASK:-16}

export DATASET_PATH=/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-06T09:07:30.npz
export CKPT_PATH=/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-001/models/fno1d_lambda_best.pt

python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-001/eval_fno1d_observables.py