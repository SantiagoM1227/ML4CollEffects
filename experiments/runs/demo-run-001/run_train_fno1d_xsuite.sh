#!/bin/bash
#SBATCH --job-name=train_xsuite
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/train_xsuite-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/train_xsuite-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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


export DATASET_PATH=/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-04-29T12:26:38.npz
srun python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-001/train_fno1d_xsuite.py