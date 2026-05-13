#!/bin/bash
#SBATCH --job-name=cp6d-neuralop-train
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/cp6d-neuralop-train-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/cp6d-neuralop-train-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

# Your dataset
export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T10:45:01.npz"

# Optional: reduce PyTorch CPU threads (can help on some clusters)
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Run training (NeuralOperator FNO on CP6D factors)
srun python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-002/train_cp6d_neuralop.py