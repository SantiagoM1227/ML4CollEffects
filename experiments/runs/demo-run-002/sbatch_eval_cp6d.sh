#!/bin/bash
#SBATCH --job-name=cp6d-neuralop-eval
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/cp6d-neuralop-eval-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/cp6d-neuralop-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=flash
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T10:45:01.npz"

# demo002 outputs
export CP6D_CKPT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-002/models/cp6d_neuralop_fno.pt"
export CP6D_META="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-002/models/cp6d_neuralop_meta.json"
export OUT_DIR="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-002/output_eval"
export SPLIT="test"

python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-002/eval_cp6d_neuralop.py