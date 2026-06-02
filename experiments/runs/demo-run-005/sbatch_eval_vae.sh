#!/bin/bash
#SBATCH --job-name=demo005-eval-vae
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-eval-vae-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-eval-vae-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

set -euo pipefail

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-13T08:37:06.npz"
export DEMO005_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-005"
cd "$DEMO005_ROOT"

export DEVICE="${DEVICE:-cuda}"
export VAE_CKPT="${VAE_CKPT:-$DEMO005_ROOT/output/stage1_vae/vae_best.pt}"
export OUT_DIR="${OUT_DIR:-$DEMO005_ROOT/output/stage1_vae/eval}"

mkdir -p "$OUT_DIR"

echo "=== demo-run-005 EVAL VAE ==="
echo "DATASET_PATH=$DATASET_PATH"
echo "DEMO005_ROOT=$DEMO005_ROOT"
echo "VAE_CKPT=$VAE_CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "DEVICE=$DEVICE"
echo "============================="

# NOTE: eval.py currently requires --dyn_ckpt.
# Since you don't have dynamics yet, pass the same ckpt for dyn_ckpt.
srun python -u eval.py \
  --data "$DATASET_PATH" \
  --vae_ckpt "$VAE_CKPT" \
  --dyn_ckpt "$VAE_CKPT" \
  --out_dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch_size 4 \
  --num_plot_batches 2 \
  --pairs_to_plot 0 1 10 11