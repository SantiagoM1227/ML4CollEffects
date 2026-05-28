#!/bin/bash
#SBATCH --job-name=demo005-eval
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-eval-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo005-eval-%j.err
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

export DATASET_PATH="/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-04-28T09:40:32.npz"
export DEMO005_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-005"
export STAGE="${STAGE:-1}"

export PYTHONPATH="$DEMO005_ROOT:${PYTHONPATH:-}"
cd "$DEMO005_ROOT"

if [ "$STAGE" = "1" ]; then
  srun python -u -m scripts.eval_vae \
    --data "$DATASET_PATH" \
    --vae-ckpt "$DEMO005_ROOT/output/stage1_vae/checkpoints/vae_ep099.pt" \
    --outdir "$DEMO005_ROOT/output/stage1_vae" \
    --device cuda \
    --split val
elif [ "$STAGE" = "2" ]; then
  srun python -u -m scripts.eval_dynamics \
    --latent-npz "$DEMO005_ROOT/output/stage1_vae/latent/latent_dataset.npz" \
    --vae-ckpt "$DEMO005_ROOT/output/stage1_vae/checkpoints/vae_ep099.pt" \
    --dyn-ckpt "$DEMO005_ROOT/output/stage2_dynamics/checkpoints/dyn_ep099.pt" \
    --outdir "$DEMO005_ROOT/output/stage2_dynamics" \
    --device cuda \
    --split val \
    --n-plots 8
else
  echo "Unknown STAGE=$STAGE (use STAGE=1 or STAGE=2)"
  exit 2
fi
