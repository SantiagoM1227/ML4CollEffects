#!/bin/bash
#SBATCH --job-name=VAESSNO-pipeline
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo006-pipeline-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo006-pipeline-%j.err
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

export DATASET_PATH="${DATASET_PATH:-/pbs/home/s/smartinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-04-28T09_46_07.npz}"
export DEMO006_ROOT="${DEMO006_ROOT:-/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-006}"
cd "$DEMO006_ROOT"

export DEVICE="${DEVICE:-cuda}"

# hyperparams (override at submission if desired)
export VAE_EPOCHS="${VAE_EPOCHS:-20}"
export VAE_BATCH="${VAE_BATCH:-16}"
export DYN_EPOCHS="${DYN_EPOCHS:-30}"
export DYN_BATCH="${DYN_BATCH:-16}"

export VAE_OUTDIR="$DEMO006_ROOT/output/stage1_vae"
export DYN_OUTDIR="$DEMO006_ROOT/output/stage2_dynamics"
mkdir -p "$VAE_OUTDIR" "$DYN_OUTDIR"

export VAE_CKPT="$VAE_OUTDIR/vae_best.pt"
export DYN_CKPT="$DYN_OUTDIR/dyn_best.pt"

echo "=== demo006 pipeline ==="
echo "DATASET_PATH=$DATASET_PATH"
echo "DEMO006_ROOT=$DEMO006_ROOT"
echo "VAE_CKPT=$VAE_CKPT"
echo "DYN_CKPT=$DYN_CKPT"
echo "VAE_EPOCHS=$VAE_EPOCHS VAE_BATCH=$VAE_BATCH"
echo "DYN_EPOCHS=$DYN_EPOCHS DYN_BATCH=$DYN_BATCH"
echo "========================"

# ---- Stage 1: train VAE ----
srun python -u train_vae_SSNO.py \
  --data "$DATASET_PATH" \
  --mode vae \
  --out_ckpt "$VAE_CKPT" \
  --epochs "$VAE_EPOCHS" \
  --batch_size "$VAE_BATCH" \
  --device "$DEVICE"

# Submit VAE eval as a new job depending on THIS job finishing OK
VAE_EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${SLURM_JOB_ID} \
  --export=ALL,DATASET_PATH="$DATASET_PATH",DEMO006_ROOT="$DEMO006_ROOT",DEVICE="$DEVICE",VAE_CKPT="$VAE_CKPT" \
  "$DEMO006_ROOT/sbatch_eval_vae.sh")
echo "Submitted VAE eval job: $VAE_EVAL_JOBID"

# ---- Stage 2: train dynamics ----
srun python -u train_vae_SSNO.py \
  --data "$DATASET_PATH" \
  --mode dynamics \
  --vae_ckpt "$VAE_CKPT" \
  --freeze_vae \
  --out_ckpt "$DYN_CKPT" \
  --epochs "$DYN_EPOCHS" \
  --batch_size "$DYN_BATCH" \
  --device "$DEVICE"

# Submit dynamics eval as a new job depending on THIS job finishing OK
DYN_EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${SLURM_JOB_ID} \
  --export=ALL,DATASET_PATH="$DATASET_PATH",DEMO006_ROOT="$DEMO006_ROOT",DEVICE="$DEVICE",VAE_CKPT="$VAE_CKPT",DYN_CKPT="$DYN_CKPT" \
  "$DEMO006_ROOT/sbatch_eval_vae.sh")
echo "Submitted dynamics eval job: $DYN_EVAL_JOBID"

echo "Pipeline finished training; eval jobs submitted."