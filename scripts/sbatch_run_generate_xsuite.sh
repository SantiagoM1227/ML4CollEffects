#!/bin/bash
#SBATCH --job-name=pycollef-gen
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/pycollef-gen-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/pycollef-gen-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# If your cluster requires a specific partition, replace the line below with:
# SBATCH --partition=<partition_name>
# Otherwise Slurm will use the default partition (omit the --partition directive).
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@domain.example

# --- environment setup (conda example) ---
export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

# BLAS/OpenMP thread limits to avoid oversubscription in workers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Let wrapper know how many CPUs are allocated
export JOBLIB_CPUS=${SLURM_CPUS_PER_TASK:-16}

# Optional external wake TXT (two columns: s/mm, W; lines starting with # are comments)
# export WAKE_TXT_PATH="/path/to/W_total_long.txt"
# Optional wake orientation flip (accepted values: 1/true/yes/on)
# export WAKE_FLIP=1
# export WAKE_DIAG_FLIP=1   # alias supported by wrapper for compatibility
#
# Optional wake diagnostics plots (generated during tracking)
# export WAKE_DIAG_DIR="/path/to/wake-diag"
# export WAKE_DIAG_MAX=10
# export WAKE_DIAG_EVERY=1

# Run the wrapper (uses srun to launch within the allocation)
srun python -u "$(dirname "$0")/run_generate_xsuite.py"
