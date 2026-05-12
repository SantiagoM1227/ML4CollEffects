#!/bin/bash
#SBATCH --job-name=WAKEFIELD_GEN
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/WAKEFIELD_GEN-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/WAKEFIELD_GEN-%j.err
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

# --- wake model: use external wake file (txt) ---
export WAKE_TXT_PATH="/pbs/home/s/smartinez/ML4CollEffects/third_party/fcc_ee_booster_pywit_model/fcc_ee_booster_pywit_model/data/total/W_total_long (sigma 0.4 mm).txt"

# --- wake diagnostics plots ---
export WAKE_DIAG_DIR="/pbs/home/s/smartinez/ML4CollEffects/outputs/wakes/wake_diag_${SLURM_JOB_ID}"
export WAKE_DIAG_MAX="20"          # save only first N samples
export WAKE_DIAG_EVERY="1"         # plot every sample until max reached
export WAKE_DIAG_FLIP="0"          # set to 1 if wake orientation is wrong
export WAKE_DIAG_CAUSAL="0"        # keep your current FFT convolution by default

# Let wrapper know how many CPUs are allocated
export JOBLIB_CPUS=${SLURM_CPUS_PER_TASK:-16}

# Run the wrapper (uses srun to launch within the allocation)
python -u /pbs/home/s/smartinez/ML4CollEffects/scripts/run_generate_xsuite.py