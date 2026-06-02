#!/bin/bash
#SBATCH --job-name=demo006-compile-check
#SBATCH --output=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo006-compile-check-%j.out
#SBATCH --error=/pbs/home/s/smartinez/ML4CollEffects/outputs/demo006-compile-check-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smartinezsa@unal.edu.co

set -euo pipefail

export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

# ---- paths ----
export DEMO006_ROOT="/pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-006"
cd "$DEMO006_ROOT"

FILE="train_vae_SSNO.py"

echo "=== compile check ==="
echo "PWD=$(pwd)"
echo "Python=$(which python)"
python -V
echo "File=$FILE"
echo "====================="

# 1) Syntax-only compilation (catches SyntaxError)
echo "[1/3] python -m py_compile $FILE"
python -m py_compile "$FILE"
echo "OK: py_compile passed"

# 2) Import check (catches missing modules / errors at import time)
# NOTE: This executes top-level code, so it may fail if your file does work on import.
echo "[2/3] python -c 'import train_vae_SSNO'"
python -c "import train_vae_SSNO"
echo "OK: import passed"

# 3) Optional: minimal object construction check (catches some NameError/shape defaults)
echo "[3/3] Instantiate model (no training)"
python - <<'PY'
import torch
import train_vae_SSNO as m

mu_dim = 3
model = m.BeamLatentTrackingModel(mu_dim=mu_dim, latent_dim=256, d_model=512)
print("OK: BeamLatentTrackingModel instantiated")

# lightweight forward smoke test (CPU, tiny tensors)
B, Np, N_elem = 1, 16, 4
X = torch.randn(B, Np, 6)
MU = torch.randn(B, mu_dim)

mu_prep = m.MUPreprocessor(m.MUPreprocessConfig(specs=[1.0]*mu_dim, standardize=False))
MUe = mu_prep(MU)

elem_params, elem_s = m.make_dummy_lattice_tokens(B, N_elem, device=torch.device("cpu"))

with torch.no_grad():
    out = model(X, MUe, elem_params, elem_s, mode="AR")

print("OK: forward pass completed")
print("x_hist:", tuple(out["x_hist"].shape))
print("z_pred_traj:", tuple(out["z_pred_traj"].shape))
print("xN_hat:", tuple(out["xN_hat"].shape))
PY

echo "=== ALL CHECKS PASSED ==="