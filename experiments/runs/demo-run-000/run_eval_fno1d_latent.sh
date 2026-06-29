export DATASET_PATH=/home/martinez/ML4CollEffects/data/neural/neural_xsuite_dataset_2026-05-04T11:55:54.npz
export TOKEN_AE_CKPT=/home/martinez/ML4CollEffects/experiments/runs/demo-run-000/models/token_ae.pt
export OP_CKPT=/home/martinez/ML4CollEffects/experiments/runs/demo-run-000/models/latent_fno.pt
export OUT_DIR=/home/martinez/ML4CollEffects/experiments/runs/demo-run-000/output
export SPLIT=test

python -u /pbs/home/s/smartinez/ML4CollEffects/experiments/runs/demo-run-000/eval_fno1s_latent.py