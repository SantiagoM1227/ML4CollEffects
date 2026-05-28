# demo-run-005

Notebook-free experiment package mirroring `demo-run-004`, with VAE code placed in `scripts/vae_tf.py`.

## Quick start

From repository root:

```bash
DATASET_PATH=/absolute/path/to/your_dataset.npz \
  bash experiments/runs/demo-run-005/run.sh
```

Environment variables supported by `run.sh`:

- `DATASET_PATH` (default: `data/processed/neural_dataset.npz` under repo)
- `OUT_ROOT` (default: `experiments/runs/demo-run-005/output`)
- `DEVICE` (`cuda` or `cpu`, auto-falls back to CPU in Python when CUDA is unavailable)
- `BINS`, `EPOCHS_STAGE1`, `BATCH_STAGE1`, `EPOCHS_STAGE2`, `BATCH_STAGE2`

## Entrypoints

- Stage 1 train: `python -m scripts.trainers.train_vae`
- Stage 1 eval: `python -m scripts.evaluation.evaluate_vae`
- Latent export: `python -m scripts.export_latents`
- Stage 2 train: `python -m scripts.trainers.train_dynamics_1step`
- Stage 2 eval: `python -m scripts.evaluation.evaluate_dynamics_1step`
