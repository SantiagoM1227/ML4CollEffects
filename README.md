# ML4CollEffects

Machine learning for accelerator physics — research code for studying collective effects, beam dynamics, symplectic / Hénon-like models, neural operators, normalizing flows, and Xsuite dataset generation.

This repository collects code, datasets and reproducible experiment artifacts used for ongoing research into ML-based modeling of collective beam dynamics. It is organized for reproducibility, collaboration, and iterative experiment development.

---
## Project overview

This repository contains research code and artifacts for using machine learning (neural operators, normalizing flows, etc.) to model and analyze collective effects in accelerator beam dynamics. Major research axes include:

- Symplectic/Hénon-style low-dimensional models for beam dynamics and stability analysis.
- Data-driven surrogates: neural operators and normalizing flows.
- Xsuite-compatible dataset generation and processing for training and evaluation.
- External experimental datasets availability and integration.
- Reproducible experiment pipelines that capture configs, metrics, logs, and checkpoints.


---

## Repository structure 

Top-level folders follow the layout used across the project. Use these as the canonical organization.

- `configs/` — static configuration files and experiment templates (YAML/JSON/TOML as preferred).
- `data/`
  - `raw/` — original, immutable datasets (source-of-truth copies; do not modify in-place).
  - `external/` — datasets from third parties or external downloads.
  - `processed/` — preprocessed datasets ready for training/evaluation.
  - `metadata/` — dataset manifests, schema, checksums, provenance.
- `docs/`
  - `notes/` — working notes, design documents (human-readable).
  - `references/`
    - `papers/` — bibliographic notes and PDFs where permitted.
    - `presentations/` — slides and posters.
- `experiments/`
  - `runs/` — reproducible experiment run artifacts (see below for layout).
- `notebooks/` — exploratory analyses and experiments (NOT the source of truth).
- `outputs/`
  - `checkpoints/` — model checkpoints produced by experiments.
  - `figures/` — figures and plots exported from experiments/analysis.
  - `logs/` — aggregated logs and run transcripts.
  - `tables/` — result tables and CSV exports.
- `reports/`
  - `midterm/`
    - `src/` — LaTeX/source for reports.
    - `build/` — compiled artifacts (PDF, aux files).
- `scripts/` — runnable entry points (dataset generation, training wrappers, evaluation scripts).
- `src/collefects_ml/`
  - `data/` — dataset loaders, transforms, and I/O utilities.
  - `eval/` — evaluation metrics and tooling.
  - `models/` — model definitions (neural operators, flows, baseline models).
  - `physics/` — physics-informed utilities, symplectic integrators, Hénon variants.
  - `training/` — training loops, trainers, and checkpointing helpers.
  - `utils/` — small utility modules used across the codebase.
- `tests/` — unit and integration tests.
- `third_party/`
  - `ment-flow/` — vendored or referenced external code (kept separate).
  - `neuraloperator/` — external neural operator code (kept separate).

Note:  `src/` is intended to be the authoritative implementation. 
- Notebooks are exploratory: use notebooks for interactive analysis and plotting; don’t rely on them for reproducible experiments or canonical code paths.

---

---

## Typical workflow & reproducibility

Foreseen usage (to be implemented and documented in more detail):

1. Prepare data
   - Place original data in `data/raw/`.
   - Document provenance and checksums in `data/metadata/`.
   - Run preprocessing scripts from `scripts/` (or `src/collefects_ml/data`) to create `data/processed/`.

2. Configure experiments
   - Place experiment config templates in `configs/` and instantiate per-run configs into `experiments/runs/<run-id>/config.yaml`.

3. Run experiments
   - Use `scripts/train_*.py` or `src/collefects_ml/training` entry points to run experiments.
   - Each run should write a self-contained folder under `experiments/runs/` (see layout below).

4. Collate outputs
   - Checkpoints, metrics, logs, and figures should be stored under `experiments/runs/<run-id>/` and copied into `outputs/` as appropriate for publication or downstream analysis.

5. Record environment
   - Save environment info (Python version, pip freeze, OS, GPU details) with the run metadata.

Reproducibility relies on:
- Config file used to launch the run (YAML/JSON) — commit to `experiments/runs/<run-id>/config.yaml`.
- Random seeds and deterministic flags recorded.
- Version of the code: commit hash or tag saved under `experiments/runs/<run-id>/git-rev.txt`.
- Checkpoint(s) saved under `experiments/runs/<run-id>/checkpoints/`.
- Metrics and evaluation outputs (e.g., `metrics.json`).
- Logs and console output (e.g., `run.log`).
- Figures exported into `figures/` or within the run folder.
- Note of any manual steps or human interventions.

---

## Experiment run-folder layout (suggested example)

Intended self-contained layout for a single experiment run under `experiments/runs/<run-id>/`:

```
experiments/runs/<run-id>/
├─ config.yaml            # config used for the run (hyperparams, dataset, seed)
├─ git-rev.txt            # commit hash / branch info
├─ run.log                # stdout/stderr captured during run
├─ environment.txt        # python --version; pip freeze; GPU/OS info
├─ metrics/
│  └─ metrics.json        # structured evaluation metrics
├─ checkpoints/
│  └─ model_epoch_10.pt
├─ figures/
│  └─ loss_curve.png
├─ artifacts/
│  └─ processed_dataset_info.json
└─ notes.md               # short readable notes about the run
```

---

## Notebooks usage and  source code

- `notebooks/` are for exploration, visualization, and rapid prototyping. They are extremely useful for interactive work but are not guaranteed to be maintained or reproducible.
- `src/` is the canonical source of truth. Implementations required for experiments, testing, and CI must be located in `src/collefects_ml/` and be importable as a package (see the quick-start `pip install -e src` step).
- When a notebook stabilizes into a reusable routine, the logic should be extracted into `src/` and covered with tests in `tests/`.

---

## Data management

- Keep original, immutable files in `data/raw/`. 
- Store processing scripts and transformation code in `src/collefects_ml/data/` and `scripts/`.
- `data/metadata/` to record schema, provenance, and checksums.

---

## Development notes & contributing

- Tests: add unit tests under `tests/` alongside modules in `src/`.
- Linting & formatting: prefer a consistent formatter (black/isort/ruff/etc.). Further work includes setting up minimal formatting scdripting.


## Ongoing cleanup / restructuring

This repository is actively being cleaned and restructured to:
- Centralize canonical code into `src/` and minimize duplicated logic in notebooks.
- Move generated/derived files out of source directories into `outputs/` or `experiments/runs/`.
- Improve test coverage under `tests/`.
- Clarify data provenance under `data/metadata/`.


---

## License & references
- This repository is licensed under MIT licencing (see LICENSE file for details).
- Use `docs/references/papers/` to collect papers and presentations relevant to the project.

---

