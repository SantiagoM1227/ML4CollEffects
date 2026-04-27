
import os
import sys
# Ensure the scripts/ directory is on sys.path so we can import the generator module
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Import the generator module (script lives at scripts/data_generator_pycolleff.py)
from data_generator_pycolleff import (
    generate_dataset,
    save_dataset,
    DatasetConfig,
    GridConfig,
    RingConfig,
    ParameterRanges,
    ResonatorConfig,
    RelaxationConfig,
)

def main():
    dataset_cfg = DatasetConfig(n_samples=1024, particles_per_sample=4096, seed=42, output_dir="/pbs/home/s/smartinez/ML4CollEffects/data/neural")
    grid_cfg = GridConfig(nz=4096)
    ring_cfg = RingConfig()
    param_ranges = ParameterRanges(I_bunch=(200,400), radius=(5e-3,12e-3),neg_thickness=(1e-6,5e-6))
    resonator_cfg = ResonatorConfig(Q=1e4, wr_scale_to_rf=3.0, harm_rf=3)
    relax_cfg = RelaxationConfig(n_steps=7, mix=1e-3)

    n_jobs = int(os.environ.get('JOBLIB_CPUS', os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    data = generate_dataset(
        dataset_cfg,
        grid_cfg,
        ring_cfg,
        param_ranges,
        resonator_cfg,
        relax_cfg,
        n_jobs=n_jobs,
    )
    save_dataset(data, dataset_cfg, grid_cfg, ring_cfg, param_ranges, resonator_cfg, relax_cfg)

if __name__ == "__main__":
    main()