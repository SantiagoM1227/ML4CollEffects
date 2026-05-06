import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_generator_neural import (
    build_line,
    build_datasets,
    save_dataset_bundle,
    DatasetConfig,
    LatticeConfig,
    DensityGridConfig,
    ParameterRanges,
    WakeConfig,
    default_beam_families,
)

def main():


    dataset_cfg = DatasetConfig(
        n_samples=512,
        particles_per_sample=512,
        seed=42,
        output_dir="/pbs/home/s/smartinez/ML4CollEffects/data/neural",
        save_cloud_dataset=True,
        save_density_dataset=True,
        save_moments=True,
        train_fraction=0.8,
        val_fraction=0.1,
    )


    lattice_cfg = LatticeConfig(
        p0c_ev=1e9,
        l_mq=0.4,
        line_length=4.0,
    )


    density_cfg = DensityGridConfig(
        nz=512,
        zeta_min=-5e-3,
        zeta_max=5e-3,
        normalize_density=True,
    )


    param_ranges = ParameterRanges(
        bunch_charge_range=(0.1e-9, 3e-9),
        pipe_radius_range=(5e-3, 30e-3),
        impedance_scale_range=(0.1, 10.0),
    )


    wake_cfg = WakeConfig(
        kind="exponential",     # "gaussian" | "exponential" | "resonator"
        strength=1.0,
        sigma=1e-3,
    )

    use_collective = True


    beam_families = default_beam_families()


    line, env = build_line(lattice_cfg)

    n_jobs = int(
        os.environ.get(
            "JOBLIB_CPUS",
            os.environ.get("SLURM_CPUS_PER_TASK", "8")
        )
    )

    print(f"[INFO] Using {n_jobs} CPUs (generator currently serial)")
    

    data = build_datasets(
        line=line,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        param_ranges=param_ranges,
        beam_families=beam_families,
        use_collective=use_collective,
        wake_cfg=wake_cfg,
    )

   
    save_dataset_bundle(
        data=data,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        beam_families=beam_families,
        param_ranges=param_ranges,
    )

    print(f"[DONE] Dataset saved to {dataset_cfg.output_dir}")
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()