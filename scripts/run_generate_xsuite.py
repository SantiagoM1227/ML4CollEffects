import os
import sys
import numpy as np

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

def _load_wake_from_txt_env():
    """
    Optional external wake loading via env vars:
      - WAKE_TXT_PATH: two-column txt (s/mm, W), '#' comments supported.
      - WAKE_FLIP (or WAKE_DIAG_FLIP): if true/1/yes, reverse wake arrays.
    Returns (zeta_grid_m, W) tuple or None.
    """
    wake_txt_path = os.environ.get("WAKE_TXT_PATH", "").strip()
    if not wake_txt_path:
        return None

    try:
        raw = np.loadtxt(wake_txt_path, comments="#", ndmin=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to load WAKE_TXT_PATH file '{wake_txt_path}': {exc}") from exc
    if raw.shape[1] < 2:
        raise ValueError(f"WAKE_TXT_PATH must point to a 2-column txt file, got shape {raw.shape} at {wake_txt_path}")
    zeta_grid = raw[:, 0].astype(np.float64) * 1e-3  # mm -> m
    wake_values = raw[:, 1].astype(np.float64)

    flip_flag = os.environ.get("WAKE_FLIP", os.environ.get("WAKE_DIAG_FLIP", "0")).strip().lower()
    if flip_flag in {"1", "true", "yes", "on"}:
        zeta_grid = zeta_grid[::-1].copy()
        wake_values = wake_values[::-1].copy()

    print(f"[INFO] Using external wake TXT from WAKE_TXT_PATH: {wake_txt_path}")
    return (zeta_grid, wake_values)

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

    impedance = _load_wake_from_txt_env()
    if impedance is not None:
        wake_cfg = None


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
        impedance=impedance,
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
