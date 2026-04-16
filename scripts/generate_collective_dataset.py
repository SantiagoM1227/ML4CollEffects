from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np


PHASE_SPACE_DIM = 6
PHASE_SPACE_LABELS = ("x", "y", "zeta", "px", "py", "delta")
MU_LABELS = ("Rs", "Q", "fres_hz")


def _load_longitudinal_tracking_module(module_path: Optional[Path] = None):
    """Load the collective-effects longitudinal tracking module.

    Resolution order:
    1) installed package: pycolleff.longitudinal_tracking
    2) local file next to this script: longitudinal_tracking.py
    3) explicit --tracking-module path
    """
    try:
        from pycolleff.longitudinal_tracking import Beam, Ring, Wake, track_particles
        return Beam, Ring, Wake, track_particles
    except Exception:
        pass

    candidate = module_path
    if candidate is None:
        candidate = Path(__file__).with_name("longitudinal_tracking.py")
    if not candidate.exists():
        raise ImportError(
            "Could not import pycolleff.longitudinal_tracking and no local "
            f"longitudinal_tracking.py was found at {candidate}."
        )

    spec = importlib.util.spec_from_file_location("collective_longitudinal_tracking", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {candidate}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Beam, module.Ring, module.Wake, module.track_particles


@dataclass
class BeamFamilyConfig:
    family: Literal[
        "gaussian",
        "correlated_gaussian",
        "core_halo",
        "mixture",
        "mismatched",
    ]
    ss_sigma: float = 1.2e-3
    de_sigma: float = 8.0e-4
    halo_scale: float = 4.0
    halo_fraction: float = 0.15
    mismatch_scale: float = 2.0
    corr_strength: float = 0.45
    mixture_shift_scale: float = 1.2


@dataclass
class RingConfig:
    energy_ev: float = 3.0e9
    u0_ev: float = 871e3
    harm_num: int = 864
    rf_freq_hz: float = 499_663_824.0
    espread: float = 8.43589e-4
    mom_comp: float = 1.6446e-4
    damping_number: float = 1.6944268554007123
    cav_vgap_v: float = 3.0e6
    use_gaussian_noise: bool = False


@dataclass
class WakeParameterRanges:
    Rs_range: Tuple[float, float] = (5.0e2, 8.0e4)
    Q_range: Tuple[float, float] = (0.8, 8.0)
    fres_hz_range: Tuple[float, float] = (0.5e9, 3.0e9)


@dataclass
class DensityGridConfig:
    nz: int = 256
    zeta_min: float = -5e-3
    zeta_max: float = 5e-3
    normalize_density: bool = True


@dataclass
class DatasetConfig:
    n_samples: int = 512
    particles_per_sample: int = 4096
    seed: int = 42
    output_dir: str = "./data/neural"
    current_a: float = 0.35
    num_turns: int = 200
    save_cloud_dataset: bool = True
    save_density_dataset: bool = True
    save_moments: bool = True
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    excitation: bool = False
    damping: bool = True
    stats_every_n_turns: int = 25


class CollectiveDatasetGenerator:
    def __init__(self, tracking_module: Optional[Path] = None):
        Beam, Ring, Wake, track_particles = _load_longitudinal_tracking_module(tracking_module)
        self.Beam = Beam
        self.Ring = Ring
        self.Wake = Wake
        self.track_particles = track_particles

    def build_ring(self, ring_cfg: RingConfig):
        ring = self.Ring()
        ring.energy = float(ring_cfg.energy_ev)
        ring.u0 = float(ring_cfg.u0_ev)
        ring.harm_num = int(ring_cfg.harm_num)
        ring.rf_freq = float(ring_cfg.rf_freq_hz)
        ring.espread = float(ring_cfg.espread)
        ring.mom_comp = float(ring_cfg.mom_comp)
        ring.damping_number = float(ring_cfg.damping_number)
        ring.use_gaussian_noise = bool(ring_cfg.use_gaussian_noise)
        ring.cav_vgap = float(ring_cfg.cav_vgap_v)
        return ring

    @staticmethod
    def sample_parameters(rng: np.random.Generator, param_ranges: WakeParameterRanges) -> np.ndarray:
        return np.array(
            [
                rng.uniform(*param_ranges.Rs_range),
                rng.uniform(*param_ranges.Q_range),
                rng.uniform(*param_ranges.fres_hz_range),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _cov_from_beam_family(cfg: BeamFamilyConfig) -> np.ndarray:
        sigmas = np.array([cfg.ss_sigma, cfg.de_sigma], dtype=np.float64)
        cov = np.diag(sigmas**2)
        if cfg.family == "correlated_gaussian":
            corr = cfg.corr_strength
            cov[0, 1] = cov[1, 0] = corr * sigmas[0] * sigmas[1]
        return cov

    @classmethod
    def sample_initial_longitudinal_cloud(
        cls,
        n_particles: int,
        rng: np.random.Generator,
        beam_cfg: BeamFamilyConfig,
    ) -> np.ndarray:
        sigmas = np.array([beam_cfg.ss_sigma, beam_cfg.de_sigma], dtype=np.float64)

        if beam_cfg.family == "gaussian":
            z = rng.normal(0.0, sigmas, size=(n_particles, 2))
            return z.astype(np.float64)

        if beam_cfg.family == "correlated_gaussian":
            cov = cls._cov_from_beam_family(beam_cfg)
            z = rng.multivariate_normal(np.zeros(2), cov, size=n_particles)
            return z.astype(np.float64)

        if beam_cfg.family == "core_halo":
            n_halo = int(round(beam_cfg.halo_fraction * n_particles))
            n_core = n_particles - n_halo
            core = rng.normal(0.0, sigmas, size=(n_core, 2))
            halo = rng.normal(0.0, beam_cfg.halo_scale * sigmas, size=(n_halo, 2))
            z = np.vstack([core, halo])
            rng.shuffle(z, axis=0)
            return z.astype(np.float64)

        if beam_cfg.family == "mixture":
            shift = beam_cfg.mixture_shift_scale * sigmas
            n_a = n_particles // 2
            n_b = n_particles - n_a
            a = rng.normal(-shift, sigmas, size=(n_a, 2))
            b = rng.normal(+shift, sigmas, size=(n_b, 2))
            z = np.vstack([a, b])
            rng.shuffle(z, axis=0)
            return z.astype(np.float64)

        if beam_cfg.family == "mismatched":
            scaled = np.array(sigmas)
            scaled[[0, 1]] *= beam_cfg.mismatch_scale
            z = rng.normal(0.0, scaled, size=(n_particles, 2))
            return z.astype(np.float64)

        raise ValueError(f"Unsupported beam family: {beam_cfg.family}")

    @staticmethod
    def longitudinal_to_6d(longitudinal_cloud: np.ndarray) -> np.ndarray:
        z6 = np.zeros((longitudinal_cloud.shape[0], PHASE_SPACE_DIM), dtype=np.float64)
        z6[:, 2] = longitudinal_cloud[:, 0]  # zeta <- ss
        z6[:, 5] = longitudinal_cloud[:, 1]  # delta <- de
        return z6

    def track_cloud(
        self,
        ring,
        z0_long: np.ndarray,
        mu: np.ndarray,
        dataset_cfg: DatasetConfig,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        beam = self.Beam(num_part=z0_long.shape[0], num_buns=1, current=dataset_cfg.current_a)
        beam.ss[0, :] = z0_long[:, 0]
        beam.de[0, :] = z0_long[:, 1]

        Rs, Q, fres_hz = [float(x) for x in mu]
        wake = self.Wake(Q=Q, Rs=Rs, wr=2.0 * math.pi * fres_hz)
        stats = self.track_particles(
            ring=ring,
            beam=beam,
            wakes=[wake],
            num_turns=int(dataset_cfg.num_turns),
            stats_ev_nt=max(1, int(dataset_cfg.stats_every_n_turns)),
            dist_ev_nt=max(1, int(dataset_cfg.num_turns)),
            print_progress=False,
            save_dist=False,
            excitation=bool(dataset_cfg.excitation),
            damping=bool(dataset_cfg.damping),
        )
        z1_long = np.column_stack([beam.ss[0].copy(), beam.de[0].copy()]).astype(np.float64)
        return z1_long, stats


def cloud_moments(z: np.ndarray) -> Dict[str, np.ndarray]:
    centroid = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    return {"centroid": centroid, "cov": cov}



def line_density_from_cloud(
    z: np.ndarray,
    density_cfg: DensityGridConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, normalized_histogram) for zeta marginal."""
    hist, edges = np.histogram(
        z[:, 2],
        bins=density_cfg.nz,
        range=(density_cfg.zeta_min, density_cfg.zeta_max),
        density=False,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)
    if density_cfg.normalize_density:
        dz = (density_cfg.zeta_max - density_cfg.zeta_min) / density_cfg.nz
        norm = hist.sum() * dz
        if norm > 0:
            hist = hist / norm
    return centers, hist



def split_indices(
    n: int,
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be in (0, 1).")
    if not (0 <= val_fraction < 1):
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}



def build_datasets(
    generator: CollectiveDatasetGenerator,
    ring_cfg: RingConfig,
    dataset_cfg: DatasetConfig,
    density_cfg: DensityGridConfig,
    param_ranges: WakeParameterRanges,
    beam_families: Iterable[BeamFamilyConfig],
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(dataset_cfg.seed)
    families = list(beam_families)
    if not families:
        raise ValueError("At least one beam family must be provided.")

    ring = generator.build_ring(ring_cfg)

    cloud_in, cloud_out, mu_all = [], [], []
    lambda_in, lambda_out, zeta_grid = [], [], None
    moments_in_centroid, moments_out_centroid = [], []
    moments_in_cov, moments_out_cov = [], []
    family_id = []
    stats_std_ss = []
    stats_std_de = []
    stats_final_pot = []

    for _ in range(dataset_cfg.n_samples):
        beam_cfg = families[rng.integers(0, len(families))]
        mu = generator.sample_parameters(rng, param_ranges)

        z0_long = generator.sample_initial_longitudinal_cloud(
            dataset_cfg.particles_per_sample,
            rng,
            beam_cfg,
        )
        z1_long, stats = generator.track_cloud(ring, z0_long, mu, dataset_cfg)

        z0 = generator.longitudinal_to_6d(z0_long)
        z1 = generator.longitudinal_to_6d(z1_long)

        if dataset_cfg.save_cloud_dataset:
            cloud_in.append(z0.astype(np.float32))
            cloud_out.append(z1.astype(np.float32))
            mu_all.append(mu.astype(np.float32))

        if dataset_cfg.save_density_dataset:
            grid, lam0 = line_density_from_cloud(z0, density_cfg)
            _, lam1 = line_density_from_cloud(z1, density_cfg)
            zeta_grid = grid.astype(np.float32)
            lambda_in.append(lam0.astype(np.float32))
            lambda_out.append(lam1.astype(np.float32))

        if dataset_cfg.save_moments:
            m0 = cloud_moments(z0)
            m1 = cloud_moments(z1)
            moments_in_centroid.append(m0["centroid"].astype(np.float32))
            moments_out_centroid.append(m1["centroid"].astype(np.float32))
            moments_in_cov.append(m0["cov"].astype(np.float32))
            moments_out_cov.append(m1["cov"].astype(np.float32))

        family_id.append(beam_cfg.family)
        stats_std_ss.append(float(stats["std_ss"][-1, 0]))
        stats_std_de.append(float(stats["std_de"][-1, 0]))
        if stats["pot_wakes"].size:
            stats_final_pot.append(np.complex128(stats["pot_wakes"][-1, 0]))
        else:
            stats_final_pot.append(np.complex128(0.0))

    out: Dict[str, np.ndarray] = {}
    if cloud_in:
        out["X_cloud"] = np.stack(cloud_in, axis=0)
        out["Y_cloud"] = np.stack(cloud_out, axis=0)
        out["MU"] = np.stack(mu_all, axis=0)
    if lambda_in:
        out["zeta_grid"] = zeta_grid
        out["X_lambda"] = np.stack(lambda_in, axis=0)
        out["Y_lambda"] = np.stack(lambda_out, axis=0)
        if "MU" not in out:
            out["MU"] = np.stack(mu_all, axis=0)
    if moments_in_centroid:
        out["centroid_in"] = np.stack(moments_in_centroid, axis=0)
        out["centroid_out"] = np.stack(moments_out_centroid, axis=0)
        out["cov_in"] = np.stack(moments_in_cov, axis=0)
        out["cov_out"] = np.stack(moments_out_cov, axis=0)

    out["family_id"] = np.array(family_id)
    out["std_ss_out"] = np.array(stats_std_ss, dtype=np.float32)
    out["std_de_out"] = np.array(stats_std_de, dtype=np.float32)
    out["wake_potential_final"] = np.array(stats_final_pot)
    return out



def save_dataset_bundle(
    data: Dict[str, np.ndarray],
    dataset_cfg: DatasetConfig,
    density_cfg: DensityGridConfig,
    ring_cfg: RingConfig,
    beam_families: Iterable[BeamFamilyConfig],
    param_ranges: WakeParameterRanges,
) -> None:
    out_dir = Path(dataset_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = None
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > 1 and key != "zeta_grid":
            n_samples = value.shape[0]
            break
    if n_samples is None:
        raise RuntimeError("Could not infer dataset size.")

    rng = np.random.default_rng(dataset_cfg.seed + 17)
    split = split_indices(n_samples, dataset_cfg.train_fraction, dataset_cfg.val_fraction, rng)

    np.savez(out_dir / "xsuite_neural_dataset.npz", **data, **split)

    metadata = {
        "source": "collective_effects.longitudinal_tracking",
        "phase_space_labels": PHASE_SPACE_LABELS,
        "mu_labels": MU_LABELS,
        "dataset_config": asdict(dataset_cfg),
        "density_grid_config": asdict(density_cfg),
        "ring_config": asdict(ring_cfg),
        "wake_parameter_ranges": asdict(param_ranges),
        "beam_families": [asdict(b) for b in beam_families],
        "keys": sorted(list(data.keys()) + list(split.keys())),
        "notes": {
            "X_cloud/Y_cloud": "Pseudo-6D clouds with transverse coordinates set to zero; zeta<-ss and delta<-de.",
            "X_lambda/Y_lambda": "Initial/final line-density histograms on zeta_grid.",
            "MU": list(MU_LABELS),
        },
    }
    (out_dir / "xsuite_neural_dataset_metadata.json").write_text(json.dumps(metadata, indent=2))



def default_beam_families() -> List[BeamFamilyConfig]:
    return [
        BeamFamilyConfig(family="gaussian"),
        BeamFamilyConfig(family="correlated_gaussian"),
        BeamFamilyConfig(family="core_halo"),
        BeamFamilyConfig(family="mixture"),
        BeamFamilyConfig(family="mismatched"),
    ]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate collective-effects longitudinal datasets in the same .npz layout used by Neural_Operators_FNO.ipynb."
    )
    parser.add_argument("--tracking-module", type=str, default="", help="Optional path to longitudinal_tracking.py")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--particles-per-sample", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./data/neural")
    parser.add_argument("--current-a", type=float, default=0.35)
    parser.add_argument("--num-turns", type=int, default=200)
    parser.add_argument("--stats-every-n-turns", type=int, default=25)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--zeta-min", type=float, default=-5e-3)
    parser.add_argument("--zeta-max", type=float, default=5e-3)
    parser.add_argument("--Rs-min", type=float, default=5.0e2)
    parser.add_argument("--Rs-max", type=float, default=8.0e4)
    parser.add_argument("--Q-min", type=float, default=0.8)
    parser.add_argument("--Q-max", type=float, default=8.0)
    parser.add_argument("--fres-min", type=float, default=0.5e9)
    parser.add_argument("--fres-max", type=float, default=3.0e9)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--skip-cloud", action="store_true")
    parser.add_argument("--skip-density", action="store_true")
    parser.add_argument("--disable-moments", action="store_true")
    parser.add_argument("--enable-excitation", action="store_true", help="Turn on stochastic excitation. Disabled by default so the operator remains deterministic.")
    parser.add_argument("--disable-damping", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    tracking_module = Path(args.tracking_module) if args.tracking_module else None
    generator = CollectiveDatasetGenerator(tracking_module)

    ring_cfg = RingConfig()
    dataset_cfg = DatasetConfig(
        n_samples=args.n_samples,
        particles_per_sample=args.particles_per_sample,
        seed=args.seed,
        output_dir=args.output_dir,
        current_a=args.current_a,
        num_turns=args.num_turns,
        save_cloud_dataset=not args.skip_cloud,
        save_density_dataset=not args.skip_density,
        save_moments=not args.disable_moments,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        excitation=args.enable_excitation,
        damping=not args.disable_damping,
        stats_every_n_turns=args.stats_every_n_turns,
    )
    density_cfg = DensityGridConfig(
        nz=args.nz,
        zeta_min=args.zeta_min,
        zeta_max=args.zeta_max,
    )
    param_ranges = WakeParameterRanges(
        Rs_range=(args.Rs_min, args.Rs_max),
        Q_range=(args.Q_min, args.Q_max),
        fres_hz_range=(args.fres_min, args.fres_max),
    )
    beam_families = default_beam_families()

    data = build_datasets(
        generator=generator,
        ring_cfg=ring_cfg,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        param_ranges=param_ranges,
        beam_families=beam_families,
    )
    save_dataset_bundle(
        data=data,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        ring_cfg=ring_cfg,
        beam_families=beam_families,
        param_ranges=param_ranges,
    )

    print("Saved dataset bundle to", dataset_cfg.output_dir)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
