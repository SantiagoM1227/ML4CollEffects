from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pycolleff.colleff import Ring
from pycolleff.longitudinal_equilibrium import (
    LongitudinalEquilibrium,
    ImpedanceSource,
)

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from pycolleff.longitudinal_equilibrium import ImpedanceSource, LongitudinalEquilibrium
import pycolleff.rings.sirius as sirius
import pycolleff.impedances as imp
import pycolleff.materials_params as mat_par
from numba import njit

@dataclass
class GridConfig:
    nz: int = 512
    sigmas: float = 6.0


def calculate_longitudinal_equilibrium(ring, impedance_sources, fill=None):
    """
    Calculate longitudinal equilibrium.
    """
    if fill is None:
        fill = np.ones(ring.harm_num) / ring.harm_num
    
    longeq = LongitudinalEquilibrium(
        ring=ring, impedance_sources=impedance_sources, fillpattern=fill)
    longeq.feedback_on = False  # main cavity is simulated with effective impedance
    longeq.zgrid = np.linspace(-1, 1, 2001) * ring.rf_lamb / 2
    longeq.max_mode = 1000*ring.harm_num  # define maximum frequency to consider
    longeq.min_mode0_ratio = 1e-10  # criteria for convergence

    print('Calculating Longitudinal Equilibrium...')
    _ = longeq.calc_longitudinal_equilibrium(
        niter=1000, tol=1e-8, beta=0.1, m=3, print_flag=True)
    return longeq


def print_results(longeq: LongitudinalEquilibrium, gc: GridConfig, uniform=True) -> None:
    """
    Print and plot the results of longitudinal equilibrium.
    """
    z0, sigmaz = longeq.calc_moments(longeq.zgrid, longeq.distributions)
    n_sigmas = gc.sigmas
    if uniform:
        print(f'Bunch 0 centroid: {z0[0]*1e3:.2f} mm')
        print(f'Bunch 0 length: {sigmaz[0]*1e3:.2f} mm')
        plt.figure()
        plt.plot(longeq.zgrid*1e3, longeq.distributions[0])
        plt.xlabel(r'$z$ [mm]')
        plt.ylabel(r'$\lambda(z)$ [1/mm]')
        plt.xlim((z0-n_sigmas*sigmaz)*1e3, (z0+n_sigmas*sigmaz)*1e3)
        plt.tight_layout()
        plt.show()
    else:
        fig, (af, ax, ay) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        fill = longeq.fillpattern.copy()
        idcs = np.arange(fill.size)
        zef = np.isclose(fill, 0)
        af.plot(idcs, fill*longeq.ring.total_current*1e3, 'o')
        ax.plot(idcs[~zef], z0[~zef]*1e3, 'o')
        ay.plot(idcs[~zef], sigmaz[~zef]*1e3, 'o')
        ax.plot(idcs[zef], z0[zef]*1e3, 'o', color='gray')
        ay.plot(idcs[zef], sigmaz[zef]*1e3, 'o', color='gray')
        af.set_ylabel('Filling Pattern [mA]')
        ax.set_ylabel('Centroid [mm]')
        ay.set_ylabel('Bunch Length [mm]')
        ay.set_xlabel('Bunch Index')
        fig.tight_layout()
        plt.show()


@dataclass
class DatasetConfig:
    n_samples: int = 512
    particles_per_sample: int = 4096
    seed: int = 42
    output_dir: str = "/pbs/home/s/smartinez/ML4CollEffects/data/neural"
    train_fraction: float = 0.8
    val_fraction: float = 0.1





@dataclass
class RingConfig:
    energy_eV: float = 3.0e9
    rf_freq_Hz: float = 500e6
    harm_num: int = 1
    mom_comp: float = 1.7e-4
    sync_tune: float = 0.004
    espread: float = 8.0e-4
    bunlen_m: float = 4.0e-3
    en_lost_rad_eV: float = 8.0e5
    gap_voltage_V: float = 3.0e6
    total_current_A: float = 0.1



@dataclass
class ParameterRanges:
    # mu will be interpreted as [bunch current, radius, neg thickness]
    I_bunch : Tuple[float, float] = (200, 350)  # [A]
    radius : Tuple[float, float] = (5e-3, 12e-3)  # [m]
    neg_thickness : Tuple[float, float] = (1e-6, 3e-6)  # [m]


@dataclass
class ResonatorConfig:
    Q: float = 1.0e4
    wr_scale_to_rf: float = 3.0   # resonator angular freq = wr_scale * rf_ang_freq
    harm_rf: int = 3


@dataclass
class RelaxationConfig:
    n_steps: int = 3
    mix: float = 0.7   # under-relaxation: dist <- (1-mix) dist + mix new_dist
    


@njit
def normalize_density(z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 0.0)
    area = np.trapz(lam, z)
    if area <= 0:
        raise ValueError("Density integral must be positive.")
    return lam / area



def sample_mu(rng: np.random.Generator, pr: ParameterRanges) -> np.ndarray:
    # return order: [bunch current, radius, neg thickness]
    return np.array(
        [
            rng.uniform(*pr.I_bunch),
            rng.uniform(*pr.radius),
            rng.uniform(*pr.neg_thickness)
        ],
        dtype=np.float64,
    )


def generate_sample(
    sample_idx: int,
    seed: int,
    dataset_cfg: DatasetConfig,
    grid_cfg: GridConfig,
    ring_cfg: RingConfig,
    param_ranges: ParameterRanges,
    resonator_cfg: ResonatorConfig,
    relax_cfg: RelaxationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + sample_idx)
    mu = sample_mu(rng, param_ranges)

    ring = sirius.create_ring()
    ring.harm_num = 1
    ring.num_bun = 1
    ring.total_current = float(mu[0])
    radius_val = float(mu[1])
    neg_thick = float(mu[2])

    hcav = ImpedanceSource()
    hcav.harm_rf = resonator_cfg.harm_rf
    hcav.Q = resonator_cfg.Q
    RoverQ = 87.5
    hcav.shunt_impedance = RoverQ * hcav.Q
    twopi = 2 * np.pi
    hcav.ang_freq_rf = twopi * ring.rf_freq
    hcav.ang_freq = hcav.harm_rf * hcav.ang_freq_rf
    hcav.detune_w = twopi * 45e3
    hcav.calc_method = ImpedanceSource.Methods.ImpedanceDFT
    hcav.active_passive = ImpedanceSource.ActivePassive.Passive

    cu_cond = 59e6
    cu_rel_time = 2.7e-14
    neg_cond = 1e6
    neg_rel_time = 0.0
    ndfe_cond = 1e6
    ndfe_mur = 40
    energy = 3e9  # [eV]
    length = 500  # [m]

    epb = np.array([1, 1, 1, 1, 1])
    mub = np.array([1, 1, 1, 1, ndfe_mur])
    ange = np.array([0, 0, 0, 0, 0])
    angm = np.array([0, 0, 0, 0, 0])
    sigmadc = np.array([0, neg_cond, cu_cond, 0, ndfe_cond])
    tau = np.array([0, neg_rel_time, cu_rel_time, 0, 0])

    radius = radius_val + np.array([-neg_thick, 0, 1e-3, 3e-3])

    ang_freq = imp.get_default_reswall_w(radius=radius[0], energy=energy)
    epr, mur = imp.prepare_inputs_epr_mur(
        ang_freq, epb, mub, ange, angm, sigmadc, tau)

    Zll, Zdx, Zdy = imp.multilayer_round_chamber(
        ang_freq,
        length,
        energy,
        epr,
        mur,
        radius,
        precision=70,
        wmax_arb_prec=1e12,
        arb_prec_incl_long=False,
        print_progress=True,
    )

    Zll, ang_freq = imp.get_impedance_for_negative_w(
        Zll, ang_freq, impedance_type='ll'
    )

    generic = ImpedanceSource()
    generic.zl_table = Zll
    generic.ang_freq_table = ang_freq
    generic.calc_method = ImpedanceSource.Methods.ImpedanceDFT
    generic.active_passive = ImpedanceSource.ActivePassive.Passive

    impedance_sources = [generic, hcav]

    leq = calculate_longitudinal_equilibrium(
        ring=ring,
        impedance_sources=impedance_sources,
    )

    zgrid = leq.create_zgrid(nr_points=grid_cfg.nz, sigmas=grid_cfg.sigmas)
    leq.zgrid = zgrid

    lam_in = make_input_profile(zgrid, rng)
    dist_in = lam_in[None, :]
    dist_out = one_relaxation_map(leq, dist_in, relax_cfg)
    lam_out = dist_out[0]

    zeta_in_cloud = sample_cloud_from_density(
        zgrid, lam_in, dataset_cfg.particles_per_sample, rng
    )
    zeta_out_cloud = sample_cloud_from_density(
        zgrid, lam_out, dataset_cfg.particles_per_sample, rng
    )

    x_cloud = embed_zeta_cloud_in_6d(zeta_in_cloud)
    y_cloud = embed_zeta_cloud_in_6d(zeta_out_cloud)

    return (
        lam_in.astype(np.float32),
        lam_out.astype(np.float32),
        x_cloud,
        y_cloud,
        mu.astype(np.float32),
        zgrid.astype(np.float32),
    )

def build_ring(base: RingConfig, mu: np.ndarray) -> Ring:
    ring = Ring()
    ring.energy = base.energy_eV
    ring.rf_freq = base.rf_freq_Hz
    ring.harm_num = base.harm_num
    ring.num_bun = base.harm_num
    ring.mom_comp = base.mom_comp
    ring.sync_tune = base.sync_tune
    ring.espread = base.espread
    ring.bunlen = base.bunlen_m
    ring.en_lost_rad = base.en_lost_rad_eV

    # MU[0], MU[1], MU[2] are interpreted according to ParameterRanges
    # caller may override ring parameters after creation as needed
    ring.gap_voltage = base.gap_voltage_V
    ring.total_current = base.total_current_A
    return ring


def build_impedance_sources(
    ring: Ring,
    mu: np.ndarray,
    resonator_cfg: ResonatorConfig,
) -> list[ImpedanceSource]:
    # This function can be adapted to use mu semantics when needed.
    # Currently, mu[0]=en_lost_rad_eV, mu[1]=gap_voltage_V, mu[2]=total_current_A
    # Example: build a simple resonator source based on resonator_cfg and ring
    src = ImpedanceSource(
        Rs=float(1.0e4),  # placeholder; adapt if you want mu to carry shunt impedance
        Q=float(resonator_cfg.Q),
        ang_freq=float(resonator_cfg.wr_scale_to_rf * ring.rf_ang_freq),
        harm_rf=int(resonator_cfg.harm_rf),
        calc_method=ImpedanceSource.Methods.ImpedanceDFT,
    )
    src.ang_freq_rf = ring.rf_ang_freq
    return [src]


@njit
def make_input_profile(
    z: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # diverse 1D family, but still simple enough for your current FNO
    family = rng.integers(0, 4)

    if family == 0:
        # centered gaussian
        sigma = rng.uniform(0.5e-3, 1.8e-3)
        z0 = rng.uniform(-0.3e-3, 0.3e-3)
        lam = np.exp(-0.5 * ((z - z0) / sigma) ** 2)

    elif family == 1:
        # off-centered gaussian
        sigma = rng.uniform(0.7e-3, 2.2e-3)
        z0 = rng.uniform(-1.2e-3, 1.2e-3)
        lam = np.exp(-0.5 * ((z - z0) / sigma) ** 2)

    elif family == 2:
        # two-gaussian mixture
        sigma1 = rng.uniform(0.5e-3, 1.5e-3)
        sigma2 = rng.uniform(0.5e-3, 1.8e-3)
        z01 = rng.uniform(-1.3e-3, -0.2e-3)
        z02 = rng.uniform(0.2e-3, 1.3e-3)
        a2 = rng.uniform(0.3, 0.9)
        lam = np.exp(-0.5 * ((z - z01) / sigma1) ** 2)
        lam += a2 * np.exp(-0.5 * ((z - z02) / sigma2) ** 2)

    else:
        # slightly modulated gaussian
        sigma = rng.uniform(0.8e-3, 1.8e-3)
        eps = rng.uniform(0.05, 0.25)
        k = rng.integers(2, 6)
        base = np.exp(-0.5 * (z / sigma) ** 2)
        # NumPy 2.0 removed ndarray.ptp(); use np.ptp(z) for compatibility
        lam = base * (1.0 + eps * np.cos(2 * np.pi * k * (z - z.min()) / (np.ptp(z) + 1e-15)))

    return normalize_density(z, lam).astype(np.float64)



def one_relaxation_map(
    leq: LongitudinalEquilibrium,
    dist_in: np.ndarray,              # shape (1, Nz)
    relax_cfg: RelaxationConfig,
) -> np.ndarray:
    """
    Apply a small number of collective-effect relaxation steps.
    This preserves dependence on lambda_in, which is what your current model needs.
    """
    dist = dist_in.copy()

    for _ in range(relax_cfg.n_steps):
        v_main = leq.main_voltage[0]  # shape (Nz,)
        v_ind = leq.calc_induced_voltage_impedance_dft(dist=dist)[0]  # shape (Nz,)
        v_tot = v_main + v_ind

        # Ensure v_tot is 2D: (1, Nz)
        if v_tot.ndim == 1:
            v_tot = v_tot[None, :]

        dist_new, _ = leq.calc_distributions_from_voltage(v_tot)
        #dist_new has shape (harm_num, Nz). Here harm_num=1.
        dist = (1.0 - relax_cfg.mix) * dist + relax_cfg.mix * dist_new

    # Plotting is disabled for parallel generation to avoid worker GUI issues.
    # plt.plot(leq.zgrid*1e3, dist[0])
    # plt.xlabel('z [mm]')
    # plt.ylabel('lambda(z) [1/mm]')
    # plt.title(f"Relaxation step {_+1}/{relax_cfg.n_steps}")
    # plt.xlim(-5,5)
    # plt.show()

        # renormalize explicitly for safety
        dist[0] = normalize_density(leq.zgrid, dist[0])

    return dist



@njit
def split_indices(
    n: int,
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }



def generate_dataset(
    dataset_cfg: DatasetConfig,
    grid_cfg: GridConfig,
    ring_cfg: RingConfig,
    param_ranges: ParameterRanges,
    resonator_cfg: ResonatorConfig,
    relax_cfg: RelaxationConfig,
    n_jobs: int = 1,
) -> Dict[str, np.ndarray]:
    seed = dataset_cfg.seed

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(generate_sample)(i, seed, dataset_cfg, grid_cfg, ring_cfg, param_ranges, resonator_cfg, relax_cfg)
        for i in range(dataset_cfg.n_samples)
    )

    X_lambda = np.stack([r[0] for r in results], axis=0)
    Y_lambda = np.stack([r[1] for r in results], axis=0)
    X_cloud = np.stack([r[2] for r in results], axis=0)
    Y_cloud = np.stack([r[3] for r in results], axis=0)
    MU = np.stack([r[4] for r in results], axis=0)
    zeta_grid_ref = results[0][5] if results else None

    X_lambda = np.stack(X_lambda, axis=0)
    Y_lambda = np.stack(Y_lambda, axis=0)
    X_cloud = np.stack(X_cloud, axis=0)
    Y_cloud = np.stack(Y_cloud, axis=0)
    MU = np.stack(MU, axis=0)

    split = split_indices(
        n=X_lambda.shape[0],
        train_fraction=dataset_cfg.train_fraction,
        val_fraction=dataset_cfg.val_fraction,
        rng=np.random.default_rng(dataset_cfg.seed + 17),
    )

    out = {
        "X_lambda": X_lambda,
        "Y_lambda": Y_lambda,
        "X_cloud": X_cloud,
        "Y_cloud": Y_cloud,
        "MU": MU,
        "zeta_grid": zeta_grid_ref,
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
    }
    return out



@njit
def sample_cloud_from_density(
    zgrid: np.ndarray,
    lam: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample zeta particles from a 1D line density lambda(zeta).
    Returns shape (n_particles,).
    """
    lam = np.maximum(lam, 0.0)
    cdf = np.cumsum(lam)
    if cdf[-1] <= 0:
        raise ValueError("Density must have positive mass.")
    cdf = cdf / cdf[-1]

    u = rng.uniform(0.0, 1.0, size=n_particles)
    zeta = np.interp(u, cdf, zgrid)
    return zeta.astype(np.float32)


@njit
def embed_zeta_cloud_in_6d(zeta_cloud: np.ndarray) -> np.ndarray:
    """
    Build a fake 6D cloud [x, y, zeta, px, py, delta]
    with only zeta populated.
    Shape: (n_particles, 6)
    """
    n = zeta_cloud.shape[0]
    cloud = np.zeros((n, 6), dtype=np.float32)
    cloud[:, 2] = zeta_cloud
    return cloud


def save_dataset(
    data: Dict[str, np.ndarray],
    dataset_cfg: DatasetConfig,
    grid_cfg: GridConfig,
    ring_cfg: RingConfig,
    param_ranges: ParameterRanges,
    resonator_cfg: ResonatorConfig,
    relax_cfg: RelaxationConfig,
) -> None:
    out_dir = Path(dataset_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save with the SAME filename your existing model already expects.
    time_of_generation = np.datetime64("now").astype(str)
    filename = f"neural_pycolleff_dataset_{time_of_generation}.npz"
    np.savez(out_dir / filename, **data)

    metadata = {
        "generator": "pycolleff",
        "dataset_config": asdict(dataset_cfg),
        "grid_config": asdict(grid_cfg),
        "ring_config": asdict(ring_cfg),
        "parameter_ranges": asdict(param_ranges),
        "resonator_config": asdict(resonator_cfg),
        "relaxation_config": asdict(relax_cfg),
        # reflect new MU semantics
        "mu_semantics": ["I_bunch", "radius","neg_thickness"],
        "keys": sorted(data.keys()),
    }
    print("Saving dataset metadata:")
    print(json.dumps(metadata, indent=2))
    (out_dir / "xsuite_neural_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--particles-per-sample", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="./data/neural")
    p.add_argument("--nz", type=int, default=256)

    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--bunch-current-min", type=float, default=200)
    p.add_argument("--bunch-current-max", type=float, default=300)

    p.add_argument("--radius-min", type=float, default=5e-3)
    p.add_argument("--radius-max", type=float, default=12e-3)

    p.add_argument("--neg-thickness-min", type=float, default=1e-6)
    p.add_argument("--neg-thickness-max", type=float, default=3e-6)

    p.add_argument("--Q", type=float, default=1.0e4)
    p.add_argument("--wr-scale", type=float, default=3.0)

    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--mix", type=float, default=0.7)
    return p.parse_args()

