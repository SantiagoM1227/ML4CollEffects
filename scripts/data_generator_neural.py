"""
Generate Xsuite datasets for neural operator learning with simplified collective effects.

mu = [Q_bunch, pipe_radius, impedance_scale]

The generator includes:
- external transport (linear optics)
- simple longitudinal collective kick (wake-like model)

Designed for operator learning:
    lambda_out = G(lambda_in, mu)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import xtrack as xt


import os
import re
from pathlib import Path
from cpymad.madx import Madx

INCLUDE_RE = re.compile(r'^\s*(?:call|include)\s*,\s*file\s*=\s*["\']?([^,"\']+)["\']?', flags=re.IGNORECASE | re.MULTILINE)


PHASE_SPACE_DIM = 6
PHASE_SPACE_LABELS = ("x", "y", "zeta", "px", "py", "delta")


@dataclass
class BeamFamilyConfig:
    family: Literal[
        "gaussian",
        "correlated_gaussian",
        "core_halo",
        "mixture",
        "mismatched"
    ]
    x_sigma: float = 1e-3
    y_sigma: float = 1e-3
    zeta_sigma: float = 1e-3
    px_sigma: float = 1e-3
    py_sigma: float = 1e-3
    delta_sigma: float = 1e-4
    halo_scale: float = 4.0
    halo_fraction: float = 0.15
    mismatch_scale: float = 2.0
    corr_strength: float = 0.45
    mixture_shift_scale: float = 1.2


@dataclass
class LatticeConfig:
    p0c_ev: float = 1e9
    mass0_ev: float = xt.PROTON_MASS_EV
    l_mq: float = 0.4
    line_length: float = 4.0
    qf1_at: float = 1.0
    qd1_at: float = 2.0
    qf2_at: float = 3.0

@dataclass
class WakeConfig:
    kind: Literal["gaussian", "exponential", "resonator"] = "gaussian"
    strength: float = 1.0
    sigma: float = 1e-3
    decay: float = 1e3
    freq: float = 1e10

@dataclass
class ParameterRanges:
    bunch_charge_range: Tuple[float, float] = (0.1e-9, 3e-9)   # Coulombs
    pipe_radius_range: Tuple[float, float] = (5e-3, 30e-3)     # meters
    impedance_scale_range: Tuple[float, float] = (0.1, 10.0)   # dimensionless


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
    save_cloud_dataset: bool = True
    save_density_dataset: bool = True
    save_moments: bool = True
    train_fraction: float = 0.8
    val_fraction: float = 0.1

def build_wake_kernel(zeta_grid, wake_cfg):
    z = zeta_grid - zeta_grid.mean()

    if wake_cfg.kind == "gaussian":
        W = np.exp(-0.5 * (z / wake_cfg.sigma)**2)

    elif wake_cfg.kind == "exponential":
        W = np.exp(-wake_cfg.decay * np.maximum(z, 0.0))

    elif wake_cfg.kind == "resonator":
        W = np.sin(2*np.pi*wake_cfg.freq*z) * np.exp(-np.abs(z)/wake_cfg.sigma)

    else:
        raise ValueError

    return wake_cfg.strength * W / (np.abs(W).sum() + 1e-12)

def apply_wake(lambda_z, W):
    n = len(lambda_z)
    return np.convolve(lambda_z, W, mode="same")

def build_wake_kernel(zeta_grid: np.ndarray, wake_cfg) -> np.ndarray:
    """Fallback wake builder (keeps your earlier semantics)."""
    z = zeta_grid - zeta_grid.mean()
    kind = getattr(wake_cfg, "kind", "gaussian")
    if kind == "gaussian":
        W = np.exp(-0.5 * (z / wake_cfg.sigma) ** 2)
    elif kind == "exponential":
        W = np.exp(-wake_cfg.decay * np.maximum(z, 0.0))
    elif kind == "resonator":
        W = np.sin(2 * np.pi * wake_cfg.freq * z) * np.exp(-np.abs(z) / wake_cfg.sigma)
    else:
        raise ValueError(f"Unknown wake kind: {kind}")
    W = float(getattr(wake_cfg, "strength", 1.0)) * W
    # normalize (same normalization as before)
    W = W / (np.abs(W).sum() + 1e-12)
    return W


def load_impedance(
    impedance: Optional[Union[str, Tuple[np.ndarray, np.ndarray]]],
    density_cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve an impedance/wake specification into (zeta_grid, W).
    impedance may be:
      - None: returns (None, None)
      - tuple (zeta_grid, W)
      - path to .npz (keys: 'zeta_grid' and 'W' or 'wake') or .npy (W only)
      - 1D numpy array W (then zeta_grid from density_cfg is used)
    """
    if impedance is None:
        return None, None

    # tuple already
    if isinstance(impedance, tuple) and len(impedance) == 2:
        zg, W = impedance
        return np.asarray(zg), np.asarray(W)

    # numpy array directly
    if isinstance(impedance, np.ndarray):
        nz = density_cfg.nz
        zg = np.linspace(density_cfg.zeta_min, density_cfg.zeta_max, nz)
        return zg, impedance

    # path string
    if isinstance(impedance, str) or isinstance(impedance, Path):
        p = Path(impedance)
        if not p.exists():
            raise FileNotFoundError(f"Impedance file not found: {impedance}")
        if p.suffix == ".npz":
            d = np.load(str(p))
            # common keys
            if "zeta_grid" in d and ("W" in d or "wake" in d):
                zg = d["zeta_grid"]
                W = d.get("W", d.get("wake"))
                return np.asarray(zg), np.asarray(W)
            # fallback: keys 'z' and 'Z' etc.
            if "z" in d and "W" in d:
                return np.asarray(d["z"]), np.asarray(d["W"])
            raise RuntimeError(f".npz does not contain recognized keys (zeta_grid,W) in {p}")
        elif p.suffix == ".npy":
            W = np.load(str(p))
            nz = density_cfg.nz
            zg = np.linspace(density_cfg.zeta_min, density_cfg.zeta_max, nz)
            return zg, np.asarray(W)
        else:
            raise RuntimeError(f"Unsupported impedance file extension: {p.suffix}")

    raise RuntimeError("Unsupported impedance spec. Provide (zeta_grid,W), np.array(W), or filepath to .npz/.npy")


def track_with_collective_effects(
    line: xt.Line,
    z0: np.ndarray,
    mu: np.ndarray,
    density_cfg,
    wake_cfg: Optional[object] = None,
    impedance: Optional[Union[str, Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Track a particle cloud through 'line' then apply a longitudinal collective kick
    derived from either wake_cfg (fallback) or a provided impedance/wake model.

    Parameters
    ----------
    line : xtrack.Line
        Tracker line to use for external transport.
    z0 : ndarray, shape [Np,6]
        Input cloud.
    mu : ndarray-like
        Parameter vector [Q_bunch, pipe_radius, impedance_scale] (used for scaling).
    density_cfg: object
        An object with attributes nz, zeta_min, zeta_max (same as your DensityGridConfig).
    wake_cfg : optional
        If impedance is None, wake_cfg is used to build a kernel (build_wake_kernel).
    impedance : optional
        If provided, used in preference to wake_cfg. Accepts (zeta_grid,W) tuple, path to .npz/.npy, or W array.
    eps : float
        small numerical constant.

    Returns
    -------
    z1 : ndarray shape [Np,6]
        Tracked cloud after external transport and delta kicks applied.
    """
    Q, a, Z_scale = mu

    p = line.build_particles(
        x=z0[:, 0],
        y=z0[:, 1],
        zeta=z0[:, 2],
        px=z0[:, 3],
        py=z0[:, 4],
        delta=z0[:, 5],
    )
    line.track(p)
    z1 = np.column_stack(
        [
            np.asarray(p.x),
            np.asarray(p.y),
            np.asarray(p.zeta),
            np.asarray(p.px),
            np.asarray(p.py),
            np.asarray(p.delta),
        ]
    ).astype(np.float64)

    if impedance is None and wake_cfg is None:
        return z1

    nz = density_cfg.nz
    zeta_grid, lambda_z = None, None
    zeta_grid, lambda_z = line_density_from_cloud(z1, density_cfg)

    zg_imp, W_imp = load_impedance(impedance, density_cfg)
    if zg_imp is None:
        if wake_cfg is None:
            raise RuntimeError("No wake_cfg provided and impedance not given; cannot compute wake kernel.")
        W = build_wake_kernel(zeta_grid, wake_cfg)
        zg = zeta_grid
    else:
        zg = np.asarray(zg_imp)
        W = np.asarray(W_imp)
        if zg.shape != zeta_grid.shape or not np.allclose(zg, zeta_grid):
            W = np.interp(zeta_grid, zg, W)
            zg = zeta_grid

    W = W / (np.abs(W).sum() + eps)

    wake_conv = np.fft.ifft(np.fft.fft(lambda_z) * np.fft.fft(W)).real

    delta_kick = np.interp(z1[:, 2], zeta_grid, wake_conv)

    delta_kick *= (Q * Z_scale) / (a ** 2 + eps)

    z1[:, 5] += delta_kick

    return z1


def build_line(lattice_cfg: LatticeConfig) -> Tuple[xt.Line, xt.Environment]:
    """Build a simple Xsuite line with three quadrupoles."""
    part_ref = xt.Particles(mass0=lattice_cfg.mass0_ev, p0c=lattice_cfg.p0c_ev)

    env = xt.Environment()
    env.vars.default_to_zero = True
    env["l_mq"] = lattice_cfg.l_mq

    env.new("mq", xt.Quadrupole, length="l_mq")
    env.new("qf1", "mq", k1="kf1")
    env.new("qd1", "mq", k1="kd1")
    env.new("qf2", "mq", k1="kf2")
    env.new("start_cell", xt.Marker)
    env.new("end_cell", xt.Marker)

    line = env.new_line(
        length=lattice_cfg.line_length,
        components=[
            env.place("start_cell", at=0.0),
            env.place("qf1", at=lattice_cfg.qf1_at),
            env.place("qd1", at=lattice_cfg.qd1_at),
            env.place("qf2", at=lattice_cfg.qf2_at),
            env.place("end_cell", at=lattice_cfg.line_length),
        ],
    )
    line.particle_ref = part_ref
    line.build_tracker()
    return line, env

def build_line_from_madx(
    madx_file: str,
    seq_name: str,
    p0c_ev: float = 1e9,
    mass0_ev: float | None = None,
    verbose: bool = False,
) -> "xtrack.Line":
    """
    Robust importer: load MAD-X via cpymad, find sequence (seq_name),
    obtain element order (handles multiple cpymad variants), map common
    MAD-X elements to xtrack elements, build tracker and return xtrack.Line.

    Parameters
    ----------
    madx_file : str
        Path to MAD-X file.
    seq_name : str
        Name of sequence to import.
    p0c_ev : float
        Reference momentum (eV).
    mass0_ev : float | None
        Particle rest energy in eV (if None, fallback to xt.PROTON_MASS_EV).
    verbose : bool
        Print mapping info/warnings.
    """
    from pathlib import Path
    import warnings
    from cpymad.madx import Madx
    import xtrack as xt
    import numpy as np

    p = Path(madx_file)
    if not p.exists():
        raise FileNotFoundError(f"MAD-X file not found: {madx_file!r}")

    mad = Madx()
    mad.call(str(p))

    if seq_name not in mad.sequence:
        seqs = list(mad.sequence.keys())
        lower_map = {s.lower(): s for s in seqs}
        if seq_name.lower() in lower_map:
            seq_name = lower_map[seq_name.lower()]
            if verbose:
                print(f"[build_line_from_madx] using case-insensitive match -> '{seq_name}'")
        else:
            raise RuntimeError(f"Sequence {seq_name!r} not found. Available: {seqs}")

    seq = mad.sequence[seq_name]

    names = []
    try:
        en = getattr(seq, "element_names", None)
        if callable(en):
            names = list(en())
            if verbose:
                print(f"[import] used seq.element_names() -> {len(names)} elements")
    except Exception:
        names = []

    if not names:
        try:
            for elem in seq.elements:
                nm = getattr(elem, "name", None)
                if nm is None:
                    nm = str(elem).split()[0]
                names.append(str(nm))
            if verbose:
                print(f"[import] obtained names by iterating seq.elements -> {len(names)} elements")
        except Exception:
            names = []

    if not names:
        elems_obj = seq.elements
        possible_attrs = ["_list", "_elems", "_dict", "_elements"]
        found = False
        for a in possible_attrs:
            if hasattr(elems_obj, a):
                container = getattr(elems_obj, a)
                try:
                    if isinstance(container, dict):
                        names = list(container.keys())
                    else:
                        for e in container:
                            nm = getattr(e, "name", None) or str(e).split()[0]
                            names.append(str(nm))
                    found = True
                    if verbose:
                        print(f"[import] obtained names from seq.elements.{a} -> {len(names)} elements")
                    break
                except Exception:
                    continue
        if not found:
            try:
                i = 0
                while True:
                    e = seq.elements[i]  # may raise
                    nm = getattr(e, "name", None) or str(e).split()[0]
                    names.append(str(nm))
                    i += 1
            except Exception:
                pass

    if not names:
        raise RuntimeError(
            "Could not determine element order from MAD-X sequence. "
            "Try inspecting mad.sequence keys and seq.elements in an interactive session."
        )

    xt_elements = []
    element_names = []
    for ename in names:
        try:
            elem = seq.elements[ename]
        except Exception:
            found_elem = None
            try:
                for e in seq.elements:
                    nm = getattr(e, "name", None) or str(e).split()[0]
                    if str(nm) == str(ename):
                        found_elem = e
                        break
            except Exception:
                found_elem = None
            if found_elem is None:
                warnings.warn(f"[build_line_from_madx] element {ename!r} not accessible; inserting Drift(0.0)")
                xt_elements.append(xt.Drift(length=0.0))
                element_names.append(str(ename))
                continue
            elem = found_elem

        etype_raw = getattr(elem, "_type", None) or getattr(elem, "type", None)
        etype = str(etype_raw).lower() if etype_raw is not None else ""
        length = float(getattr(elem, "l", 0.0) or 0.0)
        k0 = getattr(elem, "k0", None)
        k1 = getattr(elem, "k1", None)
        k2 = getattr(elem, "k2", None)
        angle = getattr(elem, "angle", None)

        try:
            if "drift" in etype or etype.strip() == "":
                xel = xt.Drift(length=length)

            elif "quad" in etype or "quadrupole" in etype:
                k1f = float(k1) if k1 is not None else 0.0
                xel = xt.Quadrupole(k1=k1f, length=length)

            elif "rbend" in etype or "sbend" in etype or "bend" in etype or "dipole" in etype:
                a = float(angle) if angle is not None else 0.0
                xel = xt.Sbend(angle=a, length=length)

            elif "marker" in etype:
                xel = xt.Marker()

            elif "multipole" in etype or "sext" in etype or "sextupole" in etype:
                knl = {}
                if k0 is not None:
                    knl[0] = float(k0)
                if k1 is not None:
                    knl[1] = float(k1)
                if k2 is not None:
                    knl[2] = float(k2)
                if hasattr(xt, "Multipole"):
                    try:
                        xel = xt.Multipole(knl=knl)
                    except Exception:
                        if verbose:
                            warnings.warn(f"Multipole mapping for '{ename}' failed; using Drift({length})")
                        xel = xt.Drift(length=length)
                else:
                    xel = xt.Drift(length=length)

            elif "rf" in etype or "cavity" in etype:
                if verbose:
                    warnings.warn(f"RF/cavity element '{ename}' encountered — inserting Drift(length={length}) placeholder")
                xel = xt.Drift(length=length)

            else:
                if verbose:
                    warnings.warn(f"Element '{ename}' type '{etype}' not explicitly mapped; using Drift(length={length}) fallback")
                xel = xt.Drift(length=length)

        except Exception as ex:
            warnings.warn(f"Error mapping element '{ename}': {ex}; using Drift(length={length}) fallback")
            xel = xt.Drift(length=length)

        xt_elements.append(xel)
        element_names.append(str(ename))

    xline = xt.Line(elements=xt_elements, element_names=element_names)

    if mass0_ev is None:
        mass0_ev = getattr(xt, "PROTON_MASS_EV", 938272088.0)
    xline.particle_ref = xt.Particles(mass0=float(mass0_ev), p0c=float(p0c_ev))

    xline.build_tracker()

    if verbose:
        print(f"[build_line_from_madx] built xtrack.Line with {len(xt_elements)} elements from seq '{seq_name}'")

    return xline


def _diag_sigmas(cfg: BeamFamilyConfig) -> np.ndarray:
    return np.array(
        [
            cfg.x_sigma,
            cfg.y_sigma,
            cfg.zeta_sigma,
            cfg.px_sigma,
            cfg.py_sigma,
            cfg.delta_sigma,
        ],
        dtype=np.float64,
    )


def sample_initial_conditions(
    n_particles: int,
    rng: np.random.Generator,
    beam_cfg: BeamFamilyConfig,
) -> np.ndarray:
    """Sample a particle cloud in R^6.

    Output order is [x, y, zeta, px, py, delta].
    """
    sigmas = _diag_sigmas(beam_cfg)

    if beam_cfg.family == "gaussian":
        z = rng.normal(0.0, sigmas, size=(n_particles, PHASE_SPACE_DIM))
        return z.astype(np.float64)

    if beam_cfg.family == "correlated_gaussian":
        cov = np.diag(sigmas ** 2)
        # modest correlations in each plane
        corr = beam_cfg.corr_strength
        cov[0, 3] = cov[3, 0] = corr * sigmas[0] * sigmas[3]
        cov[1, 4] = cov[4, 1] = corr * sigmas[1] * sigmas[4]
        cov[2, 5] = cov[5, 2] = corr * sigmas[2] * sigmas[5]
        z = rng.multivariate_normal(np.zeros(PHASE_SPACE_DIM), cov, size=n_particles)
        return z.astype(np.float64)

    if beam_cfg.family == "core_halo":
        n_halo = int(round(beam_cfg.halo_fraction * n_particles))
        n_core = n_particles - n_halo
        core = rng.normal(0.0, sigmas, size=(n_core, PHASE_SPACE_DIM))
        halo = rng.normal(0.0, beam_cfg.halo_scale * sigmas, size=(n_halo, PHASE_SPACE_DIM))
        z = np.vstack([core, halo])
        rng.shuffle(z, axis=0)
        return z.astype(np.float64)

    if beam_cfg.family == "mixture":
        shift = beam_cfg.mixture_shift_scale * sigmas
        n_a = n_particles // 2
        n_b = n_particles - n_a
        a = rng.normal(-shift, sigmas, size=(n_a, PHASE_SPACE_DIM))
        b = rng.normal(+shift, sigmas, size=(n_b, PHASE_SPACE_DIM))
        z = np.vstack([a, b])
        rng.shuffle(z, axis=0)
        return z.astype(np.float64)

    if beam_cfg.family == "mismatched":
        scaled = np.array(sigmas)
        scaled[[0, 1, 3, 4]] *= beam_cfg.mismatch_scale
        z = rng.normal(0.0, scaled, size=(n_particles, PHASE_SPACE_DIM))
        return z.astype(np.float64)

    raise ValueError(f"Unsupported beam family: {beam_cfg.family}")


def particles_to_6d(particles: xt.Particles) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(particles.x),
            np.asarray(particles.y),
            np.asarray(particles.zeta),
            np.asarray(particles.px),
            np.asarray(particles.py),
            np.asarray(particles.delta),
        ]
    ).astype(np.float64)


def sample_parameters(
    rng: np.random.Generator,
    param_ranges: ParameterRanges,
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(*param_ranges.bunch_charge_range),
            rng.uniform(*param_ranges.pipe_radius_range),
            rng.uniform(*param_ranges.impedance_scale_range),
        ],
        dtype=np.float64,
    )

"""def set_lattice_parameters(env: xt.Environment, mu: np.ndarray) -> None:
    env["kf1"] = float(mu[0])
    env["kd1"] = float(mu[1])
    env["kf2"] = float(mu[2])
"""

def track_cloud(line: xt.Line, z0: np.ndarray) -> np.ndarray:
    p = line.build_particles(
        x=z0[:, 0],
        y=z0[:, 1],
        zeta=z0[:, 2],
        px=z0[:, 3],
        py=z0[:, 4],
        delta=z0[:, 5],
    )
    line.track(p)
    return particles_to_6d(p)


def track_with_collective_effects(
    line: xt.Line,
    z0: np.ndarray,
    mu: np.ndarray,
    density_cfg,
    wake_cfg: Optional[object] = None,
    impedance: Optional[Union[str, Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Track a particle cloud through 'line' then apply a longitudinal collective kick
    derived from either wake_cfg (fallback) or a provided impedance/wake model.

    Parameters
    ----------
    line : xtrack.Line
        Tracker line to use for external transport.
    z0 : ndarray, shape [Np,6]
        Input cloud.
    mu : ndarray-like
        Parameter vector [Q_bunch, pipe_radius, impedance_scale] (used for scaling).
    density_cfg: object
        An object with attributes nz, zeta_min, zeta_max (same as your DensityGridConfig).
    wake_cfg : optional
        If impedance is None, wake_cfg is used to build a kernel (build_wake_kernel).
    impedance : optional
        If provided, used in preference to wake_cfg. Accepts (zeta_grid,W) tuple, path to .npz/.npy, or W array.
    eps : float
        small numerical constant.

    Returns
    -------
    z1 : ndarray shape [Np,6]
        Tracked cloud after external transport and delta kicks applied.
    """
    Q, a, Z_scale = mu

    # --- track externally
    p = line.build_particles(
        x=z0[:, 0],
        y=z0[:, 1],
        zeta=z0[:, 2],
        px=z0[:, 3],
        py=z0[:, 4],
        delta=z0[:, 5],
    )
    line.track(p)
    z1 = np.column_stack(
        [
            np.asarray(p.x),
            np.asarray(p.y),
            np.asarray(p.zeta),
            np.asarray(p.px),
            np.asarray(p.py),
            np.asarray(p.delta),
        ]
    ).astype(np.float64)

    if impedance is None and wake_cfg is None:
        return z1

    nz = density_cfg.nz
    zeta_grid, lambda_z = None, None
    zeta_grid, lambda_z = line_density_from_cloud(z1, density_cfg)

    zg_imp, W_imp = load_impedance(impedance, density_cfg)
    if zg_imp is None:
        if wake_cfg is None:
            raise RuntimeError("No wake_cfg provided and impedance not given; cannot compute wake kernel.")
        W = build_wake_kernel(zeta_grid, wake_cfg)
        zg = zeta_grid
    else:
        zg = np.asarray(zg_imp)
        W = np.asarray(W_imp)
        if zg.shape != zeta_grid.shape or not np.allclose(zg, zeta_grid):
            W = np.interp(zeta_grid, zg, W)
            zg = zeta_grid

    W = W / (np.abs(W).sum() + eps)

    wake_conv = np.fft.ifft(np.fft.fft(lambda_z) * np.fft.fft(W)).real

    delta_kick = np.interp(z1[:, 2], zeta_grid, wake_conv)

    delta_kick *= (Q * Z_scale) / (a ** 2 + eps)

    z1[:, 5] += delta_kick

    return z1



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


def build_datasets(
    line: xt.Line,
    env: xt.Environment,
    dataset_cfg: DatasetConfig,
    density_cfg: DensityGridConfig,
    param_ranges: ParameterRanges,
    beam_families: Iterable[BeamFamilyConfig],
    use_collective: bool,
    wake_cfg: Optional[WakeConfig]
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(dataset_cfg.seed)
    families = list(beam_families)
    if not families:
        raise ValueError("At least one beam family must be provided.")

    cloud_in, cloud_out, mu_all = [], [], []
    lambda_in, lambda_out, zeta_grid = [], [], None
    moments_in_centroid, moments_out_centroid = [], []
    moments_in_cov, moments_out_cov = [], []
    family_id = []

    for _ in range(dataset_cfg.n_samples):
        beam_cfg = families[rng.integers(0, len(families))]
        mu = sample_parameters(rng, param_ranges)
        
        z0 = sample_initial_conditions(
            dataset_cfg.particles_per_sample,
            rng,
            beam_cfg
        )

        if use_collective:
            z1 = track_with_collective_effects(
                line,
                z0,
                mu,
                density_cfg,
                wake_cfg,
            )
        else:
            z1 = track_cloud(line, z0)

        
       

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
    return out


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


def save_dataset_bundle(
    data: Dict[str, np.ndarray],
    dataset_cfg: DatasetConfig,
    density_cfg: DensityGridConfig,
    beam_families: Iterable[BeamFamilyConfig],
    param_ranges: ParameterRanges,
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

    time_of_generation = np.datetime64("now").astype(str)
    filename = f"neural_xsuite_dataset_{time_of_generation}.npz"
    np.savez(out_dir / filename, **data, **split)

    metadata = {
        "phase_space_labels": PHASE_SPACE_LABELS,
        "dataset_config": asdict(dataset_cfg),
        "density_grid_config": asdict(density_cfg),
        "parameter_ranges": asdict(param_ranges),
        "beam_families": [asdict(b) for b in beam_families],
        "keys": sorted(list(data.keys()) + list(split.keys())),
    }
    (out_dir / "xsuite_neural_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def default_beam_families() -> List[BeamFamilyConfig]:
    return [
        BeamFamilyConfig(family="gaussian"),
        BeamFamilyConfig(family="correlated_gaussian"),
        BeamFamilyConfig(family="core_halo"),
        BeamFamilyConfig(family="mixture"),
        BeamFamilyConfig(family="mismatched"),
    ]



def madx_list_includes(madx_path: str):
    """
    Return list of filenames referenced by `call,file=...` or `include,file=...`
    in the MAD-X file. Filenames are returned as they appear (may be relative).
    """
    p = Path(madx_path)
    text = p.read_text()
    return INCLUDE_RE.findall(text)

from contextlib import contextmanager

@contextmanager
def pushd(path: Path):
    cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)

def build_line_from_madx_safe(madx_file: str, seq_name: str, verbose: bool = False):
    """
    Wrapper that checks includes and runs MAD-X with working dir set to the madx file parent.
    Returns the cpymad.Madx instance (or raises RuntimeError with a helpful message).
    """
    p = Path(madx_file)
    if not p.exists():
        raise FileNotFoundError(f"MAD-X file not found: {madx_file}")

    # list referenced include files and check for existence
    includes = madx_list_includes(str(p))
    missing = []
    for inc in includes:
        # treat relative path relative to MAD-X file directory
        inc_path = (p.parent / inc).resolve()
        if not inc_path.exists():
            missing.append(str(inc_path))

    if missing:
        raise FileNotFoundError(
            "MAD-X file references included files that do not exist when resolved relative to the MAD-X file directory:\n"
            + "\n".join(missing)
            + "\nEither provide those files or adjust the includes to absolute paths."
        )

    mad = Madx()
    # run MAD-X with cwd set to madx parent so relative includes and generated .str files resolve
    try:
        with pushd(p.parent):
            if verbose:
                print(f"[build_line_from_madx_safe] running MAD-X in directory: {p.parent}")
            mad.call(str(p.name))   # call by filename (cwd is parent)
    except Exception as e:
        # Provide additional diagnostics: print includes and current cwd
        raise RuntimeError(
            f"MAD-X failed when loading {madx_file!r} from cwd={Path.cwd()!s}. "
            f"Referenced includes: {includes!r}. "
            f"Original error: {e}"
        ) from e

    # If you want to return the mad instance for further inspection:
    return mad



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Xsuite datasets for neural operator learning.")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--particles-per-sample", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./data/neural")
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--zeta-min", type=float, default=-5e-3)
    parser.add_argument("--zeta-max", type=float, default=5e-3)
    parser.add_argument("--Q-min", type=float, default=0.1e-9)
    parser.add_argument("--Q-max", type=float, default=3e-9)
    parser.add_argument("--pipe-radius-min", type=float, default=5e-3)
    parser.add_argument("--pipe-radius-max", type=float, default=30e-3)
    parser.add_argument("--impedance-min", type=float, default=0.1)
    parser.add_argument("--impedance-max", type=float, default=10.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--skip-cloud", action="store_true")
    parser.add_argument("--skip-density", action="store_true")
    parser.add_argument("--no-collective", action="store_true")

    # wake
    parser.add_argument("--wake-type", type=str, default="gaussian")
    parser.add_argument("--wake-strength", type=float, default=1.0)
    parser.add_argument("--wake-sigma", type=float, default=1e-3)
    parser.add_argument("--wake-decay", type=float, default=1e3)
    parser.add_argument("--wake-freq", type=float, default=1e10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lattice_cfg = LatticeConfig()
    dataset_cfg = DatasetConfig(
        n_samples=args.n_samples,
        particles_per_sample=args.particles_per_sample,
        seed=args.seed,
        output_dir=args.output_dir,
        save_cloud_dataset=not args.skip_cloud,
        save_density_dataset=not args.skip_density,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    density_cfg = DensityGridConfig(
        nz=args.nz,
        zeta_min=args.zeta_min,
        zeta_max=args.zeta_max,
    )
    wake_cfg = WakeConfig(
        kind="exponential",     # "gaussian" | "exponential" | "resonator"
        strength=1.0,
        sigma=1e-3,
    )
    param_ranges = ParameterRanges(
        bunch_charge_range=(args.Q_min, args.Q_max),
        pipe_radius_range=(args.pipe_radius_min, args.pipe_radius_max),
        impedance_scale_range=(args.impedance_min, args.impedance_max),
    )
    beam_families = default_beam_families()

    
    madx_path = "/home/martinez/ML4CollEffects/notebooks/ext_HEB/optics/v24_1/heb_ring_z.madx"
    seq_name = "fcc_heb"
    line = build_line_from_madx_safe(madx_path,
                             seq_name=seq_name,
                             verbose=True)

    data = build_datasets(
        line=line,
        env=env,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        param_ranges=param_ranges,
        beam_families=beam_families,
        use_collective= True,
        wake_cfg= wake_cfg
    )
    save_dataset_bundle(
        data=data,
        dataset_cfg=dataset_cfg,
        density_cfg=density_cfg,
        beam_families=beam_families,
        param_ranges=param_ranges,
    )

    print("Saved dataset bundle to", dataset_cfg.output_dir)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
