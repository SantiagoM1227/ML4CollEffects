from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class DatasetBuildConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    shuffle: bool = True
    seed: int = 42
    transpose_if_needed: bool = True


class ShapeError(ValueError):
    pass


def _as_numpy(x: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _maybe_transpose(name: str, arr: np.ndarray, n_grid: int, allow: bool) -> np.ndarray:
    """
    Accept either [Ns, Nq] or [Nq, Ns].
    Prefer [Ns, Nq]. If the last dimension is not n_grid but the first is, transpose.
    """
    if arr.ndim != 2:
        raise ShapeError(f"{name} must be 2D, got shape {arr.shape}")

    if arr.shape[-1] == n_grid:
        return arr

    if allow and arr.shape[0] == n_grid:
        return arr.T

    raise ShapeError(
        f"{name} must have shape [Ns, Nq] with Nq={n_grid} or [Nq, Ns]. Got {arr.shape}."
    )


def _prepare_vector(name: str, arr: Any, n_samples: int) -> np.ndarray:
    vec = _as_numpy(arr, dtype=np.float32)
    if vec.ndim == 2 and vec.shape[1] == 1:
        vec = vec[:, 0]
    if vec.ndim != 1:
        raise ShapeError(f"{name} must be 1D or [Ns,1], got shape {vec.shape}")
    if vec.shape[0] != n_samples:
        raise ShapeError(f"{name} must have length {n_samples}, got {vec.shape[0]}")
    return vec


def _prepare_optional_matrix(name: str, arr: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    mat = _as_numpy(arr, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[:, None]
    if mat.ndim != 2:
        raise ShapeError(f"{name} must be 2D or 1D, got shape {mat.shape}")
    if mat.shape[0] != n_samples:
        raise ShapeError(f"{name} must have first dimension {n_samples}, got {mat.shape[0]}")
    return mat


def make_split_indices(
    n_samples: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    idx = np.arange(n_samples, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_train = int(round(train_fraction * n_samples))
    n_val = int(round(val_fraction * n_samples))

    # keep all samples accounted for
    n_train = min(n_train, n_samples)
    n_val = min(n_val, n_samples - n_train)
    n_test_start = n_train + n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_test_start]
    test_idx = idx[n_test_start:]
    return train_idx, val_idx, test_idx


def build_haissinski_forward_dataset(
    q_grid: Any,
    W: Any,
    I: Any,
    lambda_target: Any,
    machine_params: Optional[Any] = None,
    *,
    config: Optional[DatasetBuildConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Build the exact NPZ-style dictionary expected by train_haissinski_fno.py.

    Required arrays
    ---------------
    q_grid:         [Nq]
    W:              [Ns, Nq] or [Nq, Ns]
    I:              [Ns] or [Ns, 1]
    lambda_target:  [Ns, Nq] or [Nq, Ns]

    Optional
    --------
    machine_params: [Ns, P] or [Ns]

    Returns a dictionary containing:
      q_grid, W, I, lambda_target, train_idx, val_idx, test_idx
    and machine_params if provided.
    """
    cfg = config or DatasetBuildConfig()

    q_grid = _as_numpy(q_grid, dtype=np.float32)
    if q_grid.ndim != 1:
        raise ShapeError(f"q_grid must be 1D, got shape {q_grid.shape}")
    if q_grid.size < 2:
        raise ShapeError("q_grid must have at least 2 points")

    W = _maybe_transpose("W", _as_numpy(W, dtype=np.float32), q_grid.size, cfg.transpose_if_needed)
    lambda_target = _maybe_transpose(
        "lambda_target",
        _as_numpy(lambda_target, dtype=np.float32),
        q_grid.size,
        cfg.transpose_if_needed,
    )

    n_samples = W.shape[0]
    if lambda_target.shape[0] != n_samples:
        raise ShapeError(
            f"W and lambda_target must have the same number of samples. Got {n_samples} and {lambda_target.shape[0]}"
        )

    I = _prepare_vector("I", I, n_samples)
    machine_params = _prepare_optional_matrix("machine_params", machine_params, n_samples)

    train_idx, val_idx, test_idx = make_split_indices(
        n_samples=n_samples,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )

    data: Dict[str, np.ndarray] = {
        "q_grid": q_grid,
        "W": W,
        "I": I,
        "lambda_target": lambda_target,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    if machine_params is not None:
        data["machine_params"] = machine_params
    return data


def save_haissinski_forward_dataset(
    out_path: str | Path,
    q_grid: Any,
    W: Any,
    I: Any,
    lambda_target: Any,
    machine_params: Optional[Any] = None,
    *,
    config: Optional[DatasetBuildConfig] = None,
) -> Dict[str, np.ndarray]:
    data = build_haissinski_forward_dataset(
        q_grid=q_grid,
        W=W,
        I=I,
        lambda_target=lambda_target,
        machine_params=machine_params,
        config=config,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)
    return data


def load_and_validate_npz(path: str | Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    required = ["q_grid", "W", "I", "lambda_target"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ShapeError(f"Missing keys in dataset: {missing}")

    # Re-run through the validator/builder but preserve explicit indices if they exist.
    rebuilt = build_haissinski_forward_dataset(
        q_grid=data["q_grid"],
        W=data["W"],
        I=data["I"],
        lambda_target=data["lambda_target"],
        machine_params=data.get("machine_params"),
        config=DatasetBuildConfig(shuffle=False),
    )

    for key in ("train_idx", "val_idx", "test_idx"):
        if key in data:
            rebuilt[key] = np.asarray(data[key], dtype=np.int64)
    return rebuilt


if __name__ == "__main__":
    # Minimal self-demo
    Ns, Nq, P = 128, 256, 2
    q = np.linspace(-10.0, 10.0, Nq, dtype=np.float32)

    rng = np.random.default_rng(42)
    W = rng.normal(size=(Ns, Nq)).astype(np.float32)
    I = rng.uniform(0.1, 2.0, size=(Ns,)).astype(np.float32)
    machine_params = rng.normal(size=(Ns, P)).astype(np.float32)

    lam_raw = np.exp(-0.5 * ((q[None, :] - 0.3 * I[:, None]) / 1.2) ** 2).astype(np.float32)
    lam_raw /= np.trapz(lam_raw, q, axis=1, keepdims=False)[:, None]

    out = Path("./demo_haissinski_forward_dataset.npz")
    save_haissinski_forward_dataset(
        out,
        q_grid=q,
        W=W,
        I=I,
        lambda_target=lam_raw,
        machine_params=machine_params,
    )
    print(f"Saved demo dataset to {out.resolve()}")
