from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


ArrayLike = np.ndarray | torch.Tensor


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _to_numpy(x: ArrayLike, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return x


def _check_uniform_grid(q: np.ndarray, atol: float = 1e-12, rtol: float = 1e-9) -> float:
    if q.ndim != 1:
        raise ValueError("q must be a 1D array.")
    if q.size < 3:
        raise ValueError("q must contain at least 3 points.")
    dq = np.diff(q)
    if not np.allclose(dq, dq[0], atol=atol, rtol=rtol):
        raise ValueError("q must be uniformly spaced for the convolution model used here.")
    return float(dq[0])


def normalize_lambdas(lambdas: ArrayLike, q: ArrayLike, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize each profile so that integral lambda(q) dq = 1.
    """
    lambdas = _to_numpy(lambdas, "lambdas")
    q = _to_numpy(q, "q")
    if lambdas.ndim == 1:
        lambdas = lambdas[None, :]
    if lambdas.shape[1] != q.size:
        raise ValueError("Each lambda profile must have the same length as q.")

    areas = np.trapz(lambdas, q, axis=1)
    bad = np.abs(areas) < eps
    if np.any(bad):
        raise ValueError("At least one profile has near-zero integral and cannot be normalized.")
    return lambdas / areas[:, None]


def compute_F_from_lambda(
    lambdas: ArrayLike,
    q: ArrayLike,
    lambda_floor: float = 1e-14,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Compute
        F(q) = d/dq log(lambda(q)) + q
    with a stable log-floor.

    Parameters
    ----------
    lambdas:
        Array of shape (n_samples, n_grid) or (n_grid,).
    q:
        Uniform grid of shape (n_grid,).
    lambda_floor:
        Positive floor used before taking the logarithm.
    renormalize:
        Whether to normalize each lambda profile to unit integral first.
    """
    lambdas = _to_numpy(lambdas, "lambdas")
    q = _to_numpy(q, "q")
    _check_uniform_grid(q)

    if lambdas.ndim == 1:
        lambdas = lambdas[None, :]
    if lambdas.shape[1] != q.size:
        raise ValueError("Each lambda profile must have the same length as q.")
    if lambda_floor <= 0.0:
        raise ValueError("lambda_floor must be positive.")

    if renormalize:
        lambdas = normalize_lambdas(lambdas, q)

    lam_safe = np.clip(lambdas, lambda_floor, None)
    loglam = np.log(lam_safe)
    dlog = np.gradient(loglam, q, axis=1, edge_order=2)
    return dlog + q[None, :]


def make_train_val_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must lie strictly between 0 and 1.")

    n_total = len(dataset)
    n_val = max(1, int(round(val_fraction * n_total)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("Validation split leaves no training samples.")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Dataset wrapper
# -----------------------------------------------------------------------------

class HaissinskiWakeDataset(Dataset):
    """
    Minimal wrapper for multi-current Haissinski data.

    Each sample contains:
      - lambda(q)      : profile on a fixed uniform grid
      - current I      : scalar current / normalized current
      - F_target(q)    : target reconstructed from lambda or supplied directly

    The intended forward model is
        F(q) \approx I * (W * lambda)(q)
    after discretization.
    """

    def __init__(
        self,
        lambdas: ArrayLike,
        currents: ArrayLike,
        q: ArrayLike,
        F_targets: Optional[ArrayLike] = None,
        lambda_floor: float = 1e-14,
        renormalize_lambdas: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        q_np = _to_numpy(q, "q")
        dq = _check_uniform_grid(q_np)

        lambdas_np = _to_numpy(lambdas, "lambdas")
        if lambdas_np.ndim == 1:
            lambdas_np = lambdas_np[None, :]
        if lambdas_np.shape[1] != q_np.size:
            raise ValueError("lambdas must have shape (n_samples, n_grid).")

        currents_np = _to_numpy(currents, "currents").reshape(-1)
        if currents_np.size != lambdas_np.shape[0]:
            raise ValueError("currents must have length n_samples.")

        if renormalize_lambdas:
            lambdas_np = normalize_lambdas(lambdas_np, q_np)

        if F_targets is None:
            F_np = compute_F_from_lambda(
                lambdas_np,
                q_np,
                lambda_floor=lambda_floor,
                renormalize=False,
            )
        else:
            F_np = _to_numpy(F_targets, "F_targets")
            if F_np.ndim == 1:
                F_np = F_np[None, :]
            if F_np.shape != lambdas_np.shape:
                raise ValueError("F_targets must have the same shape as lambdas.")

        self.q = torch.as_tensor(q_np, dtype=dtype)
        self.dq = float(dq)
        self.lambdas = torch.as_tensor(lambdas_np, dtype=dtype)
        self.currents = torch.as_tensor(currents_np, dtype=dtype)
        self.F_targets = torch.as_tensor(F_np, dtype=dtype)

    def __len__(self) -> int:
        return self.lambdas.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "lambda": self.lambdas[idx],
            "current": self.currents[idx],
            "F_target": self.F_targets[idx],
        }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class ToeplitzWakeConv1D(nn.Module):
    """
    Trainable wake kernel W(lag) on a uniform grid.

    The model computes
        F_hat_i = I * dq * sum_j W_{i-j} * lambda_j
    using a 1D convolution with zero padding.

    Parameters
    ----------
    n_grid:
        Number of q-grid points.
    dq:
        Grid spacing.
    kernel_size:
        Number of trainable lag coefficients.
        Must be odd and at most 2*n_grid - 1.
        If None, uses the full Toeplitz support 2*n_grid - 1.
    init_scale:
        Standard deviation for kernel initialization.
    learn_bias:
        Optional additive bias in F; normally False.
    """

    def __init__(
        self,
        n_grid: int,
        dq: float,
        kernel_size: Optional[int] = None,
        init_scale: float = 1e-4,
        learn_bias: bool = False,
    ) -> None:
        super().__init__()

        if n_grid < 3:
            raise ValueError("n_grid must be at least 3.")
        if dq <= 0.0:
            raise ValueError("dq must be positive.")

        full_size = 2 * n_grid - 1
        if kernel_size is None:
            kernel_size = full_size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if kernel_size < 1 or kernel_size > full_size:
            raise ValueError("kernel_size must satisfy 1 <= kernel_size <= 2*n_grid - 1.")

        self.n_grid = int(n_grid)
        self.dq = float(dq)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2

        kernel = init_scale * torch.randn(self.kernel_size, dtype=torch.float32)
        self.kernel = nn.Parameter(kernel)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32)) if learn_bias else None

    @property
    def lag_grid(self) -> torch.Tensor:
        radius = self.kernel_size // 2
        return self.dq * torch.arange(-radius, radius + 1, dtype=self.kernel.dtype, device=self.kernel.device)

    def forward(self, lambdas: torch.Tensor, currents: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lambdas : (batch, n_grid)
        currents: (batch,) or (batch, 1)
        """
        if lambdas.ndim != 2:
            raise ValueError("lambdas must have shape (batch, n_grid).")
        if lambdas.shape[1] != self.n_grid:
            raise ValueError(f"Expected n_grid={self.n_grid}, got {lambdas.shape[1]}.")

        if currents.ndim == 2 and currents.shape[1] == 1:
            currents = currents[:, 0]
        if currents.ndim != 1 or currents.shape[0] != lambdas.shape[0]:
            raise ValueError("currents must have shape (batch,) or (batch, 1).")

        x = lambdas.unsqueeze(1)  # (B,1,N)
        weight = self.kernel.flip(0).view(1, 1, self.kernel_size)
        conv = F.conv1d(x, weight, padding=self.padding).squeeze(1)  # (B,N)
        out = currents[:, None] * self.dq * conv
        if self.bias is not None:
            out = out + self.bias
        return out

    def kernel_numpy(self) -> np.ndarray:
        return self.kernel.detach().cpu().numpy().copy()

    def lag_grid_numpy(self) -> np.ndarray:
        return self.lag_grid.detach().cpu().numpy().copy()


# -----------------------------------------------------------------------------
# Regularization and training
# -----------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    train_loss: list
    train_data_loss: list
    val_loss: list
    val_data_loss: list



def finite_difference_penalty(x: torch.Tensor, order: int = 2) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("finite_difference_penalty expects a 1D tensor.")
    if order < 1:
        raise ValueError("order must be >= 1.")

    y = x
    for _ in range(order):
        if y.numel() < 2:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        y = y[1:] - y[:-1]
    return torch.mean(y.pow(2))



def kernel_l2_penalty(kernel: torch.Tensor) -> torch.Tensor:
    return torch.mean(kernel.pow(2))



def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}



def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    alpha_smooth: float = 0.0,
    beta_l2: float = 0.0,
    smoothness_order: int = 2,
) -> Dict[str, float]:
    device = torch.device(device)
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_data_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model(batch["lambda"], batch["current"])
            data_loss = mse(pred, batch["F_target"])
            reg = alpha_smooth * finite_difference_penalty(model.kernel, order=smoothness_order)
            reg = reg + beta_l2 * kernel_l2_penalty(model.kernel)
            loss = data_loss + reg

            bs = batch["lambda"].shape[0]
            total_loss += float(loss.item()) * bs
            total_data_loss += float(data_loss.item()) * bs
            total_items += bs

    return {
        "loss": total_loss / max(total_items, 1),
        "data_loss": total_data_loss / max(total_items, 1),
    }



def train_wake_model(
    model: ToeplitzWakeConv1D,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    alpha_smooth: float = 1e-4,
    beta_l2: float = 1e-8,
    smoothness_order: int = 2,
    device: str | torch.device = "cpu",
    clip_grad_norm: Optional[float] = None,
    patience: Optional[int] = 40,
    verbose: bool = True,
) -> TrainingHistory:
    device = torch.device(device)
    model = model.to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = TrainingHistory(train_loss=[], train_data_loss=[], val_loss=[], val_data_loss=[])
    best_state = None
    best_val = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_items = 0

        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            pred = model(batch["lambda"], batch["current"])
            data_loss = mse(pred, batch["F_target"])
            reg = alpha_smooth * finite_difference_penalty(model.kernel, order=smoothness_order)
            reg = reg + beta_l2 * kernel_l2_penalty(model.kernel)
            loss = data_loss + reg
            loss.backward()

            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            bs = batch["lambda"].shape[0]
            total_loss += float(loss.item()) * bs
            total_data_loss += float(data_loss.item()) * bs
            total_items += bs

        train_loss = total_loss / max(total_items, 1)
        train_data_loss = total_data_loss / max(total_items, 1)
        history.train_loss.append(train_loss)
        history.train_data_loss.append(train_data_loss)

        if val_loader is not None:
            metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                alpha_smooth=alpha_smooth,
                beta_l2=beta_l2,
                smoothness_order=smoothness_order,
            )
            val_loss = metrics["loss"]
            val_data_loss = metrics["data_loss"]
        else:
            val_loss = math.nan
            val_data_loss = math.nan

        history.val_loss.append(val_loss)
        history.val_data_loss.append(val_data_loss)

        if verbose:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train={train_loss:.6e} (data={train_data_loss:.6e}) | "
                f"val={val_loss:.6e} (data={val_data_loss:.6e})"
            )

        if val_loader is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# -----------------------------------------------------------------------------
# Synthetic data generator for quick tests
# -----------------------------------------------------------------------------


def generate_synthetic_dataset(
    n_samples: int = 128,
    n_grid: int = 257,
    q_min: float = -8.0,
    q_max: float = 8.0,
    current_range: Tuple[float, float] = (0.2, 1.0),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate a toy dataset with Gaussian-like lambda profiles and a known wake.

    This is only for debugging the pipeline before inserting real data.
    """
    rng = np.random.default_rng(seed)
    q = np.linspace(q_min, q_max, n_grid)
    dq = q[1] - q[0]

    lag = np.arange(-(n_grid - 1), n_grid) * dq
    # Example wake with oscillatory-decaying structure.
    W_true = 0.5 * np.exp(-0.5 * (lag / 0.8) ** 2) - 0.35 * np.exp(-0.5 * ((lag - 0.6) / 0.25) ** 2)
    W_true += 0.12 * np.sin(5.0 * lag) * np.exp(-0.5 * (lag / 1.3) ** 2)

    currents = rng.uniform(current_range[0], current_range[1], size=n_samples)
    lambdas = np.zeros((n_samples, n_grid), dtype=np.float64)
    F_targets = np.zeros_like(lambdas)

    kernel_torch = torch.tensor(W_true, dtype=torch.float32).flip(0).view(1, 1, -1)

    for i in range(n_samples):
        mean = rng.normal(0.0, 0.9)
        sigma = rng.uniform(0.7, 1.4)
        skew_bump = rng.uniform(-0.25, 0.25)

        lam = np.exp(-0.5 * ((q - mean) / sigma) ** 2)
        lam += 0.22 * np.exp(-0.5 * ((q - (mean + 1.1 * sigma)) / (0.55 * sigma)) ** 2)
        lam += skew_bump * np.exp(-0.5 * ((q - (mean - 1.4 * sigma)) / (0.4 * sigma)) ** 2)
        lam = np.clip(lam, 1e-10, None)
        lam = lam / np.trapz(lam, q)
        lambdas[i] = lam

        lam_t = torch.tensor(lam, dtype=torch.float32).view(1, 1, -1)
        conv = F.conv1d(lam_t, kernel_torch, padding=n_grid - 1).view(-1).numpy()
        F_targets[i] = currents[i] * dq * conv

    return {
        "q": q,
        "currents": currents,
        "lambdas": lambdas,
        "F_targets": F_targets,
        "W_true": W_true,
        "lag": lag,
    }


# -----------------------------------------------------------------------------
# Simple plotting helpers
# -----------------------------------------------------------------------------


def plot_training_history(history: TrainingHistory):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history.train_data_loss, label="train data loss")
    if not all(math.isnan(v) for v in history.val_data_loss):
        plt.plot(history.val_data_loss, label="val data loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_learned_kernel(model: ToeplitzWakeConv1D, true_lag: Optional[np.ndarray] = None, true_W: Optional[np.ndarray] = None):
    import matplotlib.pyplot as plt

    lag = model.lag_grid_numpy()
    W = model.kernel_numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(lag, W, label="learned W")
    if true_lag is not None and true_W is not None:
        plt.plot(true_lag, true_W, label="true W", linestyle="--")
    plt.xlabel("lag")
    plt.ylabel("W(lag)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Create a toy dataset
    data = generate_synthetic_dataset(n_samples=256, n_grid=257, seed=7)

    # 2) Wrap it
    dataset = HaissinskiWakeDataset(
        lambdas=data["lambdas"],
        currents=data["currents"],
        q=data["q"],
        F_targets=data["F_targets"],
    )

    # 3) Split into train/validation loaders
    train_loader, val_loader = make_train_val_loaders(dataset, batch_size=32, val_fraction=0.2, seed=7)

    # 4) Build the wake model
    model = ToeplitzWakeConv1D(
        n_grid=len(data["q"]),
        dq=dataset.dq,
        kernel_size=2 * len(data["q"]) - 1,
        init_scale=1e-3,
    )

    # 5) Train
    history = train_wake_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=300,
        lr=2e-3,
        alpha_smooth=1e-4,
        beta_l2=1e-8,
        patience=50,
        device="cpu",
        verbose=True,
    )

    # 6) Plot results
    plot_training_history(history)
    plot_learned_kernel(model, true_lag=data["lag"], true_W=data["W_true"])
