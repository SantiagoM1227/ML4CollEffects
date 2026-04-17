import numpy as np
import torch

from haissinski_wake_ml import (
    HaissinskiWakeDataset,
    ToeplitzWakeConv1D,
    make_train_val_loaders,
    train_wake_model,
    plot_training_history,
    plot_learned_kernel,
)

# Replace this with your own file.
# Expected keys:
#   q         : (N,)
#   currents  : (M,)
#   lambdas   : (M, N)
# optional:
#   F_targets : (M, N)

data = np.load("your_haissinski_data.npz")
q = data["q"]
currents = data["currents"]
lambdas = data["lambdas"]
F_targets = data["F_targets"] if "F_targets" in data else None

dataset = HaissinskiWakeDataset(
    lambdas=lambdas,
    currents=currents,
    q=q,
    F_targets=F_targets,
    lambda_floor=1e-12,
)

train_loader, val_loader = make_train_val_loaders(
    dataset,
    batch_size=16,
    val_fraction=0.2,
    seed=42,
)

model = ToeplitzWakeConv1D(
    n_grid=len(q),
    dq=dataset.dq,
    kernel_size=2 * len(q) - 1,  # full Toeplitz support
    init_scale=1e-4,
)

history = train_wake_model(
    model,
    train_loader,
    val_loader=val_loader,
    epochs=400,
    lr=1e-3,
    alpha_smooth=1e-4,
    beta_l2=1e-8,
    patience=60,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

plot_training_history(history)
plot_learned_kernel(model)

# Save learned wake
np.savez(
    "learned_wake.npz",
    lag=model.lag_grid_numpy(),
    W=model.kernel_numpy(),
    q=q,
)
print("Saved learned_wake.npz")
