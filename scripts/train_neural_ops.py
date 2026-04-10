
#   {rho_in(q,p); mu} -> rho_out(q,p)

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split



def predict_density(model, grid, rho_in, mu, device="cpu"):
    model.eval()
    coords = grid.coord_channels().to(device)

    with torch.no_grad():
        pred = model(
            rho_in.unsqueeze(0).to(device),   # [1, 1, Nq, Np]
            mu.unsqueeze(0).to(device),       # [1, 3]
            coords,
        )
        pred = normalize_density(pred, grid.dq, grid.dp)

    return pred[0, 0].cpu()   # [Nq, Np]


def predict_from_dataset(model, dataset, grid, idx, device="cpu"):
    rho_in, mu, rho_out = dataset[idx]
    pred = predict_density(model, grid, rho_in, mu, device=device)
    return rho_in[0].cpu(), pred, rho_out[0].cpu(), mu.cpu()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_density(rho: torch.Tensor, dq: float, dp: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a density so that integral rho dq dp = 1.

    Supports shapes:
      [Nx, Np]
      [1, Nx, Np]
      [B, 1, Nx, Np]
    """
    if rho.ndim == 2:
        mass = rho.sum() * dq * dp
        return rho / (mass + eps)
    if rho.ndim == 3:
        mass = rho.sum(dim=(-2, -1), keepdim=True) * dq * dp
        return rho / (mass + eps)
    if rho.ndim == 4:
        mass = rho.sum(dim=(-2, -1), keepdim=True) * dq * dp
        return rho / (mass + eps)
    raise ValueError(f"Unsupported rho shape: {rho.shape}")


def safe_kl(p: torch.Tensor, q: torch.Tensor, dq: float, dp: float, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes KL(p || q) for normalized positive densities.
    Shapes: [B, 1, Nx, Np]
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    val = (p * (torch.log(p) - torch.log(q))).sum(dim=(-2, -1)) * dq * dp
    return val.mean()


########## GRID ##########

@dataclass
class PhaseSpaceGrid:
    nq: int = 64
    np_: int = 64
    qlim: float = 4.0
    plim: float = 4.0

    def __post_init__(self) -> None:
        q = torch.linspace(-self.qlim, self.qlim, self.nq)
        p = torch.linspace(-self.plim, self.plim, self.np_)
        Q, P = torch.meshgrid(q, p, indexing="ij")

        self.q = q
        self.p = p
        self.Q = Q
        self.P = P
        self.dq = float(q[1] - q[0])
        self.dp = float(p[1] - p[0])

    def coord_channels(self) -> torch.Tensor:
        """
        Returns [1, 2, Nq, Np] with channels [q, p]
        """
        return torch.stack([self.Q, self.P], dim=0).unsqueeze(0).float()


########## RANDOM DENSITIES ##########

def gaussian_component(q: torch.Tensor, p: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    mu_q = params["mu_q"]
    mu_p = params["mu_p"]
    sig_q = params["sig_q"]
    sig_p = params["sig_p"]
    corr = params["corr"]

    x = (q - mu_q) / sig_q
    y = (p - mu_p) / sig_p
    denom = 2.0 * (1.0 - corr**2 + 1e-12)
    z = (x**2 - 2.0 * corr * x * y + y**2) / denom
    return torch.exp(-z)


def ring_component(q: torch.Tensor, p: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    c_q = params["c_q"]
    c_p = params["c_p"]
    r0 = params["r0"]
    sig_r = params["sig_r"]

    r = torch.sqrt((q - c_q) ** 2 + (p - c_p) ** 2 + 1e-12)
    return torch.exp(-0.5 * ((r - r0) / sig_r) ** 2)


def banana_component(q: torch.Tensor, p: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    c_q = params["c_q"]
    c_p = params["c_p"]
    sig_q = params["sig_q"]
    sig_p = params["sig_p"]
    a = params["a"]
    b = params["b"]

    q_shift = q - c_q
    ridge = c_p + a * q_shift + b * q_shift**2
    return torch.exp(-0.5 * (q_shift / sig_q) ** 2 - 0.5 * ((p - ridge) / sig_p) ** 2)


def sample_density_spec(rng: random.Random) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Returns a list of components:
      (kind, weight, params)
    """
    n_comp = rng.randint(1, 4)
    spec = []

    for _ in range(n_comp):
        kind = rng.choice(["gaussian", "ring", "banana"])
        weight = rng.uniform(0.4, 1.5)

        if kind == "gaussian":
            params = {
                "mu_q": rng.uniform(-2.0, 2.0),
                "mu_p": rng.uniform(-2.0, 2.0),
                "sig_q": rng.uniform(0.15, 0.8),
                "sig_p": rng.uniform(0.15, 0.8),
                "corr": rng.uniform(-0.7, 0.7),
            }
        elif kind == "ring":
            params = {
                "c_q": rng.uniform(-1.2, 1.2),
                "c_p": rng.uniform(-1.2, 1.2),
                "r0": rng.uniform(0.4, 1.8),
                "sig_r": rng.uniform(0.08, 0.30),
            }
        else:  # banana
            params = {
                "c_q": rng.uniform(-1.5, 1.5),
                "c_p": rng.uniform(-1.2, 1.2),
                "sig_q": rng.uniform(0.25, 0.9),
                "sig_p": rng.uniform(0.08, 0.25),
                "a": rng.uniform(-0.8, 0.8),
                "b": rng.uniform(-0.6, 0.6),
            }

        spec.append((kind, weight, params))

    return spec


def evaluate_density(q: torch.Tensor, p: torch.Tensor, spec: List[Tuple[str, float, Dict[str, float]]]) -> torch.Tensor:
    rho = torch.zeros_like(q)

    for kind, weight, params in spec:
        if kind == "gaussian":
            rho = rho + weight * gaussian_component(q, p, params)
        elif kind == "ring":
            rho = rho + weight * ring_component(q, p, params)
        elif kind == "banana":
            rho = rho + weight * banana_component(q, p, params)
        else:
            raise ValueError(f"Unknown component kind: {kind}")

    return rho.clamp_min(0.0)


########### PARAMETRIC SYMPLECTIC TRANSPORT ###########

def sample_mu(rng: random.Random) -> torch.Tensor:
    """
    mu has dimension 6:
      layer 1: [k1_1, k3_1, s1_1]
      layer 2: [k1_2, k3_2, s1_2]
    """
    vals = [
        rng.uniform(-0.35, 0.35),   # k1_1
        rng.uniform(-0.05, 0.05),   # k3_1
        rng.uniform(0.65, 1.25),    # s1_1
        rng.uniform(-0.35, 0.35),   # k1_2
        rng.uniform(-0.05, 0.05),   # k3_2
        rng.uniform(0.65, 1.25),    # s1_2
    ]
    return torch.tensor(vals, dtype=torch.float32)


def _kick(q: torch.Tensor, k1: float, k3: float) -> torch.Tensor:
    return k1 * q + k3 * q**3


def _drift(p: torch.Tensor, s1: float) -> torch.Tensor:
    return s1 * p + 0.05 * p**3


def inverse_transport(q_out: torch.Tensor, p_out: torch.Tensor, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of a 2-layer explicit symplectic kick-drift map.

    Forward layer:
      p <- p - dV/dq(q)
      q <- q + dT/dp(p)

    Inverse layer:
      q <- q - dT/dp(p)
      p <- p + dV/dq(q)

    Since the full map is symplectic, |det DF| = 1, so
      rho_out(z') = rho_in(F^{-1}(z'))
    """
    q = q_out.clone()
    p = p_out.clone()

    # Reverse order: layer 2 then layer 1
    for layer in [1, 0]:
        k1 = float(mu[3 * layer + 0])
        k3 = float(mu[3 * layer + 1])
        s1 = float(mu[3 * layer + 2])

        q = q - _drift(p, s1)
        p = p + _kick(q, k1, k3)

    return q, p


######## DATASET #########



class XsuiteDensityDataset(Dataset):
    def __init__(self, path: str, grid: PhaseSpaceGrid, plane=(0, 3)):
        data = np.load(path)
        self.X = data["X"]
        self.Y = data["Y"]
        self.MU = data["MU"]
        self.grid = grid
        self.iq, self.ip = plane

        self.q_edges = np.linspace(-grid.qlim, grid.qlim, grid.nq + 1)
        self.p_edges = np.linspace(-grid.plim, grid.plim, grid.np_ + 1)
        self.dq = grid.dq
        self.dp = grid.dp

    def cloud_to_density(self, Z):
        q = Z[:, self.iq]
        p = Z[:, self.ip]

        H, _, _ = np.histogram2d(
            q, p,
            bins=[self.q_edges, self.p_edges],
            density=False
        )
        rho = torch.tensor(H, dtype=torch.float32).unsqueeze(0)  # [1, nq, np]
        rho = normalize_density(rho, self.dq, self.dp)
        return rho

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        rho_in = self.cloud_to_density(self.X[idx])
        rho_out = self.cloud_to_density(self.Y[idx])
        mu = torch.tensor(self.MU[idx], dtype=torch.float32)
        return rho_in, mu, rho_out
    

class SyntheticPhaseSpaceDataset(Dataset):
    def __init__(self, n_samples: int, grid: PhaseSpaceGrid, seed: int = 42):
        self.grid = grid
        self.n_samples = n_samples
        self.rho_in_list: List[torch.Tensor] = []
        self.mu_list: List[torch.Tensor] = []
        self.rho_out_list: List[torch.Tensor] = []

        rng = random.Random(seed)

        for _ in range(n_samples):
            spec = sample_density_spec(rng)
            mu = sample_mu(rng)

            # Input density on the native grid
            rho_in = evaluate_density(grid.Q, grid.P, spec)
            rho_in = normalize_density(rho_in, grid.dq, grid.dp)

            # Output density by inverse transport on the output grid
            q0, p0 = inverse_transport(grid.Q, grid.P, mu)
            rho_out = evaluate_density(q0, p0, spec)
            rho_out = normalize_density(rho_out, grid.dq, grid.dp)

            self.rho_in_list.append(rho_in.unsqueeze(0).float())   # [1, Nq, Np]
            self.mu_list.append(mu.float())                        # [6]
            self.rho_out_list.append(rho_out.unsqueeze(0).float()) # [1, Nq, Np]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.rho_in_list[idx],
            self.mu_list[idx],
            self.rho_out_list[idx],
        )


######### CONDITIONAL FNO MODEL #########

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, Cin, X, Y], w: [Cin, Cout, X, Y]
        return torch.einsum("bcxy,coxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, n1, n2 = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            n1,
            n2 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, n1)
        m2 = min(self.modes2, n2 // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(n1, n2), norm="ortho")
        return x


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.w(x)
        y = self.norm(y)
        return F.gelu(y)


class ConditionalFNO2d(nn.Module):
    def __init__(
        self,
        mu_dim: int,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 48,
        depth: int = 4,
    ):
        super().__init__()
        self.mu_dim = mu_dim

        # channels = rho + q + p + mu-broadcast
        in_channels = 1 + 2 + mu_dim

        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(depth)]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, rho_in: torch.Tensor, mu: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        rho_in: [B, 1, Nq, Np]
        mu:     [B, m]
        coords: [1, 2, Nq, Np]
        """
        bsz, _, nq, np_ = rho_in.shape
        mu_field = mu[:, :, None, None].expand(bsz, self.mu_dim, nq, np_)
        coord_field = coords.expand(bsz, -1, -1, -1)

        x = torch.cat([rho_in, coord_field, mu_field], dim=1)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        logits = self.out_proj(x)
        rho = F.softplus(logits)
        return rho


################### TRAINING LOOP ###################

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    grid: PhaseSpaceGrid,
    device: str = "cpu",
    epochs: int = 40,
    lr: float = 3e-4,
) -> nn.Module:
    model.to(device)
    coords = grid.coord_channels().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for rho_in, mu, rho_out in train_loader:
            rho_in = rho_in.to(device)
            mu = mu.to(device)
            rho_out = rho_out.to(device)

            pred = model(rho_in, mu, coords)
            pred = normalize_density(pred, grid.dq, grid.dp)

            loss_l2 = ((pred - rho_out) ** 2).sum(dim=(-2, -1)).mean() * grid.dq * grid.dp
            loss_kl = safe_kl(rho_out, pred, grid.dq, grid.dp)
            loss = loss_l2 + 1e-2 * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rho_in.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rho_in, mu, rho_out in val_loader:
                rho_in = rho_in.to(device)
                mu = mu.to(device)
                rho_out = rho_out.to(device)

                pred = model(rho_in, mu, coords)
                pred = normalize_density(pred, grid.dq, grid.dp)

                loss_mse = F.mse_loss(pred, rho_out)
                loss_kl = safe_kl(rho_out, pred, grid.dq, grid.dp)
                loss = loss_mse + 1e-2 * loss_kl
                val_loss += loss.item() * rho_in.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6e} | val={val_loss:.6e} | best={best_val:.6e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


########## TRAINING AND INFERENCE ##########

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

grid = PhaseSpaceGrid(nq=64, np_=64, qlim=4.0, plim=4.0)


data = np.load("xsuite_operator_dataset.npz")
q_all = np.concatenate([data["X"][:, :, 0].ravel(), data["Y"][:, :, 0].ravel()])
p_all = np.concatenate([data["X"][:, :, 3].ravel(), data["Y"][:, :, 3].ravel()])

qlim = float(np.max(np.abs(q_all)))
plim = float(np.max(np.abs(p_all)))

grid = PhaseSpaceGrid(nq=64, np_=64, qlim=qlim, plim=plim)
dataset = XsuiteDensityDataset(path="xsuite_operator_dataset.npz", grid=grid, plane=(0, 3))


train_size = int(0.875 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

model = ConditionalFNO2d(
    mu_dim=3,   # kf1, kd1, kf2
    modes1=10,
    modes2=10,
    width=10,
    depth=2,
)

model = train_model(
    model,
    train_loader,
    val_loader,
    grid,
    device=device,
    epochs=40,
    lr=3e-4,
)

save_dict = {
    "model_state_dict": model.state_dict(),
    "mu_dim": 3,
    "modes1": 16,
    "modes2": 16,
    "width": 48,
    "depth": 4,
    "nq": grid.nq,
    "np_": grid.np_,
    "qlim": grid.qlim,
    "plim": grid.plim,
}

torch.save(save_dict, "conditional_fno_xsuite.pt")
print("Model saved to conditional_fno_xsuite.pt")


### SMALL INFERENCE TEST
model.eval()
coords = grid.coord_channels().to(device)
rho_in, mu, rho_out = dataset[0]
with torch.no_grad():
        pred = model(
            rho_in.unsqueeze(0).to(device),
            mu.unsqueeze(0).to(device),
            coords,
        )
        pred = normalize_density(pred, grid.dq, grid.dp)

print("Example shapes:")
print("rho_in :", rho_in.shape)     # [1, Nq, Np]
print("mu     :", mu.shape)         # [3]
print("rho_out:", rho_out.shape)    # [1, Nq, Np]
print("pred   :", pred.shape)       # [1, 1, Nq, Np]")

###### PREDICTIONS ON MULTIPLE SAMPLES ######

rho_in_2d, pred_2d, rho_out_2d, mu = predict_from_dataset(
    model, dataset, grid, idx=0, device=device
)
print("mu =", mu.numpy())
print("pred shape =", pred_2d.shape)

n_pred = 5
predictions = []

for i in range(n_pred):
    rho_in, pred, rho_out, mu = predict_from_dataset(
        model, dataset, grid, idx=i, device=device
    )
    predictions.append({
        "rho_in": rho_in.numpy(),
        "pred": pred.numpy(),
        "rho_out": rho_out.numpy(),
        "mu": mu.numpy(),
    })

np.savez(
    "density_predictions.npz",
    rho_in=np.stack([d["rho_in"] for d in predictions]),
    pred=np.stack([d["pred"] for d in predictions]),
    rho_out=np.stack([d["rho_out"] for d in predictions]),
    mu=np.stack([d["mu"] for d in predictions]),
)
###### VISUALIZATION  ######
import matplotlib.pyplot as plt

rho_in, pred, rho_out, mu = predict_from_dataset(model, dataset, grid, idx=0, device=device)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(rho_in.numpy(), origin="lower", aspect="auto")
axs[0].set_title("rho_in")

axs[1].imshow(pred.numpy(), origin="lower", aspect="auto")
axs[1].set_title("prediction")

axs[2].imshow(rho_out.numpy(), origin="lower", aspect="auto")
axs[2].set_title("target rho_out")

plt.tight_layout()
plt.show()

