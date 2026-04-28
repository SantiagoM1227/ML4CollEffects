from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np


class CloudDataset(Dataset):
    def __init__(self,npz_path:str,split : str = "train"):
        raw = np.load(npz_path, allow_pickle=True)
        idx = raw[split]
        self.x = raw["X_cloud"][idx]   #[B,Np,6]
        self.y = raw["Y_cloud"][idx]   #[B,Np,6]
        self.mu = raw["MU"][idx]       #[B,3]
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,i):
        return (
            torch.from_numpy(self.x[i]).float(),
            torch.from_numpy(self.y[i]).float(),
            torch.from_numpy(self.mu[i]).float()
        )

def cov2(a: torch.Tensor, b: torch.Tensor, eps : float = 1e-18) -> torch.Tensor:
    """
    a, b : [B,N]

    returns cov(a,b): [B]
    """
    am = a - a.mean(dim = 1, keepdim = True)
    bm = b - b.mean(dim = 1, keepdim = True)
    return (am * bm).mean(dim = 1) + eps


def emit_2d(u : torch.Tensor, pu : torch.Tensor) -> torch.Tensor:
    """
    u,pu : [B,N]
    epsilon = sqrt(<u^2><pu^2>- <upu>^2)
    returns : [B]
    """
    suu = cov2(u,u)
    spp = cov2(pu,pu)
    sup = cov2(u,pu)
    return torch.sqrt(torch.clamp(suu*spp-sup*sup,  min = 0.0))


def emittances_from_cloud(cloud: torch.Tensor):
    """
    cloud: [B, Np, 6] with ordering [x,y,zeta,px,py,delta]
    returns dict of emittances: each [B]
    """
    x = cloud[:, :, 0]
    y = cloud[:, :, 1]
    zeta = cloud[:, :, 2]
    px = cloud[:, :, 3]
    py = cloud[:, :, 4]
    delta = cloud[:, :, 5]

    ex = emit_2d(x, px)
    ey = emit_2d(y, py)
    ez = emit_2d(zeta, delta)  # optional longitudinal metric
    return {"ex": ex, "ey": ey, "ez": ez}

def chamfer_l2(a:torch.Tensor, b: torch.Tensor) -> torch.Tensor :
    """
    a : [B,Na,D], b:[B,Nb,D]
    returns scalar
    """
    #pairwise squared distances : [B,Na,Nb]
    dist = torch.cdist(a,b,p=2)**2
    min_a_to_b = dist.min(dim=2).values     #[B,Na]
    min_b_to_a = dist.min(dim=1).values     #[B,Nb]
    return (min_a_to_b.mean()+min_b_to_a.mean())

class CloudEncoder(nn.Module):
    def __init__(self, in_dim = 6, hidden = 128, latent_dim = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.GELU(),
            nn.Linear(hidden,hidden),
            nn.GELU(),
        )
        self.rho = nn.sequential(
            nn.Linear(hidden,hidden), 
            nn.GELU(),
            nn.Linear(hidden,latent_dim)
        )
    
    def forward(self,x):
        #x : [B, Np,6]
        h = self.phi(x)     # [B,Np,hidden]
        h = h.mean(dim=1)   # permutation - invariant pooling
        z = self.rho(h)     # [B, latent_dim]
        return z


class CloudDecoder(nn.Module):
    def __init__(self, latent_dim = 64, n_points = 4096, hidden = 256, out_dim = 6):
        super().__init__()
        self.n_points = n_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim,hidden),
            nn.GELU(),
            nn.Linear(hidden,hidden),
            nn.GELU(),
            nn.Linear(hidden,n_points * out_dim)
        )
    
    def forward(self,z):
        # z : [B, latent_dim]
        x = self.mlp(z)                                 #[B,n_points*out_dim]
        return x.view(z.shape[0], self.n_points, -1)    #[B,Np,6]
    

class CloudAE(nn.Module):
    def __init__(self, n_points, latent_dim = 64):
        super().__init__()
        self.enc = CloudEncoder(latent_dim=latent_dim)
        self.dec = CloudDecoder(latent_dim=latent_dim, n_points= n_points)

    def forward(self,x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat,z

#------------------------ ENCODER DECODER --------------------------------------

class TokenSetEncoder(nn.Module):
    """
    X: [B, Np, 6] -> Z: [B, T, C]
    Learned queries attend over particles (permutation-invariant).
    """
    def __init__(self, in_dim=6, hidden=128, token_dim=64, n_tokens=64):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )
        self.queries = nn.Parameter(torch.randn(n_tokens, token_dim) * 0.02)

        self.refine = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, x):
        h = self.phi(x)                       # [B, Np, C]
        q = self.queries.unsqueeze(0)          # [1, T, C]
        scores = torch.einsum("btc,bnc->btn", q, h)  # [B, T, Np]
        w = F.softmax(scores, dim=-1)
        z = torch.einsum("btn,bnc->btc", w, h)       # [B, T, C]
        return self.refine(z)

class TokenDecoder(nn.Module):
    """
    Z: [B, T, C] -> cloud_hat: [B, Np, 6]
    We flatten tokens then project to Np*6.
    """
    def __init__(self, n_tokens=64, token_dim=64, n_points=10000, hidden=512, out_dim=6):
        super().__init__()
        self.n_points = n_points
        self.out_dim = out_dim
        in_dim = n_tokens * token_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_points * out_dim),
        )

    def forward(self, z):
        B, T, C = z.shape
        x = z.reshape(B, T * C)
        out = self.net(x)
        return out.view(B, self.n_points, self.out_dim)

##------------------------- FNO LEARNING ---------------------------------------


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, _, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.compl_weight()[:, :, :n_modes],
        )
        return torch.fft.irfft(out_ft, n=T, dim=-1)

class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))

class LatentFNO1d(nn.Module):
    """
    Z_in: [B, T, Ctok]
    mu:   [B, 3]
    returns Z_out: [B, T, Ctok]
    """
    def __init__(self, token_dim=64, mu_dim=3, width=128, modes=16, depth=4, hidden_proj=128):
        super().__init__()
        self.token_dim = token_dim
        self.mu_dim = mu_dim

        in_channels = token_dim + mu_dim + 1  # + token coordinate
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Conv1d(width, hidden_proj, kernel_size=1)
        self.proj2 = nn.Conv1d(hidden_proj, token_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, z_in: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        B, T, C = z_in.shape
        tgrid = torch.linspace(0, 1, T, device=z_in.device).view(1, T, 1).expand(B, T, 1)
        mu_rep = mu.view(B, 1, self.mu_dim).expand(B, T, self.mu_dim)

        x = torch.cat([z_in, mu_rep, tgrid], dim=-1)  # [B,T,C+3+1]
        x = self.lift(x)                              # [B,T,W]
        x = x.permute(0, 2, 1)                        # [B,W,T]
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)                             # [B,C,T]
        return x.permute(0, 2, 1)                     # [B,T,C]

##--------------------------TRAINING LOOP ------------------------------------------
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    for x in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        loss = chamfer_l2(x_hat, x)
        b = x.size(0)
        loss_sum += loss.item() * b
        n += b
    return loss_sum / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-points", type=int, required=True)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="")  # path to save checkpoint
    args = ap.parse_args()

    device = args.device
    print("[INFO] device:", device)

    train_loader = DataLoader(
        CloudDataset(args.dataset, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        CloudDataset(args.dataset, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CloudAE(n_points=args.n_points, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        n = 0
        for x in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = chamfer_l2(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = x.size(0)
            tr_sum += loss.item() * b
            n += b

        tr = tr_sum / max(1, n)
        va = validate(model, val_loader, device)

        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch:03d} train_chamfer={tr:.4e} val_chamfer={va:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.save:
        torch.save({"state_dict": model.state_dict(), "cfg": vars(args)}, args.save)
        print("[INFO] saved:", args.save)

if __name__ == "__main__":
    main()