from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NPZSequenceConfig:
    bins: int = 64
    max_elements: int = 128


class XsuiteNPZDataset(Dataset):
    """Loads a dataset bundle produced by scripts/data_generator_neural.py

    Expects keys:
      - X_cloud: (N, Np, 6)
      - MU: (N, 3)

    If your generator only produces one-step outputs (Y_cloud), this dataset will still work
    for VAE pretraining (reconstruction of X_cloud -> X_cloud).

    For multi-element supervision you need sequences (X_t per element). See train_tracking.py.
    """

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        self.data = np.load(self.npz_path, allow_pickle=True)

        self.X_cloud = self.data["X_cloud"].astype(np.float32)
        self.MU = self.data["MU"].astype(np.float32)

        # split indices if exist
        self.train_idx = self.data.get("train")
        self.val_idx = self.data.get("val")
        self.test_idx = self.data.get("test")

    def __len__(self) -> int:
        return self.X_cloud.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "cloud": self.X_cloud[idx],
            "mu": self.MU[idx],
        }


def collate_cloud(batch):
    cloud = torch.from_numpy(np.stack([b["cloud"] for b in batch], axis=0))  # (B,Np,6)
    mu = torch.from_numpy(np.stack([b["mu"] for b in batch], axis=0))  # (B,3)
    return {"cloud": cloud, "mu": mu}
