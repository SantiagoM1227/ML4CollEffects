from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentNPZDataset(Dataset):
    """Stage-2 dataset exported from Stage-1 latents."""

    def __init__(self, latent_npz: str | Path, split: str = "train"):
        self.latent_npz = Path(latent_npz)
        self.data = np.load(self.latent_npz, allow_pickle=True)

        self.z0 = self.data["z0"].astype(np.float32)
        self.z1 = self.data["z1"].astype(np.float32)
        self.MU = self.data["MU"].astype(np.float32)

        if "s" in self.data:
            self.s = self.data["s"].astype(np.float32)
        else:
            self.s = np.zeros((self.z0.shape[0],), dtype=np.float32)

        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be one of train/val/test/all, got {split}")

        if split == "all":
            self.idx = np.arange(self.z0.shape[0], dtype=np.int64)
        else:
            if split in self.data:
                self.idx = self.data[split].astype(np.int64)
            else:
                # fallback for missing split arrays
                if split == "train":
                    n = self.z0.shape[0]
                    n_train = int(0.8 * n)
                    self.idx = np.arange(0, n_train, dtype=np.int64)
                elif split == "val":
                    n = self.z0.shape[0]
                    n_train = int(0.8 * n)
                    self.idx = np.arange(n_train, n, dtype=np.int64)
                else:
                    self.idx = np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = int(self.idx[i])
        return {
            "z0": torch.from_numpy(self.z0[j]),
            "z1": torch.from_numpy(self.z1[j]),
            "mu": torch.from_numpy(self.MU[j]),
            "s": torch.tensor(self.s[j], dtype=torch.float32),
        }
