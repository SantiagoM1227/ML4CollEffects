from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentNPZDataset(Dataset):
    """
    Loads latent dataset exported from Stage 1:

      z0: (N,256)
      z1: (N,256)
      MU: (N,3)
      train/val/test: index arrays
    """

    def __init__(self, latent_npz: str | Path, split: str = "train"):
        self.latent_npz = Path(latent_npz)
        self.data = np.load(self.latent_npz, allow_pickle=True)

        self.z0 = self.data["z0"].astype(np.float32)
        self.z1 = self.data["z1"].astype(np.float32)
        self.MU = self.data["MU"].astype(np.float32)

        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be one of train/val/test/all, got {split}")

        if split == "all":
            self.idx = np.arange(self.z0.shape[0], dtype=np.int64)
        else:
            if split not in self.data:
                raise KeyError(f"latent dataset missing split indices '{split}'")
            self.idx = self.data[split].astype(np.int64)

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = int(self.idx[i])
        return {
            "z0": torch.from_numpy(self.z0[j]),
            "z1": torch.from_numpy(self.z1[j]),
            "mu": torch.from_numpy(self.MU[j]),
        }