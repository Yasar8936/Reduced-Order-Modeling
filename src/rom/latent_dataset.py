from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LatentStepDataset(Dataset):
    """
    Creates (latent[t] -> latent[t+1]) pairs from the latent HDF5 file.
    """

    def __init__(self, latent_h5: str, index_key: str):
        self.latent_h5 = str(Path(latent_h5).resolve())

        with h5py.File(self.latent_h5, "r") as f:
            self.latent = f["latent"][:]  # (N, latent_dim)
            base_idx = f[index_key][:].astype(np.int64)

        # We need pairs (t, t+1). So we keep only indices where t+1 exists and stays in range.
        self.pairs = []
        valid_set = set(base_idx.tolist())
        for t in base_idx:
            if (t + 1) in valid_set:
                self.pairs.append((int(t), int(t + 1)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        t, tp1 = self.pairs[i]
        x = torch.from_numpy(self.latent[t]).float()      # (latent_dim,)
        y = torch.from_numpy(self.latent[tp1]).float()    # (latent_dim,)
        return x, y
