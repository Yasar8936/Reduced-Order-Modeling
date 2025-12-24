from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5VelocityDataset(Dataset):
   

    def __init__(self, h5_path: str, index_key: str):
        self.h5_path = str(Path(h5_path).resolve())
        self.index_key = index_key

        # read indices once
        with h5py.File(self.h5_path, "r") as f:
            self.indices = f[index_key][:].astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        with h5py.File(self.h5_path, "r") as f:
            arr = f["velocity_norm"][idx]  # (z, y, x, 3)

        # (z,y,x,3) -> (3,z,y,x)
        arr = np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)
        x = torch.from_numpy(arr)
        return x
