import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

# make src importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from rom.models_ae import Simple3DAutoEncoder


def main():
    cfg_path = project_root / "configs" / "train_gru.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_h5 = (project_root / cfg["data_h5"]).resolve()
    ae_ckpt = (project_root / cfg["ae_ckpt"]).resolve()
    latent_h5 = (project_root / cfg["latent_h5"]).resolve()
    latent_h5.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Data  :", data_h5)
    print("AE ckpt:", ae_ckpt)
    print("Out   :", latent_h5)

    # load AE checkpoint
    ckpt = torch.load(ae_ckpt, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    model = Simple3DAutoEncoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with h5py.File(str(data_h5), "r") as f:
        V = f["velocity_norm"]  # (N, z, y, x, 3)
        N = V.shape[0]
        zdim, ydim, xdim, c = V.shape[1:]
        print("Velocity_norm shape:", V.shape)

        latents = np.zeros((N, latent_dim), dtype=np.float32)

        with torch.no_grad():
            for i in range(N):
                arr = np.asarray(V[i], dtype=np.float32)      # (z,y,x,3)
                arr = np.transpose(arr, (3, 0, 1, 2))         # (3,z,y,x)
                x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,3,z,y,x)

                zvec = model.encode(x)  # (1, latent_dim)
                latents[i] = zvec.squeeze(0).cpu().numpy()

                if (i + 1) % 10 == 0 or (i + 1) == N:
                    print(f"Encoded {i+1}/{N}")

    # save to latent_h5
    with h5py.File(str(latent_h5), "w") as g:
        g.create_dataset("latent", data=latents, compression="gzip", compression_opts=4)

        # copy split indices (so GRU uses the same train/val/test split)
        with h5py.File(str(data_h5), "r") as f:
            for k in ["idx_train", "idx_val", "idx_test"]:
                g.create_dataset(k, data=f[k][:])

            if "time_index" in f:
                g.create_dataset("time_index", data=f["time_index"][:])

    print("\n✅ Saved latent vectors:", latent_h5)
    print("latent shape:", latents.shape)


if __name__ == "__main__":
    main()
