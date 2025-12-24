import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from rom.models_ae import Simple3DAutoEncoder
from rom.models_gru import LatentGRU


def to_torch_snapshot(arr_zyx3, device):
    # (z,y,x,3) -> (1,3,z,y,x)
    arr = np.transpose(arr_zyx3, (3, 0, 1, 2)).astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    return x


def to_numpy_snapshot(x_torch):
    # (1,3,z,y,x) -> (z,y,x,3)
    arr = x_torch.squeeze(0).detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 3, 0)).astype(np.float32)
    return arr


def main():
    cfg_path = project_root / "configs" / "eval_rom.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_h5 = (project_root / cfg["data_h5"]).resolve()
    ae_ckpt = (project_root / cfg["ae_ckpt"]).resolve()
    gru_ckpt = (project_root / cfg["gru_ckpt"]).resolve()
    rollout_steps = int(cfg["rollout_steps"])

    report_txt = (project_root / cfg["report_txt"]).resolve()
    report_txt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load AE ----
    ae_state = torch.load(ae_ckpt, map_location=device)
    latent_dim = int(ae_state["latent_dim"])
    ae = Simple3DAutoEncoder(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ae_state["model_state"])
    ae.eval()

    # ---- Load GRU ----
    gru_state = torch.load(gru_ckpt, map_location=device)
    gru = LatentGRU(latent_dim=latent_dim, hidden_dim=128, num_layers=1).to(device)
    gru.load_state_dict(gru_state["model_state"])
    gru.eval()

    mse_fn = nn.MSELoss(reduction="mean")

    # ---- Load data + test indices ----
    with h5py.File(str(data_h5), "r") as f:
        V = f["velocity_norm"]  # (N,z,y,x,3)
        idx_test = f["idx_test"][:].astype(np.int64)

        # Sort to ensure time order
        idx_test = np.sort(idx_test)
        print("Test indices:", idx_test)

        # We'll evaluate rollouts starting at each test index where future steps exist
        starts = []
        for t0 in idx_test:
            t_end = t0 + rollout_steps
            if t_end < V.shape[0]:
                starts.append(int(t0))

        if len(starts) == 0:
            raise RuntimeError("Not enough timesteps for the chosen rollout_steps.")

        print(f"Evaluating {len(starts)} rollouts, each {rollout_steps} steps forward")

        all_mse = []
        all_times = []

        for t0 in starts:
            # Ground truth sequence (t0 ... t0+K)
            gt = [np.asarray(V[t0 + k], dtype=np.float32) for k in range(rollout_steps + 1)]

            # ROM rollout
            with torch.no_grad():
                x0 = to_torch_snapshot(gt[0], device)

                # encode initial latent
                z = ae.encode(x0)  # (1, latent_dim)

                t_start = time.perf_counter()

                preds = [x0]  # store predicted snapshots (torch)
                z_curr = z

                for k in range(rollout_steps):
                    # GRU predicts next latent
                    z_next = gru(z_curr.unsqueeze(1))  # input (B,1,latent_dim) -> (B,latent_dim)
                    # decode to snapshot
                    x_next = ae.decode(z_next)
                    preds.append(x_next)
                    z_curr = z_next

                t_end = time.perf_counter()
                all_times.append(t_end - t_start)

            # compute MSE in physical space for steps 1..K (exclude step0 because it's input)
            rollout_mse = []
            for k in range(1, rollout_steps + 1):
                pred_np = to_numpy_snapshot(preds[k])
                mse = np.mean((pred_np - gt[k]) ** 2)
                rollout_mse.append(float(mse))

            all_mse.append(rollout_mse)

        all_mse = np.array(all_mse)  # (num_rollouts, K)
        mean_mse_per_step = all_mse.mean(axis=0)
        mean_time = float(np.mean(all_times))

    # ---- Report ----
    lines = []
    lines.append(f"ROM Evaluation Report")
    lines.append(f"Data file: {data_h5}")
    lines.append(f"AE ckpt  : {ae_ckpt}")
    lines.append(f"GRU ckpt : {gru_ckpt}")
    lines.append(f"Latent dim: {latent_dim}")
    lines.append(f"Rollout steps: {rollout_steps}")
    lines.append("")
    lines.append("Mean MSE per predicted step (velocity_norm space):")
    for i, v in enumerate(mean_mse_per_step, start=1):
        lines.append(f"  step {i}: {v:.6f}")
    lines.append("")
    lines.append(f"Average ROM rollout time for {rollout_steps} steps: {mean_time:.6f} seconds")
    lines.append("(This is only ROM compute time: encode+GRU+decode, not data download.)")

    report_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n✅ Saved report:", report_txt)
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
