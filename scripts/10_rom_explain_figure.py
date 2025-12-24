import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from rom.models_ae import Simple3DAutoEncoder
from rom.models_gru import LatentGRU


def to_torch_snapshot(arr_zyx3, device):
    # (z,y,x,3) -> (1,3,z,y,x)
    arr = np.transpose(arr_zyx3, (3, 0, 1, 2)).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def to_numpy_snapshot(x_torch):
    # (1,3,z,y,x) -> (z,y,x,3)
    arr = x_torch.squeeze(0).detach().cpu().numpy()
    return np.transpose(arr, (1, 2, 3, 0)).astype(np.float32)


def main():
    # Load same config you used for eval
    cfg_path = project_root / "configs" / "eval_rom.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_h5 = (project_root / cfg["data_h5"]).resolve()
    ae_ckpt = (project_root / cfg["ae_ckpt"]).resolve()
    gru_ckpt = (project_root / cfg["gru_ckpt"]).resolve()

    # ---- What to show ----
    t0 = 42                  # rollout start
    k = 7                    # predict t0+k
    z_slice = 16             # middle slice
    component = 0            # 0=u, 1=v, 2=w

    out_png = project_root / "results" / "figures" / f"rom_full_meaning_t{t0}_k{k}_z{z_slice}_c{component}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

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

    # ---- Load ground truth snapshots ----
    with h5py.File(str(data_h5), "r") as f:
        V = f["velocity_norm"]  # (N,z,y,x,3)
        gt_t0 = np.asarray(V[t0], dtype=np.float32)
        gt_tk = np.asarray(V[t0 + k], dtype=np.float32)

    # ---- Run AE reconstruction at t0 (quality of compression) ----
    with torch.no_grad():
        x0 = to_torch_snapshot(gt_t0, device)
        z0 = ae.encode(x0)
        x0_recon = ae.decode(z0)

    recon_t0 = to_numpy_snapshot(x0_recon)  # (z,y,x,3)

    # ---- Run ROM rollout: AE+GRU+AE to get prediction at t0+k ----
    with torch.no_grad():
        z_curr = z0
        x_pred = None
        for _ in range(k):
            z_next = gru(z_curr.unsqueeze(1))  # (1, latent_dim)
            x_next = ae.decode(z_next)         # (1,3,z,y,x)
            z_curr = z_next
            x_pred = x_next

    pred_tk = to_numpy_snapshot(x_pred)

    # ---- Prepare slices ----
    gt0_img = gt_t0[z_slice, :, :, component]
    recon0_img = recon_t0[z_slice, :, :, component]
    gtk_img = gt_tk[z_slice, :, :, component]
    predk_img = pred_tk[z_slice, :, :, component]
    err_recon = np.abs(recon0_img - gt0_img)
    err_pred = np.abs(predk_img - gtk_img)

    # ---- Plot layout (2 rows x 3 cols) ----
    plt.figure(figsize=(14, 8))

    # Row 1: AE reconstruction meaning
    plt.subplot(2, 3, 1)
    plt.imshow(gt0_img, aspect="auto")
    plt.title(f"Ground truth @ t={t0}")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 2)
    plt.imshow(recon0_img, aspect="auto")
    plt.title("AE reconstruction @ t0\n(Compression quality)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 3)
    plt.imshow(err_recon, aspect="auto")
    plt.title("Abs error (GT vs Recon)\n@ t0")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Row 2: ROM prediction meaning
    plt.subplot(2, 3, 4)
    plt.imshow(gtk_img, aspect="auto")
    plt.title(f"Ground truth @ t={t0+k}")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 5)
    plt.imshow(predk_img, aspect="auto")
    plt.title(f"ROM prediction @ t0+{k}\n(Encode→GRU→Decode)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 6)
    plt.imshow(err_pred, aspect="auto")
    plt.title("Abs error (GT vs ROM)\n@ t0+k")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Big title that explains ROM
    plt.suptitle(
        "Reduced-Order Model (ROM) meaning:\n"
        "Autoencoder compresses a 3D velocity field → GRU predicts latent evolution → Decoder reconstructs the field",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("✅ Saved:", out_png)


if __name__ == "__main__":
    main()
