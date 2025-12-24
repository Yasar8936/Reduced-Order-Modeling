import sys
from pathlib import Path

# Make src/ importable no matter where we run from
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))


import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from rom.datasets import H5VelocityDataset
from rom.models_ae import Simple3DAutoEncoder



def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "train_ae.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_h5 = (project_root / cfg["data_h5"]).resolve()
    ckpt_dir = (project_root / cfg["ckpt_dir"]).resolve()
    loss_csv = (project_root / cfg["loss_csv"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    latent_dim = int(cfg["latent_dim"])

    train_ds = H5VelocityDataset(str(data_h5), cfg["train_key"])
    val_ds = H5VelocityDataset(str(data_h5), cfg["val_key"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = Simple3DAutoEncoder(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    # CSV log
    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                loss = loss_fn(x_hat, x)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_ds)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        # log
        with open(loss_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = ckpt_dir / "best_ae.pt"
            torch.save(
                {"model_state": model.state_dict(), "latent_dim": latent_dim, "val_loss": best_val},
                ckpt_path
            )
            print("✅ Saved best checkpoint:", ckpt_path)

    print("\nDone. Best val loss:", best_val)


if __name__ == "__main__":
    main()
