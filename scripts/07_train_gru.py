import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from rom.models_gru import LatentGRU
from rom.latent_dataset import LatentStepDataset


def main():
    cfg_path = project_root / "configs" / "train_gru.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    latent_h5 = (project_root / cfg["latent_h5"]).resolve()
    ckpt_dir = (project_root / cfg["ckpt_dir"]).resolve()
    loss_csv = (project_root / cfg["loss_csv"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])

    train_ds = LatentStepDataset(str(latent_h5), "idx_train")
    val_ds = LatentStepDataset(str(latent_h5), "idx_val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # infer latent_dim
    x0, _ = train_ds[0]
    latent_dim = int(x0.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Latent dim:", latent_dim)
    print("Train pairs:", len(train_ds), " Val pairs:", len(val_ds))

    model = LatentGRU(latent_dim=latent_dim, hidden_dim=128, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # (B, latent_dim)
            y = y.to(device)

            x_in = x.unsqueeze(1)  # (B, 1, latent_dim)
            y_hat = model(x_in)

            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x.unsqueeze(1))
                loss = loss_fn(y_hat, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= max(len(val_ds), 1)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        with open(loss_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = ckpt_dir / "best_gru.pt"
            torch.save({"model_state": model.state_dict(), "latent_dim": latent_dim, "val_loss": best_val}, ckpt_path)
            print("✅ Saved best GRU:", ckpt_path)

    print("\nDone. Best val loss:", best_val)


if __name__ == "__main__":
    main()
