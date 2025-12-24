import json
from pathlib import Path

import h5py
import numpy as np
import yaml


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "preprocess_channel.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_h5 = (project_root / cfg["raw_h5"]).resolve()
    out_h5 = (project_root / cfg["processed_h5"]).resolve()
    stats_json = (project_root / cfg["stats_json"]).resolve()

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    train_frac = float(cfg["train_frac"])
    val_frac = float(cfg["val_frac"])
    test_frac = float(cfg["test_frac"])

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    norm_type = cfg.get("normalization", "standard")
    compression = cfg.get("compression", "gzip")
    comp_level = int(cfg.get("compression_level", 4))

    print("Raw file      :", raw_h5)
    print("Processed file:", out_h5)

    with h5py.File(str(raw_h5), "r") as f:
        V = f["velocity"]  # (N, z, y, x, 3)
        times = f["time_index"][:] if "time_index" in f else None

        N = V.shape[0]
        z, y, x, c = V.shape[1:]
        print("Raw velocity shape:", V.shape)

        # ---- split indices (time-ordered) ----
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        n_test = N - n_train - n_val

        idx_train = np.arange(0, n_train)
        idx_val = np.arange(n_train, n_train + n_val)
        idx_test = np.arange(n_train + n_val, N)

        print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

        # ---- stats from TRAIN only ----
        train_data = V[idx_train]  # shape (n_train, z, y, x, 3)
        mean = train_data.mean(axis=(0, 1, 2, 3))           # (3,)
        std = train_data.std(axis=(0, 1, 2, 3)) + 1e-8      # (3,)

        print("Train mean (u,v,w):", mean)
        print("Train std  (u,v,w):", std)

        # ---- write processed ----
        with h5py.File(str(out_h5), "w") as g:
            # indices
            g.create_dataset("idx_train", data=idx_train.astype(np.int32))
            g.create_dataset("idx_val", data=idx_val.astype(np.int32))
            g.create_dataset("idx_test", data=idx_test.astype(np.int32))

            if times is not None:
                g.create_dataset("time_index", data=times.astype(np.int32))

            # stats
            g.create_dataset("mean", data=mean.astype(np.float32))
            g.create_dataset("std", data=std.astype(np.float32))
            g.attrs["normalization"] = norm_type

            # normalized velocity dataset
            dset = g.create_dataset(
                "velocity_norm",
                shape=V.shape,
                dtype=np.float32,
                chunks=(1, z, y, x, 3),
                compression=compression,
                compression_opts=comp_level
            )

            # normalize and write sample-by-sample (memory safe)
            for i in range(N):
                arr = np.asarray(V[i], dtype=np.float32)

                if norm_type == "standard":
                    arr = (arr - mean) / std
                else:
                    raise ValueError(f"Unknown normalization type: {norm_type}")

                dset[i] = arr

                if (i + 1) % 10 == 0 or (i + 1) == N:
                    print(f"Wrote {i+1}/{N} normalized samples")

    # save stats json for easy reading later
    stats = {
        "raw_h5": str(raw_h5),
        "processed_h5": str(out_h5),
        "normalization": norm_type,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "splits": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    }

    with open(stats_json, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    print("\n✅ Done.")
    print("Stats saved:", stats_json)


if __name__ == "__main__":
    main()
