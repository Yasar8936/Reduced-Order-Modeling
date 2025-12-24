import os
import time
import yaml
import h5py
import numpy as np
import inspect
from pathlib import Path


from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout


def call_getcutout(ds, variable, t, x_range, y_range, z_range, x_stride=1, y_stride=1, z_stride=1):
    """
    Works with both givernylocal getCutout signatures.
    Returns a dict-like object that contains velocity_XXXX dataset.
    """
    sig = inspect.signature(getCutout)
    param_names = list(sig.parameters.keys())

    if "time" in param_names:
        xyz_axes = np.array([x_range, y_range, z_range], dtype=int)
        xyz_strides = np.array([x_stride, y_stride, z_stride], dtype=int)
        return getCutout(ds, variable, t, xyz_axes, xyz_strides)

    # Newer signature: time included in axes
    t_range = [t, t]
    xyzt_axes = np.array([x_range, y_range, z_range, t_range], dtype=int)
    xyzt_strides = np.array([x_stride, y_stride, z_stride, 1], dtype=int)
    return getCutout(ds, variable, xyzt_axes, xyzt_strides)


def find_velocity_key(result):
    # In your output it was velocity_0001, velocity_0002, ...
    for k in result.keys():
        if str(k).startswith("velocity_"):
            return k
    raise KeyError(f"No velocity_XXXX key found. Keys: {list(result.keys())}")


def main():
    project_root = Path(__file__).resolve().parents[1]  # scripts/.. = project root
    cfg_path = project_root / "configs" / "download_channel.yaml"

    print("Project root:", project_root)
    print("Config path :", cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    token = os.getenv("GIVERNY_AUTH_TOKEN", "edu.jhu.pha.turbulence.testing-201406")

    dataset_title = cfg["dataset_title"]
    variable = cfg["variable"]

    t_start, t_end, t_step = cfg["t_start"], cfg["t_end"], cfg["t_step"]
    x_range, y_range, z_range = cfg["x_range"], cfg["y_range"], cfg["z_range"]
    x_stride, y_stride, z_stride = cfg["x_stride"], cfg["y_stride"], cfg["z_stride"]

    out_h5 = Path(cfg["out_h5"]).expanduser()
    out_h5 = out_h5.resolve() if out_h5.is_absolute() else (project_root / out_h5).resolve()

    print("✅ Will save to (absolute):", out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    ds = turb_dataset(dataset_title=dataset_title, output_path="./giverny_output", auth_token=token)

    # 1) Grab one sample to learn the shape (z,y,x,3)
    print("Fetching one sample to determine array shape...")
    first = call_getcutout(ds, variable, t_start, x_range, y_range, z_range, x_stride, y_stride, z_stride)
    vel_key = find_velocity_key(first)
    sample = np.asarray(first[vel_key], dtype=np.float32)
    # sample shape: (z, y, x, 3)
    z, y, x, c = sample.shape
    assert c == 3, f"Expected 3 velocity components, got {c}"
    print(f"Sample shape (z,y,x,3): {sample.shape}")

    # 2) Create HDF5 with an extendable dataset: (N, z, y, x, 3)
    comp = cfg.get("compression", "gzip")
    comp_lvl = int(cfg.get("compression_level", 4))

    with h5py.File(str(out_h5), "w") as f:
        dset = f.create_dataset(
            "velocity",
            shape=(0, z, y, x, 3),
            maxshape=(None, z, y, x, 3),
            dtype=np.float32,
            chunks=(1, z, y, x, 3),
            compression=comp,
            compression_opts=comp_lvl
        )

        # Save coordinates once (nice for later plotting/debugging)
        for coord_name in ["xcoor", "ycoor", "zcoor"]:
            if coord_name in first:
                f.create_dataset(coord_name, data=np.asarray(first[coord_name]))

        f.create_dataset("time_index", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(256,))

        # Save metadata
        f.attrs["dataset_title"] = dataset_title
        f.attrs["variable"] = variable
        f.attrs["x_range"] = str(x_range)
        f.attrs["y_range"] = str(y_range)
        f.attrs["z_range"] = str(z_range)
        f.attrs["strides"] = str([x_stride, y_stride, z_stride])

        # 3) Loop timesteps and append
        n = 0
        t0 = time.time()
        for t in range(t_start, t_end + 1, t_step):
            print(f"Downloading t={t} ...")
            res = call_getcutout(ds, variable, t, x_range, y_range, z_range, x_stride, y_stride, z_stride)
            k = find_velocity_key(res)
            arr = np.asarray(res[k], dtype=np.float32)

            # Append one sample
            dset.resize((n + 1, z, y, x, 3))
            dset[n] = arr

            f["time_index"].resize((n + 1,))
            f["time_index"][n] = t

            n += 1

        dt = time.time() - t0
        print(f"\n✅ Done. Saved {n} samples to: {out_h5}")
        print(f"Total download time: {dt:.2f} s  |  Avg per sample: {dt/max(n,1):.2f} s")


if __name__ == "__main__":
    main()