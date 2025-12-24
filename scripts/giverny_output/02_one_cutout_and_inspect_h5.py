import os
import h5py
import numpy as np
import inspect

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout, write_cutout_hdf5_and_xmf_files


def call_getcutout(ds, variable, time, x_range, y_range, z_range, x_stride=1, y_stride=1, z_stride=1):
    """
    Supports both givernylocal getCutout signatures:
      - old: getCutout(ds, variable, time, axes_ranges, strides)
      - new: getCutout(ds, variable, xyzt_axes_ranges, xyzt_strides)
    """
    sig = inspect.signature(getCutout)
    param_names = list(sig.parameters.keys())

    # Build xyz arrays
    xyz_axes = np.array([x_range, y_range, z_range], dtype=int)
    xyz_strides = np.array([x_stride, y_stride, z_stride], dtype=int)

    if "time" in param_names:
        # Old style (time passed separately)
        return getCutout(ds, variable, time, xyz_axes, xyz_strides)

    # New style (time included in axes)
    # Use a single time index as a range [t, t]
    t_range = [time, time]
    xyzt_axes = np.array([x_range, y_range, z_range, t_range], dtype=int)
    xyzt_strides = np.array([x_stride, y_stride, z_stride, 1], dtype=int)
    return getCutout(ds, variable, xyzt_axes, xyzt_strides)


def main():
    token = os.getenv("GIVERNY_AUTH_TOKEN", "edu.jhu.pha.turbulence.testing-201406")

    dataset_title = "channel"
    output_path = "./giverny_output"
    ds = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=token)

    variable = "velocity"
    time = 1

    x_range = [1, 4]
    y_range = [1, 32]
    z_range = [1, 32]

    result = call_getcutout(ds, variable, time, x_range, y_range, z_range)

    out_name = "one_cutout_debug"
    write_cutout_hdf5_and_xmf_files(ds, result, out_name)

    h5_path = os.path.join(output_path, f"{out_name}.h5")
    print(f"\n✅ Wrote: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        print("\n--- HDF5 top-level keys ---")
        for k in f.keys():
            print(" ", k)

        print("\n--- Walk all datasets ---")
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name} : shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(visitor)


if __name__ == "__main__":
    main()
