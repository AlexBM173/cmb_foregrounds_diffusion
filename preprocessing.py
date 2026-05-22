import numpy as np
import os

root_directory = "/home/apb86/"
data_directory = os.path.join(root_directory, "rds/hpc-work")
output_directory = "/home/apb86/rds/hpc-work/halo_catalogue"

os.makedirs(output_directory, exist_ok=True)

# --- Extract and save per-slice catalogues ---

for slice_num in range(93, 100):
    filename = f"haloslc/haloslc_rot_{slice_num}_v050223.npz"
    working_file_path = os.path.join(data_directory, filename)

    with np.load(working_file_path, allow_pickle=True, mmap_mode="r") as x:
        mask = x["totm500"] >= 1e13
        filtered = {key: x[key][mask] for key in x.files}

    output_path = os.path.join(output_directory, f"halos_rot_{slice_num:03d}_m500gt1e13.npz")
    np.savez(output_path, **filtered)
    print(f"Slice {slice_num:3d}: {mask.sum()} halos saved -> {output_path}")

# --- Concatenate into a single catalogue ---

all_data = {}

for slice_num in range(4, 201):
    input_path = os.path.join(output_directory, f"halos_rot_{slice_num:03d}_m500gt1e13.npz")
    with np.load(input_path, allow_pickle=True) as x:
        if not all_data:
            all_data = {key: [x[key]] for key in x.files}
        else:
            for key in x.files:
                all_data[key].append(x[key])

combined = {key: np.concatenate(arrays, axis=0) for key, arrays in all_data.items()}

catalogue_path = os.path.join(output_directory, "halo_catalogue_m500gt3e14.npy")
np.savez(catalogue_path, **combined)

total = len(combined["totm500"])
print(f"Combined catalogue: {total} halos -> {catalogue_path}")
