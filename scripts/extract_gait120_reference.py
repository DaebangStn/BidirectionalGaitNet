#!/usr/bin/env python3
"""
Extract normative gait data from Gait120 parquet and export to HDF5.
Output format matches /kinematics group structure for RenderCkpt overlay.
"""
import polars as pl
import h5py
import numpy as np
from scipy import interpolate
from pathlib import Path

# Paths
PARQUET_PATH = Path("/mnt/blue8T/Gait120/dataset/averaged_data/averaged_kinematics_data.parquet")
OUTPUT_PATH = Path("/mnt/blue8T/Gait120/normative_kinematics.h5")

# Joint mapping: parquet name -> simulation key
JOINT_MAPPING = {
    "Hip_Flexion": "angle_HipR",
    "Hip_Int_Rot": "angle_HipIRR",
    "Hip_Adduction": "angle_HipAbR",
    "Knee_Flexion": "angle_KneeR",
    "Ankle_Dorsiflexion": "angle_AnkleR",
}


def main():
    # Load parquet
    df = pl.read_parquet(PARQUET_PATH)

    # Filter for LevelWalking task
    df = df.filter(pl.col("task") == "LevelWalking")

    # Target 100 evenly spaced points
    target_percent = np.linspace(0, 100, 100)

    # Prepare output data
    joint_keys = []
    mean_data = {}
    std_data = {}
    num_cycles = 0

    for parquet_joint, sim_key in JOINT_MAPPING.items():
        joint_data = df.filter(pl.col("joint") == parquet_joint)

        if joint_data.is_empty():
            print(f"Warning: No data for {parquet_joint}")
            continue

        # Sort by time_percent to ensure proper interpolation
        joint_data = joint_data.sort("time_percent")

        # Extract arrays
        time_vals = joint_data["time_percent"].to_numpy()
        mean_vals = joint_data["mean_angle"].to_numpy()
        std_vals = joint_data["std_angle"].to_numpy()

        # Get sample count (use max count across joints)
        counts = joint_data["count"].to_numpy()
        num_cycles = max(num_cycles, int(np.max(counts)))

        # Resample to 100 points
        f_mean = interpolate.interp1d(time_vals, mean_vals, kind='linear', fill_value='extrapolate')
        f_std = interpolate.interp1d(time_vals, std_vals, kind='linear', fill_value='extrapolate')

        mean_resampled = f_mean(target_percent)
        std_resampled = f_std(target_percent)

        joint_keys.append(sim_key)
        mean_data[sim_key] = mean_resampled
        std_data[sim_key] = std_resampled

        print(f"{parquet_joint} -> {sim_key}: mean range [{mean_resampled.min():.1f}, {mean_resampled.max():.1f}]")

    # Write HDF5
    with h5py.File(OUTPUT_PATH, 'w') as f:
        kin_group = f.create_group("kinematics")

        # Attributes
        kin_group.attrs["num_cycles"] = num_cycles
        kin_group.attrs["num_samples"] = 100
        # Use fixed-length string to match C++ HDF export format
        joint_names_str = ",".join(joint_keys)
        kin_group.attrs.create("joint_names", joint_names_str, dtype=f"S{len(joint_names_str) + 1}")

        # Datasets
        for key in joint_keys:
            kin_group.create_dataset(f"{key}_mean", data=mean_data[key].astype(np.float64))
            kin_group.create_dataset(f"{key}_std", data=std_data[key].astype(np.float64))

    print(f"\nExported to {OUTPUT_PATH}")
    print(f"  Joints: {', '.join(joint_keys)}")
    print(f"  Samples: 100, Cycles: {num_cycles}")


if __name__ == "__main__":
    main()
