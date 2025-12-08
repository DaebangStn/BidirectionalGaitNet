#!/bin/bash
# Check if each patient has exactly 2 mocap folders
# Expected structure: /mnt/blue8T/CP/Gait/GMFCS*/[patient_name]/[mocap1] [mocap2]

GAIT_DIR="/mnt/blue8T/CP/Gait"

echo "Checking mocap folders in $GAIT_DIR..."
echo "=========================================="

errors=0
total=0

for gmfcs_dir in "$GAIT_DIR"/GMFCS*; do
    if [ ! -d "$gmfcs_dir" ]; then
        continue
    fi

    gmfcs_name=$(basename "$gmfcs_dir")

    for patient_dir in "$gmfcs_dir"/*/; do
        if [ ! -d "$patient_dir" ]; then
            continue
        fi

        patient_name=$(basename "$patient_dir")
        folder_count=$(find "$patient_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
        total=$((total + 1))

        if [ "$folder_count" -eq 2 ]; then
            echo "  [OK] $gmfcs_name/$patient_name: $folder_count folders"
        else
            echo "  [ERROR] $gmfcs_name/$patient_name: $folder_count folders (expected 2)"
            # List the folders
            find "$patient_dir" -maxdepth 1 -mindepth 1 -type d -printf "          - %f\n"
            errors=$((errors + 1))
        fi
    done
done

echo "=========================================="
echo "Total patients: $total"
echo "Patients with exactly 2 folders: $((total - errors))"
echo "Patients with issues: $errors"

if [ $errors -gt 0 ]; then
    exit 1
fi
