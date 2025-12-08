#!/bin/bash
# Copy DICOM folders to RM patient directories with reorganization
# Source: /mnt/blue8T/CP/Dicom/GMFCS */[PID]_*/[ser_no]_수술[1|2]_[name]/
# Dest:   /mnt/blue8T/CP/RM/[PID]/dicom/op[1|2]/[name]/

DICOM_DIR="/mnt/blue8T/CP/Dicom"
RM_DIR="/mnt/blue8T/CP/RM"

echo "Copying DICOM folders to RM directories..."
echo "=================================================="

for pid_dir in "$RM_DIR"/*/; do
    if [ ! -d "$pid_dir" ]; then
        continue
    fi

    pid=$(basename "$pid_dir")

    # Find source directory by matching PID prefix
    source_dir=$(find "$DICOM_DIR" -maxdepth 2 -type d -name "${pid}_*" 2>/dev/null | head -1)

    if [ -z "$source_dir" ] || [ ! -d "$source_dir" ]; then
        echo "  [SKIP] $pid - source not found"
        continue
    fi

    # Destination dicom directory
    dicom_dir="$pid_dir/dicom"

    # Clear existing dicom contents
    rm -rf "$dicom_dir"/*

    # Create op1 and op2 directories
    mkdir -p "$dicom_dir/op1"
    mkdir -p "$dicom_dir/op2"

    echo "  [COPY] $pid"

    # Process each subfolder
    for src_folder in "$source_dir"/*/; do
        if [ ! -d "$src_folder" ]; then
            continue
        fi

        folder_name=$(basename "$src_folder")

        # Parse folder name: [ser_no]_수술[1|2]_[optional_n]_[name]
        # Extract surgery number (1 or 2)
        if [[ "$folder_name" =~ _수술1_ ]]; then
            op_dir="op1"
            # Remove prefix: [ser_no]_수술1_
            clean_name=$(echo "$folder_name" | sed 's/^[0-9]*_수술1_//')
        elif [[ "$folder_name" =~ _수술2_ ]]; then
            op_dir="op2"
            # Remove prefix: [ser_no]_수술2_ or [ser_no]_수술2_[n]_
            clean_name=$(echo "$folder_name" | sed 's/^[0-9]*_수술2_//' | sed 's/^[0-9]*_//')
        else
            echo "         [WARN] Unknown pattern: $folder_name"
            continue
        fi

        dest_path="$dicom_dir/$op_dir/$clean_name"

        # Handle duplicates by appending _n
        if [ -d "$dest_path" ]; then
            n=2
            while [ -d "${dest_path}_$n" ]; do
                n=$((n + 1))
            done
            dest_path="${dest_path}_$n"
        fi

        cp -r "$src_folder" "$dest_path"
    done

    # Show result
    op1_count=$(find "$dicom_dir/op1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    op2_count=$(find "$dicom_dir/op2" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    echo "         op1: $op1_count folders, op2: $op2_count folders"

done

echo "=================================================="
echo "Done."
