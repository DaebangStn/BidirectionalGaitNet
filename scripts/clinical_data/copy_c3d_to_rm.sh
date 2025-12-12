#!/bin/bash
# Copy entire mocap session folders to RM patient c3d directories
# Source: /mnt/blue8T/CP/Gait/GMFCS*/[patient_name]/[session_id]/
# Dest:   /mnt/blue8T/CP/RM/[PID]/c3d/(pre|post|post2)/

GAIT_DIR="/mnt/blue8T/CP/Gait"
RM_DIR="/mnt/blue8T/CP/RM"

echo "Copying mocap session folders to RM directories..."
echo "=================================================="

for pid_dir in "$RM_DIR"/*/; do
    if [ ! -d "$pid_dir" ]; then
        continue
    fi

    pid=$(basename "$pid_dir")
    metadata_file="$pid_dir/metadata.yaml"

    if [ ! -f "$metadata_file" ]; then
        echo "  [SKIP] $pid - no metadata.yaml"
        continue
    fi

    # Extract name and gmfcs from metadata.yaml
    name=$(grep "^name:" "$metadata_file" | sed 's/^name: *//')
    gmfcs=$(grep "^gmfcs:" "$metadata_file" | sed 's/^gmfcs: *//')

    if [ -z "$name" ] || [ -z "$gmfcs" ]; then
        echo "  [SKIP] $pid - missing name or gmfcs in metadata"
        continue
    fi

    # Find source directory
    source_dir="$GAIT_DIR/GMFCS$gmfcs/$name"

    if [ ! -d "$source_dir" ]; then
        echo "  [SKIP] $pid ($name) - source not found: $source_dir"
        continue
    fi

    # Get session folders sorted ascending
    sessions=($(find "$source_dir" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -n))
    session_count=${#sessions[@]}

    if [ "$session_count" -lt 2 ]; then
        echo "  [SKIP] $pid ($name) - only $session_count session(s)"
        continue
    fi

    # Destination c3d directory
    c3d_dir="$pid_dir/gait"

    # Copy sessions
    echo "  [COPY] $pid ($name) - $session_count sessions"

    # Session 1 -> pre
    mkdir -p "$c3d_dir/pre"
    cp -r "$source_dir/${sessions[0]}/." "$c3d_dir/pre/"
    echo "         pre: ${sessions[0]}"

    # Session 2 -> post
    mkdir -p "$c3d_dir/post"
    cp -r "$source_dir/${sessions[1]}/." "$c3d_dir/post/"
    echo "         post: ${sessions[1]}"

    # Session 3 -> post2 (if exists)
    if [ "$session_count" -ge 3 ]; then
        mkdir -p "$c3d_dir/post2"
        cp -r "$source_dir/${sessions[2]}/." "$c3d_dir/post2/"
        echo "         post2: ${sessions[2]}"
    fi

    # Write c3d/metadata.yaml
    c3d_metadata="$c3d_dir/metadata.yaml"
    echo "pre: ${sessions[0]}" > "$c3d_metadata"
    echo "post: ${sessions[1]}" >> "$c3d_metadata"
    if [ "$session_count" -ge 3 ]; then
        echo "post2: ${sessions[2]}" >> "$c3d_metadata"
    fi

done

echo "=================================================="
echo "Done."
