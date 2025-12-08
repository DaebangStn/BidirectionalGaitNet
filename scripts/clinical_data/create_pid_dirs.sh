#!/bin/bash
# Create directories for each PID in /mnt/blue8T/CP/RM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/home/geon/BidirectionalGaitNet/data/pid"

# Read PIDs from file (comma-separated)
IFS=',' read -ra PIDS < "$PID_FILE"

echo "Creating patient directories..."

for pid in "${PIDS[@]}"; do
    # Trim whitespace
    pid=$(echo "$pid" | tr -d '[:space:]')

    if [ -n "$pid" ]; then
        "$SCRIPT_DIR/add_patient.sh" "$pid"
    fi
done

echo "Done. Total PIDs processed: ${#PIDS[@]}"
