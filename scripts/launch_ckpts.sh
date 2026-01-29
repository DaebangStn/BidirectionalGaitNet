#!/bin/bash

# Define the directory containing your renamed checkpoints
CHECKPOINT_DIR="./runs"
# CHECKPOINT_DIR="./ray_results"
FILTER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-f|--filter PATTERN] [-h|--help]"
            echo "Options:"
            echo "  -f, --filter PATTERN   Only launch checkpoints containing PATTERN in filename"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     Launch all checkpoints"
            echo "  $0 -f b20             Launch only checkpoints containing 'b20'"
            echo "  $0 --filter df_dn1    Launch only checkpoints containing 'df_dn1'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Kill all previous MuscleSim viewer instances
echo "Killing previous MuscleSim viewer instances..."
pkill -f "MuscleSim" 2>/dev/null || true
wmctrl -c "MuscleSim" 2>/dev/null || true
# Also kill by executable name as fallback
pkill -f "./build/release/viewer/render_ckpt" 2>/dev/null || true
sleep 0.5

if [ -n "$FILTER" ]; then
    echo "Starting render_ckpt processes for checkpoints containing '$FILTER' in '${CHECKPOINT_DIR}/'..."
else
    echo "Starting render_ckpt processes for all checkpoints in '${CHECKPOINT_DIR}/'..."
fi
echo "Each process will run in the background. "
echo "---------------------------------------------------------"

# Loop through all items in the specified checkpoint directory.
# We check if each item is a directory (RLlib checkpoint) before attempting to render it.
for CKPT_PATH in "${CHECKPOINT_DIR}"/*; do
    # Check if the current item is a directory.
    # RLlib checkpoints are stored as directories containing algorithm_state.pkl and policies/
    if [ -d "${CKPT_PATH}" ]; then
        # Extract just the directory name for filtering
        FILENAME=$(basename "${CKPT_PATH}")

        # Apply filter if specified
        if [ -n "$FILTER" ]; then
            # Check if the directory name contains the filter pattern
            if [[ "$FILENAME" != *"$FILTER"* ]]; then
                echo "Skipping: ${FILENAME} (doesn't match filter '$FILTER')"
                continue
            fi
        fi

        echo "Launching: scripts/render_ckpt \"${CKPT_PATH}\""
        # Execute the 'scripts/render_ckpt' command with the checkpoint relative path.
        # The '&' symbol at the end sends the command to the background,
        # allowing the script to continue without waiting for render_ckpt to finish.
        scripts/render_ckpt "${CKPT_PATH}" &
    fi
done

echo "---------------------------------------------------------"
echo "All detected checkpoint render_ckpt jobs have been launched in the background."
echo "You can use the 'jobs' command to list background jobs in your current shell session."
echo "For a more comprehensive view of running processes, use 'ps aux | grep render_ckpt'."
echo "Output from viewer might appear in your terminal or be redirected to log files."
