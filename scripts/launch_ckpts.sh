#!/bin/bash

# Define the directory containing your renamed checkpoints
CHECKPOINT_DIR="./ray_results"
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

if [ -n "$FILTER" ]; then
    echo "Starting viewer processes for checkpoints containing '$FILTER' in '${CHECKPOINT_DIR}/'..."
else
    echo "Starting viewer processes for all checkpoints in '${CHECKPOINT_DIR}/'..."
fi
echo "Each process will run in the background. "
echo "---------------------------------------------------------"

# Loop through all items in the specified checkpoint directory.
# We check if each item is a regular file before attempting to render it.
for CKPT_PATH in "${CHECKPOINT_DIR}"/*; do
    # Check if the current item is a regular file.
    # This prevents trying to render subdirectories or other non-file entries.
    if [ -f "${CKPT_PATH}" ]; then
        # Extract just the filename for filtering
        FILENAME=$(basename "${CKPT_PATH}")
        
        # Apply filter if specified
        if [ -n "$FILTER" ]; then
            # Check if the filename contains the filter pattern
            if [[ "$FILENAME" != *"$FILTER"* ]]; then
                echo "Skipping: ${FILENAME} (doesn't match filter '$FILTER')"
                continue
            fi
        fi
        
        echo "Launching: scripts/viewer \"${CKPT_PATH}\""
        # Execute the 'scripts/viewer' command with the checkpoint relative path.
        # The '&' symbol at the end sends the command to the background,
        # allowing the script to continue without waiting for the viewer to finish.
        scripts/viewer "${CKPT_PATH}" &
    fi
done

echo "---------------------------------------------------------"
echo "All detected checkpoint viewer jobs have been launched in the background."
echo "You can use the 'jobs' command to list background jobs in your current shell session."
echo "For a more comprehensive view of running processes, use 'ps aux | grep viewer'."
echo "Output from viewer might appear in your terminal or be redirected to log files."
