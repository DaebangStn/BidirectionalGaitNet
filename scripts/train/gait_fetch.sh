#!/bin/bash
#
# Prefetch workflow for cluster training
# Usage: ./scripts/train/gait_fetch.sh <config_name>
# Example: ./scripts/train/gait_fetch.sh 12964246_refm
#
# This script:
# 1. Waits for cache lock to be released
# 2. Clears the rm cache
# 3. Prefetches @pid resources on login node
# 4. Submits sbatch job with --no-clear-cache flag
#

set -e

CONFIG_NAME="${1:?Usage: $0 <config_name>}"
ENV_FILE="data/env/${CONFIG_NAME}.yaml"
LOCK_FILE=".tmp/rm_cache.lock"
CACHE_DIR=".tmp/rm_cache"

# Verify config file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Config file not found: $ENV_FILE"
    exit 1
fi

# 1. Wait for lock to be released (another job may be using cache)
WAITED=0
while [ -f "$LOCK_FILE" ]; do
    echo "Cache locked, waiting... (${WAITED}s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

# 2. Clear cache
rm -rf "$CACHE_DIR"
echo "Cleared rm cache"

# 3. Run prefetcher on login node (creates lock file)
echo "Prefetching resources..."
python rm/python/prefetch.py "$ENV_FILE"

# Check if prefetch succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Prefetch failed"
    exit 1
fi

# 4. Submit job with --no-clear-cache
sbatch --job-name="${CONFIG_NAME}" \
    --export=ALL,ENV_FILE="$ENV_FILE",NO_CLEAR_CACHE=1 \
    ./scripts/train/gait.sh

echo "Submitted job: ${CONFIG_NAME}"
echo "Cache locked until job loads data"
