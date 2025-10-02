#!/bin/bash

# Run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root (sudo)."
    exit 1
fi

# Array of nodes
nodes=(n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15)

# Loop through each node
for node in "${nodes[@]}"; do
    echo "Connecting to $node..."
    ssh "$node" "rm -rf /tmp/_ray_lockfiles" && echo "Cleaned lockfiles on $node" || echo "Failed to clean lockfiles on $node"
done

echo "Completed lockfile cleanup on all nodes."
