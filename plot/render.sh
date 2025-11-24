#!/bin/bash
# Render all DOT files to PNG (300 DPI) and PDF

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOT_DIR="$SCRIPT_DIR/dot"
DIAGRAM_DIR="$SCRIPT_DIR/diagram"

# Create output directory if it doesn't exist
mkdir -p "$DIAGRAM_DIR"

# Check if dot command is available
if ! command -v dot &> /dev/null; then
    echo "Error: graphviz 'dot' command not found"
    echo "Install with: sudo apt-get install graphviz"
    exit 1
fi

# Render all DOT files
echo "Rendering DOT files..."
for f in "$DOT_DIR"/*.dot; do
    if [ -f "$f" ]; then
        base=$(basename "$f" .dot)

        # Render PNG at 300 DPI
        dot -Tpng -Gdpi=300 "$f" -o "$DIAGRAM_DIR/${base}_clean.png"

        # Get file sizes
        png_size=$(du -h "$DIAGRAM_DIR/${base}_clean.png" | cut -f1)

        echo "✓ Rendered $base (PNG: $png_size)"
    fi
done

echo ""
echo "All diagrams rendered successfully!"
echo "Output directory: $DIAGRAM_DIR"
