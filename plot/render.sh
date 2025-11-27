#!/bin/bash
# Incremental DOT Renderer with Hash-Based Caching and Parallel Execution
#
# Features:
#   - SHA-256 content-based change detection (not timestamp)
#   - Parallel rendering with configurable concurrency
#   - Atomic cache updates (only after successful render)
#   - Clear per-file feedback and summary statistics
#
# Usage:
#   ./render.sh              # Incremental render
#   ./render.sh --force      # Force rebuild all
#   ./render.sh --clean      # Delete cache and exit
#   ./render.sh --jobs 4     # Limit to 4 parallel jobs
#   ./render.sh --dry-run    # Show what would be rendered
#   ./render.sh --verbose    # Show detailed timing

set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOT_DIR="$SCRIPT_DIR/dot"
DIAGRAM_DIR="$SCRIPT_DIR/diagram"
CACHE_DIR="$SCRIPT_DIR/.rendercache"
MANIFEST="$CACHE_DIR/manifest.txt"

# CLI options
FORCE=false
CLEAN=false
DRY_RUN=false
VERBOSE=false
JOBS="${RENDER_JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Statistics
TOTAL=0
RENDERED=0
SKIPPED=0
FAILED=0

# ─────────────────────────────────────────────────────────────────────────────
# CLI Parsing
# ─────────────────────────────────────────────────────────────────────────────

print_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Render DOT files to PNG with incremental caching and parallel execution.

Options:
  -f, --force       Force rebuild all files (ignore cache)
  -c, --clean       Delete cache directory and exit
  -j, --jobs N      Set parallel job count (default: $JOBS)
  -n, --dry-run     Show what would be rendered without doing it
  -v, --verbose     Show detailed timing per file
  -h, --help        Show this help message

Environment:
  RENDER_JOBS       Default parallel job count (overridden by --jobs)

Examples:
  ./render.sh                 # Normal incremental render
  ./render.sh --force         # Rebuild everything
  ./render.sh --jobs 2        # Limit parallelism
  RENDER_JOBS=8 ./render.sh   # Use 8 parallel jobs
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--force)
                FORCE=true
                shift
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                print_help >&2
                exit 1
                ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Dependency Checks
# ─────────────────────────────────────────────────────────────────────────────

check_dependencies() {
    if ! command -v dot &>/dev/null; then
        echo "Error: graphviz 'dot' command not found" >&2
        echo "Install with: sudo apt-get install graphviz" >&2
        exit 1
    fi

    # Determine hash command (sha256sum on Linux, shasum on macOS)
    if command -v sha256sum &>/dev/null; then
        HASH_CMD="sha256sum"
    elif command -v shasum &>/dev/null; then
        HASH_CMD="shasum -a 256"
    else
        echo "Error: Neither sha256sum nor shasum found" >&2
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Cache Functions
# ─────────────────────────────────────────────────────────────────────────────

init_cache() {
    mkdir -p "$CACHE_DIR"
    touch "$MANIFEST"
}

clean_cache() {
    if [[ -d "$CACHE_DIR" ]]; then
        rm -rf "$CACHE_DIR"
        echo "Cache cleaned: $CACHE_DIR"
    else
        echo "No cache to clean"
    fi
}

# Get cached hash for a file (empty string if not found)
get_cached_hash() {
    local filename="$1"
    grep -E "^[a-f0-9]+  $filename$" "$MANIFEST" 2>/dev/null | cut -d' ' -f1 || true
}

# Update cached hash atomically (with flock for parallel safety)
set_cached_hash() {
    local filename="$1"
    local hash="$2"

    {
        flock -x 200
        local tmp="$MANIFEST.tmp.$$"
        grep -v -E "^[a-f0-9]+  $filename$" "$MANIFEST" 2>/dev/null > "$tmp" || true
        echo "$hash  $filename" >> "$tmp"
        mv "$tmp" "$MANIFEST"
    } 200>"$MANIFEST.lock"
}

# ─────────────────────────────────────────────────────────────────────────────
# Hash Functions
# ─────────────────────────────────────────────────────────────────────────────

compute_hash() {
    local file="$1"
    $HASH_CMD "$file" | cut -d' ' -f1
}

# ─────────────────────────────────────────────────────────────────────────────
# Render Functions
# ─────────────────────────────────────────────────────────────────────────────

# Determine if a file needs rendering
# Returns: "render:reason" or "skip:reason"
should_render() {
    local dotfile="$1"
    local filename
    filename=$(basename "$dotfile")
    local base="${filename%.dot}"
    local output="$DIAGRAM_DIR/${base}_clean.png"

    # Force mode: always render
    if [[ "$FORCE" == true ]]; then
        echo "render:forced"
        return
    fi

    # Output missing: must render
    if [[ ! -f "$output" ]]; then
        echo "render:output missing"
        return
    fi

    # Compare hashes
    local current_hash cached_hash
    current_hash=$(compute_hash "$dotfile")
    cached_hash=$(get_cached_hash "$filename")

    if [[ "$current_hash" != "$cached_hash" ]]; then
        echo "render:content changed"
    else
        echo "skip:unchanged"
    fi
}

# Render a single DOT file and print result
# Returns 0 on success, 1 on failure
render_single() {
    local dotfile="$1"
    local reason="$2"
    local filename
    filename=$(basename "$dotfile")
    local base="${filename%.dot}"
    local output="$DIAGRAM_DIR/${base}_clean.png"
    local start_time
    start_time=$(date +%s.%N)

    # Attempt render
    local error_output=""
    if error_output=$(dot -Tpng -Gdpi=300 "$dotfile" -o "$output" 2>&1); then
        local end_time duration size
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        size=$(du -h "$output" | cut -f1)

        # Update cache (only on success)
        local hash
        hash=$(compute_hash "$dotfile")
        set_cached_hash "$filename" "$hash"

        if [[ "$VERBOSE" == true ]]; then
            printf "✓ %s (rendered, %.2fs, %s, %s)\n" "$filename" "$duration" "$size" "$reason"
        else
            echo "✓ $filename (rendered, ${duration}s, $size)"
        fi
        return 0
    else
        local error
        error=$(echo "$error_output" | head -1)
        echo "✗ $filename (failed: $error)"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local start_time
    start_time=$(date +%s.%N)

    parse_args "$@"
    check_dependencies

    # Handle --clean
    if [[ "$CLEAN" == true ]]; then
        clean_cache
        exit 0
    fi

    # Setup
    mkdir -p "$DIAGRAM_DIR"
    init_cache

    # Discover DOT files
    local -a dotfiles=()
    if [[ -d "$DOT_DIR" ]]; then
        while IFS= read -r -d '' f; do
            dotfiles+=("$f")
        done < <(find "$DOT_DIR" -maxdepth 1 -name "*.dot" -print0 | sort -z)
    fi

    TOTAL=${#dotfiles[@]}

    if [[ $TOTAL -eq 0 ]]; then
        echo "No DOT files found in $DOT_DIR"
        exit 0
    fi

    # Cap jobs at file count
    [[ "$JOBS" -gt "$TOTAL" ]] && JOBS="$TOTAL"

    # Determine what needs rendering
    local -a to_render=()
    local -a to_skip=()
    declare -A render_reasons

    for dotfile in "${dotfiles[@]}"; do
        local result
        result=$(should_render "$dotfile")
        local action="${result%%:*}"
        local reason="${result#*:}"
        local filename
        filename=$(basename "$dotfile")

        if [[ "$action" == "render" ]]; then
            to_render+=("$dotfile")
            render_reasons["$dotfile"]="$reason"
        else
            to_skip+=("$filename")
        fi
    done

    # Dry-run mode
    if [[ "$DRY_RUN" == true ]]; then
        echo "Dry run - would render ${#to_render[@]} of $TOTAL files:"
        for dotfile in "${to_render[@]}"; do
            local filename
            filename=$(basename "$dotfile")
            echo "  ✓ $filename (${render_reasons[$dotfile]})"
        done
        for filename in "${to_skip[@]}"; do
            echo "  · $filename (skipped)"
        done
        exit 0
    fi

    # Fast path: all files up-to-date
    if [[ ${#to_render[@]} -eq 0 ]]; then
        local end_time elapsed
        end_time=$(date +%s.%N)
        elapsed=$(printf "%.2f" "$(echo "$end_time - $start_time" | bc)")
        echo "All $TOTAL files up-to-date (${elapsed}s)"
        exit 0
    fi

    echo "Rendering DOT files (${#to_render[@]} of $TOTAL, $JOBS parallel jobs)..."

    # Print skipped files first
    for filename in "${to_skip[@]}"; do
        echo "· $filename (skipped, unchanged)"
        ((SKIPPED++))
    done

    # Render files with parallel execution using background jobs
    local -a pids=()
    local -a job_files=()
    local running=0

    for dotfile in "${to_render[@]}"; do
        local reason="${render_reasons[$dotfile]}"

        # Wait if we're at max concurrency
        while [[ $running -ge $JOBS ]]; do
            # Wait for any job to finish
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}" 2>/dev/null || true
                    unset 'pids[i]'
                    ((running--)) || true
                    break
                fi
            done
            # Compact array
            pids=("${pids[@]}")
            sleep 0.05
        done

        # Launch background job
        render_single "$dotfile" "$reason" &
        pids+=($!)
        job_files+=("$dotfile")
        ((running++))
    done

    # Wait for all remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            ((RENDERED++))
        else
            ((FAILED++))
        fi
    done

    # Summary
    local end_time elapsed
    end_time=$(date +%s.%N)
    elapsed=$(printf "%.2f" "$(echo "$end_time - $start_time" | bc)")

    echo ""
    echo "Summary: $TOTAL files, $RENDERED rendered, $SKIPPED skipped, $FAILED failed (${elapsed}s)"
    echo "Output directory: $DIAGRAM_DIR"

    # Exit with error if any failures
    [[ $FAILED -gt 0 ]] && exit 1
    exit 0
}

main "$@"
