#!/usr/bin/env python3
"""
Prefetch @pid resources from env config YAML.
Run on login node before submitting SLURM job.
Creates lock file to protect cache during job execution.
"""
import sys
import yaml
from pathlib import Path

# Project root (resolve to absolute path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add pyrm to path (built into source directory)
sys.path.insert(0, str(PROJECT_ROOT / "rm/python"))

try:
    import pyrm
except ImportError as e:
    print(f"ERROR: Failed to import pyrm module from {PROJECT_ROOT / 'rm/python'}")
    print(f"  {e}")
    print(f"\nPlease build the project: ninja -C build/release")
    sys.exit(1)

CONFIG_PATH = str(PROJECT_ROOT / "data/rm_config.yaml")
CACHE_DIR = PROJECT_ROOT / ".tmp" / "rm_cache"
LOCK_FILE = PROJECT_ROOT / ".tmp" / "rm_cache.lock"


def expand_pid_uri(uri: str, default_pid: str) -> str:
    """Expand @pid:/path to @pid:{default_pid}/path if default_pid is set."""
    if default_pid and uri.startswith("@pid:/"):
        return f"@pid:{default_pid}/" + uri[6:]
    return uri


def extract_pid_uris(config: dict, default_pid: str = None) -> list[str]:
    """Recursively extract all @pid: URIs from config dict.

    If default_pid is provided, expands @pid:/path to @pid:{default_pid}/path.
    """
    uris = []

    def walk(obj):
        if isinstance(obj, str):
            if obj.startswith("@pid:"):
                expanded = expand_pid_uri(obj, default_pid)
                uris.append(expanded)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(config)
    return uris


def prefetch(env_file: str):
    """Prefetch all @pid resources from env config."""
    print(f"Prefetching resources for: {env_file}")

    # Load env config
    with open(env_file) as f:
        config = yaml.safe_load(f)

    # Get global pid for expanding @pid:/ URIs
    # Try root level first, then environment.pid
    default_pid = config.get("pid")
    if not default_pid and "environment" in config:
        default_pid = config["environment"].get("pid")
    if default_pid:
        print(f"Using global pid: {default_pid}")

    # Extract @pid URIs (with expansion if default_pid is set)
    uris = extract_pid_uris(config, default_pid)
    print(f"Found {len(uris)} @pid URIs to prefetch")

    if not uris:
        print("No @pid URIs found, nothing to prefetch")
        return 0

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create lock file (right after directory is created)
    LOCK_FILE.touch()
    print(f"Created lock file: {LOCK_FILE}")

    # Initialize resource manager
    rm = pyrm.ResourceManager(CONFIG_PATH)

    # Fetch each URI (triggers caching)
    success = 0
    for uri in uris:
        try:
            print(f"  Fetching: {uri} ... ", end="", flush=True)
            handle = rm.fetch(uri)
            print(f"OK ({handle.size()} bytes)")
            success += 1
        except pyrm.RMError as e:
            print(f"FAILED: {e}")

    print(f"\nPrefetched {success}/{len(uris)} resources")

    if success != len(uris):
        # On failure, remove lock to allow retry
        LOCK_FILE.unlink(missing_ok=True)
        print("Removed lock file due to prefetch failure")
        return 1

    print(f"Cache locked. Will be released after job loads data.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <env_file.yaml>")
        sys.exit(1)
    sys.exit(prefetch(sys.argv[1]))
