#!/usr/bin/env python3
"""Minimal test to check if pyrollout works with Ray workers"""

import ray
import sys

@ray.remote
def test_pyrollout_import():
    """Test if pyrollout can be imported in a Ray worker"""
    try:
        from pyrollout import RolloutEnvironment, RolloutRecord
        return "SUCCESS: pyrollout imported in Ray worker"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

if __name__ == "__main__":
    print("Testing pyrollout with Ray...")

    # Test local import first
    try:
        from pyrollout import RolloutEnvironment, RolloutRecord
        print("✓ Local import successful")
    except Exception as e:
        print(f"✗ Local import failed: {e}")
        sys.exit(1)

    # Initialize Ray
    ray.init()

    # Test remote import
    result = ray.get(test_pyrollout_import.remote())
    print(result)

    ray.shutdown()
