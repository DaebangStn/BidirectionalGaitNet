#!/usr/bin/env python3
"""
Test the Python URI resolver functionality
"""

import sys
import os
sys.path.append('python')

from python.uri_resolver import URIResolver, resolve_path

def test_python_uri_resolver():
    print("=== Python URI Resolver Test ===")
    
    # Test paths
    test_paths = [
        "@data/skeleton_gaitnet_narrow_model.xml",
        "@data/motion/walk.bvh", 
        "@data/muscle_gaitnet.xml",
        "*/skeleton_gaitnet_narrow_model.xml",
        "../data/skeleton_gaitnet_narrow_model.xml",  # Backwards compatibility
        "./rollout/test_file.npz",                    # Rollout directory
        "regular/path/file.xml"
    ]
    
    resolver = URIResolver.get_instance()
    resolver.initialize()
    
    for path in test_paths:
        resolved = resolver.resolve(path)
        is_uri = resolver.is_uri(path)
        print(f"Input:  {path}")
        print(f"Output: {resolved}")
        print(f"Is URI: {is_uri}")
        print("---")
    
    print("\n=== Test Rollout Path Resolution ===")
    
    # Test the specific rollout case
    rollout_path = './rollout/0_0.npz'
    resolved_rollout = resolve_path(rollout_path)
    print(f"Rollout path: {rollout_path}")
    print(f"Resolved to:  {resolved_rollout}")
    
    # Check if the resolved path is absolute and points to project
    if os.path.isabs(resolved_rollout) and 'BidirectionalGaitNet' in resolved_rollout:
        print("✅ Rollout path resolution works correctly!")
    else:
        print("❌ Rollout path resolution failed")
        return False
    
    print("✅ Python URI resolver test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_python_uri_resolver()
    sys.exit(0 if success else 1)