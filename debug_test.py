#!/usr/bin/env python3

"""
Minimal test to reproduce the tree library import issue
"""

def test_tree_import():
    try:
        print("Testing direct tree import...")
        import tree
        print(f"✓ tree imported successfully, version: {getattr(tree, '__version__', 'unknown')}")
        return True
    except Exception as e:
        print(f"✗ tree import failed: {e}")
        return False

def test_ray_import():
    try:
        print("Testing ray.rllib import...")
        import ray.rllib.models.torch.torch_modelv2
        print("✓ ray.rllib imported successfully")
        return True
    except Exception as e:
        print(f"✗ ray.rllib import failed: {e}")
        return False

def test_python_ray_model():
    try:
        print("Testing python.ray_model import...")
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'python'))
        import python.ray_model
        print("✓ python.ray_model imported successfully")
        return True
    except Exception as e:
        print(f"✗ python.ray_model import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Python Import Test ===")
    test_tree_import()
    test_ray_import()
    test_python_ray_model()