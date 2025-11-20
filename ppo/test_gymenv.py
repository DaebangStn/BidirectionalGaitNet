#!/usr/bin/env python
"""Test script for gymenv C++ binding"""

import sys
import os

# Add parent directory to path so we can import ppo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import GymEnvManager
import numpy as np

def test_import():
    """Test that gymenv can be imported"""
    print("✓ Import successful")
    return True

def test_create_env():
    """Test environment creation"""
    with open('data/env/base_lonly.xml') as f:
        env_content = f.read()

    env = GymEnvManager(env_content)
    print("✓ Environment created")
    return env

def test_reset(env):
    """Test environment reset"""
    obs, info = env.reset()
    print(f"✓ Reset successful, obs shape: {obs.shape}")
    return obs

def test_step(env):
    """Test environment step"""
    action = np.zeros(env.getNumAction(), dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful, reward: {reward:.4f}")
    return obs, reward, terminated, truncated

def test_muscle_tuples(env):
    """Test muscle tuple collection"""
    is_two_level = env.isTwoLevelController()
    use_cascading = env.getUseCascading()
    print(f"✓ Two-level: {is_two_level}, Cascading: {use_cascading}")

    if not is_two_level:
        print("⊘ Skipping muscle tests (not two-level controller)")
        return

    # Take steps to collect tuples
    action = np.zeros(env.getNumAction(), dtype=np.float32)
    for _ in range(5):
        env.step(action)

    tuples = env.get_muscle_tuples()
    expected = 5 if use_cascading else 3
    print(f"✓ Collected {len(tuples)} tuple components (expected {expected})")

    for i, t in enumerate(tuples):
        print(f"  Component {i}: type={type(t)}, len={len(t) if hasattr(t, '__len__') else 'N/A'}")
        if len(t) > 0:
            print(f"    First item type: {type(t[0])}, shape: {t[0].shape if hasattr(t[0], 'shape') else 'N/A'}")

    # Test buffer clearing
    tuples2 = env.get_muscle_tuples()
    all_empty = all(len(t) == 0 for t in tuples2)
    print(f"✓ Buffer cleared: {all_empty}")

if __name__ == "__main__":
    try:
        test_import()
        env = test_create_env()
        test_reset(env)
        test_step(env)
        test_muscle_tuples(env)
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
