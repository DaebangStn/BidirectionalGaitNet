"""CleanRL PPO implementation with hierarchical muscle control."""

import sys
import os

# Add ppo directory to path for gymenv.so
_ppo_dir = os.path.dirname(os.path.abspath(__file__))
if _ppo_dir not in sys.path:
    sys.path.insert(0, _ppo_dir)

# Import C++ modules
try:
    import gymenv as _gymenv
    GymEnvManager = _gymenv.GymEnvManager
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import gymenv C++ module. "
        f"Make sure the project is built with 'ninja'. "
        f"Error: {e}"
    )
    _gymenv = None
    GymEnvManager = None

try:
    import batchenv as _batchenv
    BatchEnv = _batchenv.BatchEnv
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import batchenv C++ module. "
        f"Make sure the project is built with 'ninja'. "
        f"Error: {e}"
    )
    _batchenv = None
    BatchEnv = None

__version__ = "0.1.0"
__all__ = ['GymEnvManager', 'BatchEnv']
