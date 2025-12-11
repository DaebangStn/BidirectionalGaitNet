# Sample rollout module
# Contains pysamplerollout.so after build

try:
    from .pysamplerollout import RolloutSampleEnv, RecordConfig
except ImportError as e:
    import sys
    print(f"Warning: Could not import pysamplerollout: {e}", file=sys.stderr)
    print("Make sure you have built the project with 'ninja -C build/release'", file=sys.stderr)
    RolloutSampleEnv = None
    RecordConfig = None

__all__ = ['RolloutSampleEnv', 'RecordConfig']
