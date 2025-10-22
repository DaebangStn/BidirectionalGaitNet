#!/usr/bin/env python3
"""
Example usage of the Optimizer class.

This script demonstrates how to use the optimizer to find optimal gait parameters
that minimize metabolic cost.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from python.learn.optimizer import Optimizer


def main():
    # Path to trained checkpoint (use the latest v07 checkpoint)
    ckpt_path = "sampled/base_anchor_all-026000-1014_092619+metabolic+on_20251023_060352/v07/checkpoints/ep_010000.ckpt"

    print("="*80)
    print("EXAMPLE 1: Minimize Metabolic Cost of Transport")
    print("="*80)

    # Create optimizer (minimization mode)
    opt = Optimizer(ckpt_path, maximize=False, trials=20)

    # Run optimization
    best_params, best_value = opt.run('metabolic/cot/MA/mean')

    print("\n" + "="*80)
    print("RESULT:")
    print(f"  Minimum metabolic cost: {best_value:.4f}")
    print(f"  Optimal gait parameters:")
    print(f"    Cadence: {best_params['gait_cadence']:.4f} Hz")
    print(f"    Stride:  {best_params['gait_stride']:.4f} m")
    print("="*80)

    print("\n" + "="*80)
    print("EXAMPLE 2: Optimize with Fixed Cadence")
    print("="*80)

    # Fix cadence to 1.0 Hz, optimize stride
    fixed_cadence = 1.0
    print(f"\nFixing cadence to {fixed_cadence} Hz")

    best_params, best_value = opt.run('metabolic/cot/MA/mean',
                                     fixed_fields={'gait_cadence': fixed_cadence})

    print("\n" + "="*80)
    print("RESULT:")
    print(f"  Minimum metabolic cost: {best_value:.4f}")
    print(f"  Optimal parameters:")
    print(f"    Cadence: {best_params['gait_cadence']:.4f} Hz (fixed)")
    print(f"    Stride:  {best_params['gait_stride']:.4f} m (optimized)")
    print("="*80)

    print("\n" + "="*80)
    print("EXAMPLE 3: Find Maximum (Worst) Metabolic Cost")
    print("="*80)

    # Create optimizer in maximize mode
    opt_max = Optimizer(ckpt_path, maximize=True, trials=20)

    best_params, best_value = opt_max.run('metabolic/cot/MA/mean')

    print("\n" + "="*80)
    print("RESULT:")
    print(f"  Maximum metabolic cost: {best_value:.4f}")
    print(f"  Parameters at maximum:")
    print(f"    Cadence: {best_params['gait_cadence']:.4f} Hz")
    print(f"    Stride:  {best_params['gait_stride']:.4f} m")
    print("="*80)


if __name__ == '__main__':
    main()
