#!/usr/bin/env python3
"""
Cleanup script for old distillation checkpoint system.
Migrates old checkpoints to new experiment structure and removes redundant files.
"""

import os
import shutil
import pickle
from pathlib import Path
from typing import List, Dict
import argparse

def analyze_checkpoint_usage(distillation_dir: str) -> Dict:
    """Analyze existing checkpoint storage usage."""
    stats = {
        'total_files': 0,
        'total_size_mb': 0,
        'experiments': {}
    }
    
    if not os.path.exists(distillation_dir):
        print(f"Distillation directory {distillation_dir} does not exist")
        return stats
    
    for exp_name in os.listdir(distillation_dir):
        exp_path = os.path.join(distillation_dir, exp_name)
        if os.path.isdir(exp_path):
            files = [f for f in os.listdir(exp_path) if not f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(exp_path, f)) for f in files)
            
            stats['experiments'][exp_name] = {
                'files': len(files),
                'size_mb': total_size / (1024 * 1024),
                'files_list': files
            }
            stats['total_files'] += len(files)
            stats['total_size_mb'] += total_size / (1024 * 1024)
    
    return stats

def migrate_checkpoints(old_dir: str, new_experiments_dir: str, keep_every_n: int = 10) -> None:
    """Migrate old checkpoints to new experiment structure with selective retention."""
    
    old_path = Path(old_dir)
    new_path = Path(new_experiments_dir)
    new_path.mkdir(parents=True, exist_ok=True)
    
    if not old_path.exists():
        print(f"Old directory {old_dir} does not exist")
        return
    
    for exp_name in old_path.iterdir():
        if not exp_name.is_dir():
            continue
            
        print(f"\\nMigrating experiment: {exp_name.name}")
        
        # Get all checkpoint files
        checkpoint_files = []
        for f in exp_name.iterdir():
            if f.is_file() and not f.name.endswith('.json'):
                try:
                    # Extract iteration number from filename
                    iter_num = int(f.name.split('_')[-1])
                    checkpoint_files.append((iter_num, f))
                except:
                    continue
        
        checkpoint_files.sort(key=lambda x: x[0])  # Sort by iteration
        
        if not checkpoint_files:
            print(f"  No valid checkpoints found")
            continue
            
        print(f"  Found {len(checkpoint_files)} checkpoints")
        
        # Create migration directory
        migration_dir = new_path / f"migrated_{exp_name.name}"
        migration_dir.mkdir(exist_ok=True)
        
        # Keep every nth checkpoint + first and last
        checkpoints_to_keep = []
        
        # Always keep first and last
        checkpoints_to_keep.append(checkpoint_files[0])
        if len(checkpoint_files) > 1:
            checkpoints_to_keep.append(checkpoint_files[-1])
        
        # Keep every nth checkpoint
        for i in range(0, len(checkpoint_files), keep_every_n):
            if checkpoint_files[i] not in checkpoints_to_keep:
                checkpoints_to_keep.append(checkpoint_files[i])
        
        # Remove duplicates and sort
        checkpoints_to_keep = list(set(checkpoints_to_keep))
        checkpoints_to_keep.sort(key=lambda x: x[0])
        
        print(f"  Keeping {len(checkpoints_to_keep)} checkpoints (every {keep_every_n}th + first/last)")
        
        # Copy selected checkpoints
        kept_size = 0
        for iter_num, checkpoint_path in checkpoints_to_keep:
            new_name = f"migrated_{checkpoint_path.name}"
            new_checkpoint_path = migration_dir / new_name
            shutil.copy2(checkpoint_path, new_checkpoint_path)
            kept_size += checkpoint_path.stat().st_size
            
        total_size = sum(f[1].stat().st_size for f in checkpoint_files)
        print(f"  Size reduction: {total_size/(1024*1024):.1f}MB → {kept_size/(1024*1024):.1f}MB "
              f"({100*kept_size/total_size:.1f}% retained)")

def cleanup_old_structure(distillation_dir: str, confirm: bool = False) -> None:
    """Remove old distillation directory structure."""
    
    if not os.path.exists(distillation_dir):
        print(f"Directory {distillation_dir} does not exist")
        return
        
    if not confirm:
        print(f"\\n⚠️  This will DELETE the entire {distillation_dir} directory!")
        print("Use --confirm to actually perform deletion")
        return
        
    print(f"Removing {distillation_dir}...")
    shutil.rmtree(distillation_dir)
    print("✅ Old checkpoint structure removed")

def main():
    parser = argparse.ArgumentParser(description="Cleanup old checkpoint system")
    parser.add_argument("--analyze", action="store_true", help="Analyze current checkpoint usage")
    parser.add_argument("--migrate", action="store_true", help="Migrate old checkpoints to new structure")
    parser.add_argument("--cleanup", action="store_true", help="Remove old distillation directory")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    parser.add_argument("--keep-every", type=int, default=10, help="Keep every Nth checkpoint during migration")
    parser.add_argument("--old-dir", default="distillation", help="Old distillation directory")
    parser.add_argument("--new-dir", default="experiments", help="New experiments directory")
    
    args = parser.parse_args()
    
    if args.analyze:
        print("📊 Analyzing checkpoint usage...")
        stats = analyze_checkpoint_usage(args.old_dir)
        
        print(f"\\nTotal files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print("\\nPer experiment:")
        
        for exp_name, exp_stats in stats['experiments'].items():
            print(f"  {exp_name}: {exp_stats['files']} files, {exp_stats['size_mb']:.1f} MB")
    
    if args.migrate:
        print("🔄 Migrating checkpoints...")
        migrate_checkpoints(args.old_dir, args.new_dir, args.keep_every)
        
    if args.cleanup:
        cleanup_old_structure(args.old_dir, args.confirm)
        
    if not (args.analyze or args.migrate or args.cleanup):
        print("Use --analyze, --migrate, or --cleanup")
        print("Example: python cleanup_old_checkpoints.py --analyze --migrate --keep-every 5")

if __name__ == "__main__":
    main()