#!/usr/bin/env python3
"""
RM Data Structure Migration Script

Migrates from nested gait-centric structure to visit-based flat structure.

Current Structure:
    {pid}/
    ├── metadata.yaml          # pid, name, gmfcs, surgery1, surgery2
    ├── rom.yaml               # pre_op, post_op
    ├── dicom/op1/, op2/
    └── gait/
        ├── metadata.yaml      # pre: {timestamp, height, weight, foot}, post: {...}
        ├── pre/
        │   ├── Generated_C3D_files/
        │   ├── h5/
        │   ├── skeleton/
        │   └── muscle/
        └── post/

Target Structure:
    {pid}/
    ├── metadata.yaml         # pid, name, gmfcs
    ├── pre/
    │   ├── gait/             # C3D files (from Gait_long)
    │   ├── motion/           # HDF5 files
    │   ├── skeleton/
    │   ├── muscle/
    │   ├── ckpt/
    │   ├── dicom/
    │   ├── rom.yaml          # pre_op rom data
    │   └── metadata.yaml     # timestamp, age, height, weight, foot
    └── op1/                  # current "post" data
        └── (same structure as pre)
"""

import os
import shutil
import yaml
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationOperation:
    """Record of a single migration operation for rollback support"""
    op_type: str  # 'mkdir', 'copy', 'move', 'write', 'remove'
    source: Optional[str] = None
    dest: Optional[str] = None
    content: Optional[str] = None  # For write operations
    backup_path: Optional[str] = None  # For overwrite operations
    status: str = 'pending'


@dataclass
class PatientSnapshot:
    """Pre-migration snapshot for verification"""
    pid: str
    c3d_pre_count: int = 0
    c3d_post_count: int = 0
    h5_pre_count: int = 0
    h5_post_count: int = 0
    skeleton_pre: List[str] = field(default_factory=list)
    skeleton_post: List[str] = field(default_factory=list)
    muscle_pre: List[str] = field(default_factory=list)
    muscle_post: List[str] = field(default_factory=list)
    dicom_op1_count: int = 0
    dicom_op2_count: int = 0
    total_size_bytes: int = 0


@dataclass
class MigrationLog:
    """Transaction log for a patient migration"""
    pid: str
    status: str = 'pending'
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    operations: List[MigrationOperation] = field(default_factory=list)
    snapshot_before: Optional[PatientSnapshot] = None
    errors: List[str] = field(default_factory=list)


class RMMigrator:
    """Handles migration of RM data structure"""

    def __init__(self,
                 rm_root: str = "/mnt/blue8T/CP/RM",
                 gait_long_root: str = "/mnt/blue8T/CP/Gait_long",
                 log_dir: str = None,
                 dry_run: bool = False):
        self.rm_root = Path(rm_root)
        self.gait_long_root = Path(gait_long_root)
        self.dry_run = dry_run

        # Setup log directory
        if log_dir is None:
            log_dir = self.rm_root / ".migration_logs"
        self.log_dir = Path(log_dir)

        if not dry_run:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_all_patients(self) -> List[str]:
        """Get list of all patient IDs"""
        patients = []
        for item in self.rm_root.iterdir():
            if item.is_dir() and item.name.isdigit():
                patients.append(item.name)
        return sorted(patients)

    def get_gait_long_path(self, name: str, gmfcs: int, timestamp: int) -> Path:
        """Build path to Gait_long source directory"""
        return self.gait_long_root / f"GMFCS{gmfcs}" / name / str(timestamp) / "Generated_C3D_files"

    def create_snapshot(self, pid: str) -> PatientSnapshot:
        """Create pre-migration snapshot of patient data"""
        snapshot = PatientSnapshot(pid=pid)
        patient_dir = self.rm_root / pid

        # Count C3D files (from Generated_C3D_files)
        c3d_pre = patient_dir / "gait" / "pre" / "Generated_C3D_files"
        c3d_post = patient_dir / "gait" / "post" / "Generated_C3D_files"

        if c3d_pre.exists():
            snapshot.c3d_pre_count = len(list(c3d_pre.glob("*.c3d")))
        if c3d_post.exists():
            snapshot.c3d_post_count = len(list(c3d_post.glob("*.c3d")))

        # Count H5 files
        h5_pre = patient_dir / "gait" / "pre" / "h5"
        h5_post = patient_dir / "gait" / "post" / "h5"

        if h5_pre.exists():
            snapshot.h5_pre_count = len(list(h5_pre.glob("**/*.h5"))) + len(list(h5_pre.glob("**/*.hdf")))
        if h5_post.exists():
            snapshot.h5_post_count = len(list(h5_post.glob("**/*.h5"))) + len(list(h5_post.glob("**/*.hdf")))

        # List skeleton files
        skel_pre = patient_dir / "gait" / "pre" / "skeleton"
        skel_post = patient_dir / "gait" / "post" / "skeleton"

        if skel_pre.exists():
            snapshot.skeleton_pre = [f.name for f in skel_pre.iterdir() if f.is_file()]
        if skel_post.exists():
            snapshot.skeleton_post = [f.name for f in skel_post.iterdir() if f.is_file()]

        # List muscle files
        muscle_pre = patient_dir / "gait" / "pre" / "muscle"
        muscle_post = patient_dir / "gait" / "post" / "muscle"

        if muscle_pre.exists():
            snapshot.muscle_pre = [f.name for f in muscle_pre.iterdir() if f.is_file()]
        if muscle_post.exists():
            snapshot.muscle_post = [f.name for f in muscle_post.iterdir() if f.is_file()]

        # Count DICOM files
        dicom_op1 = patient_dir / "dicom" / "op1"
        dicom_op2 = patient_dir / "dicom" / "op2"

        if dicom_op1.exists():
            snapshot.dicom_op1_count = sum(1 for _ in dicom_op1.rglob("*") if _.is_file())
        if dicom_op2.exists():
            snapshot.dicom_op2_count = sum(1 for _ in dicom_op2.rglob("*") if _.is_file())

        # Total size
        def get_dir_size(path: Path) -> int:
            total = 0
            if path.exists():
                for f in path.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
            return total

        snapshot.total_size_bytes = get_dir_size(patient_dir)

        return snapshot

    def load_patient_metadata(self, pid: str) -> Dict[str, Any]:
        """Load patient metadata.yaml"""
        meta_path = self.rm_root / pid / "metadata.yaml"
        with open(meta_path) as f:
            return yaml.safe_load(f)

    def load_gait_metadata(self, pid: str) -> Dict[str, Any]:
        """Load gait/metadata.yaml"""
        meta_path = self.rm_root / pid / "gait" / "metadata.yaml"
        with open(meta_path) as f:
            return yaml.safe_load(f)

    def load_rom_data(self, pid: str) -> Dict[str, Any]:
        """Load rom.yaml"""
        rom_path = self.rm_root / pid / "rom.yaml"
        with open(rom_path) as f:
            return yaml.safe_load(f)

    def _exec_mkdir(self, path: Path, log: MigrationLog) -> bool:
        """Execute mkdir operation"""
        op = MigrationOperation(op_type='mkdir', dest=str(path))
        log.operations.append(op)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] mkdir: {path}")
            op.status = 'dry_run'
            return True

        try:
            path.mkdir(parents=True, exist_ok=True)
            op.status = 'completed'
            logger.debug(f"  mkdir: {path}")
            return True
        except Exception as e:
            op.status = 'failed'
            log.errors.append(f"mkdir failed: {path} - {e}")
            return False

    def _exec_copy(self, src: Path, dest: Path, log: MigrationLog) -> bool:
        """Execute copy operation"""
        op = MigrationOperation(op_type='copy', source=str(src), dest=str(dest))
        log.operations.append(op)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] copy: {src} -> {dest}")
            op.status = 'dry_run'
            return True

        try:
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            op.status = 'completed'
            logger.debug(f"  copy: {src} -> {dest}")
            return True
        except Exception as e:
            op.status = 'failed'
            log.errors.append(f"copy failed: {src} -> {dest} - {e}")
            return False

    def _exec_move(self, src: Path, dest: Path, log: MigrationLog) -> bool:
        """Execute move operation"""
        op = MigrationOperation(op_type='move', source=str(src), dest=str(dest))
        log.operations.append(op)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] move: {src} -> {dest}")
            op.status = 'dry_run'
            return True

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            op.status = 'completed'
            logger.debug(f"  move: {src} -> {dest}")
            return True
        except Exception as e:
            op.status = 'failed'
            log.errors.append(f"move failed: {src} -> {dest} - {e}")
            return False

    def _exec_write(self, path: Path, content: str, log: MigrationLog) -> bool:
        """Execute write operation"""
        # Backup existing file if present
        backup_path = None
        if path.exists() and not self.dry_run:
            backup_path = str(path) + ".bak"
            shutil.copy2(path, backup_path)

        op = MigrationOperation(
            op_type='write',
            dest=str(path),
            content=content,
            backup_path=backup_path
        )
        log.operations.append(op)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] write: {path}")
            op.status = 'dry_run'
            return True

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            op.status = 'completed'
            logger.debug(f"  write: {path}")
            return True
        except Exception as e:
            op.status = 'failed'
            log.errors.append(f"write failed: {path} - {e}")
            return False

    def _exec_remove(self, path: Path, log: MigrationLog) -> bool:
        """Execute remove operation"""
        op = MigrationOperation(op_type='remove', source=str(path))
        log.operations.append(op)

        if self.dry_run:
            logger.info(f"  [DRY-RUN] remove: {path}")
            op.status = 'dry_run'
            return True

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            op.status = 'completed'
            logger.debug(f"  remove: {path}")
            return True
        except Exception as e:
            op.status = 'failed'
            log.errors.append(f"remove failed: {path} - {e}")
            return False

    def migrate_patient(self, pid: str) -> MigrationLog:
        """Migrate a single patient's data structure"""
        log = MigrationLog(pid=pid, started_at=datetime.now().isoformat())
        patient_dir = self.rm_root / pid

        logger.info(f"Migrating patient {pid}...")

        # Check if already migrated
        if (patient_dir / "pre").exists() and not (patient_dir / "gait").exists():
            logger.info(f"  Patient {pid} already migrated, skipping")
            log.status = 'skipped'
            return log

        # Create snapshot
        log.snapshot_before = self.create_snapshot(pid)
        log.status = 'in_progress'

        try:
            # Load metadata
            patient_meta = self.load_patient_metadata(pid)
            gait_meta = self.load_gait_metadata(pid)
            rom_data = self.load_rom_data(pid)

            name = patient_meta.get('name')
            gmfcs = patient_meta.get('gmfcs')

            # === Step 1: Create directory structure ===
            logger.info(f"  Creating directory structure...")

            for visit in ['pre', 'op1']:
                for subdir in ['gait', 'motion', 'skeleton', 'muscle', 'ckpt', 'dicom']:
                    if not self._exec_mkdir(patient_dir / visit / subdir, log):
                        raise RuntimeError(f"Failed to create {visit}/{subdir}")

            # === Step 2: Copy C3D files from Gait_long ===
            logger.info(f"  Copying C3D files from Gait_long...")

            ts_pre = gait_meta.get('pre', {}).get('timestamp')
            ts_post = gait_meta.get('post', {}).get('timestamp')

            if ts_pre and name and gmfcs:
                gait_long_pre = self.get_gait_long_path(name, gmfcs, ts_pre)
                if gait_long_pre.exists():
                    for c3d in gait_long_pre.glob("*.c3d"):
                        dest = patient_dir / "pre" / "gait" / c3d.name
                        if not self._exec_copy(c3d, dest, log):
                            logger.warning(f"  Failed to copy {c3d.name}")
                else:
                    logger.warning(f"  Gait_long pre not found: {gait_long_pre}")

            if ts_post and name and gmfcs:
                gait_long_post = self.get_gait_long_path(name, gmfcs, ts_post)
                if gait_long_post.exists():
                    for c3d in gait_long_post.glob("*.c3d"):
                        dest = patient_dir / "op1" / "gait" / c3d.name
                        if not self._exec_copy(c3d, dest, log):
                            logger.warning(f"  Failed to copy {c3d.name}")
                else:
                    logger.warning(f"  Gait_long post not found: {gait_long_post}")

            # === Step 3: Move h5 -> motion ===
            logger.info(f"  Moving H5 files to motion...")

            h5_pre = patient_dir / "gait" / "pre" / "h5"
            if h5_pre.exists():
                for f in h5_pre.iterdir():
                    if f.is_file():
                        dest = patient_dir / "pre" / "motion" / f.name
                        self._exec_copy(f, dest, log)

            h5_post = patient_dir / "gait" / "post" / "h5"
            if h5_post.exists():
                for f in h5_post.iterdir():
                    if f.is_file():
                        dest = patient_dir / "op1" / "motion" / f.name
                        self._exec_copy(f, dest, log)

            # === Step 4: Move skeleton files ===
            logger.info(f"  Moving skeleton files...")

            skel_pre = patient_dir / "gait" / "pre" / "skeleton"
            if skel_pre.exists():
                for f in skel_pre.iterdir():
                    if f.is_file():
                        dest = patient_dir / "pre" / "skeleton" / f.name
                        self._exec_copy(f, dest, log)

            skel_post = patient_dir / "gait" / "post" / "skeleton"
            if skel_post.exists():
                for f in skel_post.iterdir():
                    if f.is_file():
                        dest = patient_dir / "op1" / "skeleton" / f.name
                        self._exec_copy(f, dest, log)

            # === Step 5: Move muscle files ===
            logger.info(f"  Moving muscle files...")

            muscle_pre = patient_dir / "gait" / "pre" / "muscle"
            if muscle_pre.exists():
                for f in muscle_pre.iterdir():
                    if f.is_file():
                        dest = patient_dir / "pre" / "muscle" / f.name
                        self._exec_copy(f, dest, log)

            muscle_post = patient_dir / "gait" / "post" / "muscle"
            if muscle_post.exists():
                for f in muscle_post.iterdir():
                    if f.is_file():
                        dest = patient_dir / "op1" / "muscle" / f.name
                        self._exec_copy(f, dest, log)

            # === Step 6: Transform and write metadata.yaml ===
            logger.info(f"  Transforming metadata files...")

            # Root metadata (simplified)
            root_meta = {
                'pid': patient_meta.get('pid'),
                'name': name,
                'gmfcs': gmfcs
            }
            root_meta_content = yaml.dump(root_meta, allow_unicode=True, default_flow_style=False)
            self._exec_write(patient_dir / "metadata.yaml.new", root_meta_content, log)

            # Pre visit metadata
            pre_gait = gait_meta.get('pre', {})
            pre_meta = {
                'timestamp': pre_gait.get('timestamp'),
                'age': patient_meta.get('surgery1', {}).get('age'),
                'height': pre_gait.get('height'),
                'weight': pre_gait.get('weight'),
                'foot': pre_gait.get('foot')
            }
            pre_meta_content = yaml.dump(pre_meta, allow_unicode=True, default_flow_style=False)
            self._exec_write(patient_dir / "pre" / "metadata.yaml", pre_meta_content, log)

            # Post (op1) visit metadata
            post_gait = gait_meta.get('post', {})
            op1_meta = {
                'timestamp': post_gait.get('timestamp'),
                'age': patient_meta.get('surgery1', {}).get('age'),  # or surgery2 if different
                'height': post_gait.get('height'),
                'weight': post_gait.get('weight'),
                'foot': post_gait.get('foot')
            }
            op1_meta_content = yaml.dump(op1_meta, allow_unicode=True, default_flow_style=False)
            self._exec_write(patient_dir / "op1" / "metadata.yaml", op1_meta_content, log)

            # === Step 7: Transform and write rom.yaml ===
            logger.info(f"  Transforming ROM files...")

            pre_rom = rom_data.get('pre_op', {})
            pre_rom_content = yaml.dump(pre_rom, allow_unicode=True, default_flow_style=False)
            self._exec_write(patient_dir / "pre" / "rom.yaml", pre_rom_content, log)

            post_rom = rom_data.get('post_op', {})
            post_rom_content = yaml.dump(post_rom, allow_unicode=True, default_flow_style=False)
            self._exec_write(patient_dir / "op1" / "rom.yaml", post_rom_content, log)

            # === Step 8: Move DICOM files ===
            logger.info(f"  Moving DICOM files...")

            dicom_op1 = patient_dir / "dicom" / "op1"
            if dicom_op1.exists():
                for item in dicom_op1.iterdir():
                    dest = patient_dir / "pre" / "dicom" / item.name
                    self._exec_copy(item, dest, log)

            dicom_op2 = patient_dir / "dicom" / "op2"
            if dicom_op2.exists():
                for item in dicom_op2.iterdir():
                    dest = patient_dir / "op1" / "dicom" / item.name
                    self._exec_copy(item, dest, log)

            # === Step 9: Cleanup (only if not dry-run) ===
            if not self.dry_run:
                logger.info(f"  Cleaning up old structure...")

                # Replace root metadata
                old_meta = patient_dir / "metadata.yaml"
                new_meta = patient_dir / "metadata.yaml.new"
                if new_meta.exists():
                    if old_meta.exists():
                        shutil.copy2(old_meta, str(old_meta) + ".bak")
                    shutil.move(str(new_meta), str(old_meta))

                # Remove old directories
                for old_dir in ['gait', 'anatomy', 'ckpt']:
                    old_path = patient_dir / old_dir
                    if old_path.exists():
                        self._exec_remove(old_path, log)

                # Remove old dicom directory (now empty parent)
                old_dicom = patient_dir / "dicom"
                if old_dicom.exists():
                    self._exec_remove(old_dicom, log)

                # Remove old rom.yaml (already backed up)
                old_rom = patient_dir / "rom.yaml"
                if old_rom.exists():
                    shutil.copy2(old_rom, str(old_rom) + ".bak")
                    old_rom.unlink()

            log.status = 'completed'
            log.completed_at = datetime.now().isoformat()
            logger.info(f"  Migration completed for {pid}")

        except Exception as e:
            log.status = 'failed'
            log.errors.append(str(e))
            logger.error(f"  Migration failed for {pid}: {e}")
            raise

        # Save log
        self._save_log(log)

        return log

    def _save_log(self, log: MigrationLog):
        """Save migration log to file"""
        if self.dry_run:
            return

        log_file = self.log_dir / f"{log.pid}.yaml"

        log_dict = {
            'pid': log.pid,
            'status': log.status,
            'started_at': log.started_at,
            'completed_at': log.completed_at,
            'errors': log.errors,
            'operations': [
                {
                    'op_type': op.op_type,
                    'source': op.source,
                    'dest': op.dest,
                    'status': op.status,
                    'backup_path': op.backup_path
                }
                for op in log.operations
            ]
        }

        if log.snapshot_before:
            log_dict['snapshot_before'] = {
                'c3d_pre_count': log.snapshot_before.c3d_pre_count,
                'c3d_post_count': log.snapshot_before.c3d_post_count,
                'h5_pre_count': log.snapshot_before.h5_pre_count,
                'h5_post_count': log.snapshot_before.h5_post_count,
                'skeleton_pre': log.snapshot_before.skeleton_pre,
                'skeleton_post': log.snapshot_before.skeleton_post,
                'muscle_pre': log.snapshot_before.muscle_pre,
                'muscle_post': log.snapshot_before.muscle_post,
                'total_size_bytes': log.snapshot_before.total_size_bytes
            }

        with open(log_file, 'w') as f:
            yaml.dump(log_dict, f, allow_unicode=True, default_flow_style=False)

    def verify_patient(self, pid: str) -> bool:
        """Verify migration success for a patient"""
        patient_dir = self.rm_root / pid

        logger.info(f"Verifying migration for {pid}...")

        # Check new structure exists
        required_dirs = [
            patient_dir / "pre" / "gait",
            patient_dir / "pre" / "motion",
            patient_dir / "pre" / "skeleton",
            patient_dir / "pre" / "muscle",
            patient_dir / "op1" / "gait",
            patient_dir / "op1" / "motion",
            patient_dir / "op1" / "skeleton",
            patient_dir / "op1" / "muscle",
        ]

        for d in required_dirs:
            if not d.exists():
                logger.error(f"  Missing directory: {d}")
                return False

        # Check metadata files
        required_files = [
            patient_dir / "metadata.yaml",
            patient_dir / "pre" / "metadata.yaml",
            patient_dir / "pre" / "rom.yaml",
            patient_dir / "op1" / "metadata.yaml",
            patient_dir / "op1" / "rom.yaml",
        ]

        for f in required_files:
            if not f.exists():
                logger.error(f"  Missing file: {f}")
                return False

            # Validate YAML
            try:
                with open(f) as fh:
                    yaml.safe_load(fh)
            except Exception as e:
                logger.error(f"  Invalid YAML: {f} - {e}")
                return False

        # Check root metadata has required fields
        with open(patient_dir / "metadata.yaml") as f:
            root_meta = yaml.safe_load(f)

        for field in ['pid', 'name', 'gmfcs']:
            if field not in root_meta:
                logger.error(f"  Missing field in root metadata: {field}")
                return False

        # Check pre metadata has required fields
        with open(patient_dir / "pre" / "metadata.yaml") as f:
            pre_meta = yaml.safe_load(f)

        for field in ['timestamp', 'height', 'weight']:
            if field not in pre_meta:
                logger.error(f"  Missing field in pre metadata: {field}")
                return False

        # Check old structure removed
        old_dirs = [
            patient_dir / "gait",
            patient_dir / "anatomy",
        ]

        for d in old_dirs:
            if d.exists():
                logger.warning(f"  Old directory still exists: {d}")

        logger.info(f"  Verification passed for {pid}")
        return True

    def rollback_patient(self, pid: str) -> bool:
        """Rollback migration for a patient using log file"""
        log_file = self.log_dir / f"{pid}.yaml"

        if not log_file.exists():
            logger.error(f"No migration log found for {pid}")
            return False

        with open(log_file) as f:
            log_dict = yaml.safe_load(f)

        logger.info(f"Rolling back migration for {pid}...")

        # Reverse operations in LIFO order
        operations = log_dict.get('operations', [])
        for op in reversed(operations):
            if op['status'] != 'completed':
                continue

            op_type = op['op_type']

            if op_type == 'mkdir':
                # Remove directory if empty
                path = Path(op['dest'])
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()
                    logger.debug(f"  Removed directory: {path}")

            elif op_type == 'copy':
                # Remove copied file
                dest = Path(op['dest'])
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                    logger.debug(f"  Removed copy: {dest}")

            elif op_type == 'move':
                # Move back
                src = Path(op['source'])
                dest = Path(op['dest'])
                if dest.exists() and not src.exists():
                    shutil.move(str(dest), str(src))
                    logger.debug(f"  Restored: {dest} -> {src}")

            elif op_type == 'write':
                # Restore backup or remove
                dest = Path(op['dest'])
                backup = op.get('backup_path')
                if backup and Path(backup).exists():
                    shutil.move(backup, str(dest))
                    logger.debug(f"  Restored from backup: {dest}")
                elif dest.exists():
                    dest.unlink()
                    logger.debug(f"  Removed written file: {dest}")

            elif op_type == 'remove':
                # Cannot restore removed files - warn
                logger.warning(f"  Cannot restore removed: {op['source']}")

        # Restore backup files
        patient_dir = self.rm_root / pid
        for backup in patient_dir.glob("*.bak"):
            original = backup.with_suffix('')
            if not original.exists():
                shutil.move(str(backup), str(original))
                logger.info(f"  Restored: {original}")

        logger.info(f"Rollback completed for {pid}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Migrate RM data structure')
    parser.add_argument('action', choices=['migrate', 'verify', 'rollback', 'list', 'snapshot'],
                       help='Action to perform')
    parser.add_argument('--patient', '-p', type=str, help='Specific patient ID')
    parser.add_argument('--all', '-a', action='store_true', help='Process all patients')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Dry run without making changes')
    parser.add_argument('--rm-root', type=str, default='/mnt/blue8T/CP/RM',
                       help='RM root directory')
    parser.add_argument('--gait-long-root', type=str, default='/mnt/blue8T/CP/Gait_long',
                       help='Gait_long root directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    migrator = RMMigrator(
        rm_root=args.rm_root,
        gait_long_root=args.gait_long_root,
        dry_run=args.dry_run
    )

    if args.action == 'list':
        patients = migrator.get_all_patients()
        print(f"Found {len(patients)} patients:")
        for p in patients:
            print(f"  {p}")
        return

    if args.action == 'snapshot':
        if not args.patient:
            parser.error("--patient required for snapshot")
        snapshot = migrator.create_snapshot(args.patient)
        print(yaml.dump(snapshot.__dict__, allow_unicode=True))
        return

    # Get patients to process
    if args.all:
        patients = migrator.get_all_patients()
    elif args.patient:
        patients = [args.patient]
    else:
        parser.error("Specify --patient or --all")
        return

    # Process each patient
    success_count = 0
    fail_count = 0

    for pid in patients:
        try:
            if args.action == 'migrate':
                log = migrator.migrate_patient(pid)
                if log.status in ['completed', 'skipped', 'dry_run']:
                    success_count += 1
                else:
                    fail_count += 1

            elif args.action == 'verify':
                if migrator.verify_patient(pid):
                    success_count += 1
                else:
                    fail_count += 1

            elif args.action == 'rollback':
                if migrator.rollback_patient(pid):
                    success_count += 1
                else:
                    fail_count += 1

        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")
            fail_count += 1

    # Summary
    print(f"\nSummary: {success_count} succeeded, {fail_count} failed")


if __name__ == '__main__':
    main()
