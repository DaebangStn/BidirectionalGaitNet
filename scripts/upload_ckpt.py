#!/usr/bin/env python3
"""
Checkpoint Upload Tool - ncurses UI for uploading checkpoints to FTP.

Usage:
    python scripts/upload_ckpt.py [runs_path]

Navigation:
    Up/Down     - Navigate list
    Enter       - Enter directory
    Space/s     - Select/deselect directory
    a           - Select all
    c           - Clear selections
    u           - Upload selected
    Backspace   - Go back
    q           - Quit
"""

import curses
import sys
import os
import re
from pathlib import Path
from ftplib import FTP
from typing import Optional
import time
import shutil
import yaml


def load_ftp_config(config_path: Path) -> dict:
    """Load FTP configuration from rm_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['backends']['pid_ftp']


def detect_pid_from_metadata(ckpt_dir: Path) -> Optional[str]:
    """
    Read metadata.yaml, find @pid: URIs, extract pid/visit.
    Returns: "29792292/pre" or None
    """
    metadata_path = ckpt_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
    except Exception:
        return None

    def find_pid_uri(obj):
        if isinstance(obj, str) and obj.startswith('@pid:'):
            # @pid:29792292/pre/motion/... → 29792292/pre
            match = re.match(r'@pid:([^/]+/[^/]+)/', obj)
            if match:
                return match.group(1)
        elif isinstance(obj, dict):
            for v in obj.values():
                result = find_pid_uri(v)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_pid_uri(item)
                if result:
                    return result
        return None

    # Check environment.pid directly
    pid = metadata.get('pid')
    if not pid and 'environment' in metadata:
        pid = metadata['environment'].get('pid')

    return pid or find_pid_uri(metadata)


def get_directory_stats(path: Path) -> tuple[int, int, int]:
    """Get counts of dates, checkpoints, and total size in bytes."""
    date_count = 0
    ckpt_count = 0
    total_size = 0

    if not path.is_dir():
        return 0, 0, 0

    # Check if this is a date directory (contains checkpoint subdirs)
    subdirs = [d for d in path.iterdir() if d.is_dir()]

    for subdir in subdirs:
        # Check if subdir is a checkpoint (contains .pt files)
        pt_files = list(subdir.glob("*.pt"))
        if pt_files:
            # This is a date directory, subdir is a checkpoint
            ckpt_count += 1
            for f in subdir.iterdir():
                if f.is_file():
                    total_size += f.stat().st_size
        else:
            # This might be a date directory, count its checkpoints
            date_count += 1
            for ckpt_dir in subdir.iterdir():
                if ckpt_dir.is_dir() and list(ckpt_dir.glob("*.pt")):
                    ckpt_count += 1
                    for f in ckpt_dir.iterdir():
                        if f.is_file():
                            total_size += f.stat().st_size

    # If we found checkpoints but no dates, this is a date directory
    if ckpt_count > 0 and date_count == 0:
        date_count = 1

    return date_count, ckpt_count, total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def get_checkpoints_in_date(date_path: Path) -> list[Path]:
    """Get list of checkpoint directories in a date directory."""
    checkpoints = []
    for d in sorted(date_path.iterdir()):
        if d.is_dir() and list(d.glob("*.pt")):
            checkpoints.append(d)
    return checkpoints


def get_dates_in_env(env_path: Path) -> list[Path]:
    """Get list of date directories in an env directory."""
    dates = []
    for d in sorted(env_path.iterdir()):
        if d.is_dir():
            # Check if it contains checkpoint subdirs
            ckpts = get_checkpoints_in_date(d)
            if ckpts:
                dates.append(d)
    return dates


class CheckpointBrowser:
    """ncurses-based directory browser for runs/"""

    def __init__(self, stdscr, runs_path: Path):
        self.stdscr = stdscr
        self.runs_path = runs_path
        self.current_path = runs_path
        self.selected_idx = 0
        self.scroll_offset = 0
        self.items: list[Path] = []
        self.mode = 'browse'  # browse, confirm, input, upload
        self.selected_paths: set[Path] = set()  # Selected directories for upload

        # Colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # Directory
        curses.init_pair(2, curses.COLOR_GREEN, -1)   # Selected
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Status
        curses.init_pair(4, curses.COLOR_RED, -1)     # Error
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header

        self.refresh_items()

    def refresh_items(self):
        """Refresh the directory listing."""
        self.items = []
        if self.current_path != self.runs_path:
            self.items.append(None)  # Parent directory marker

        if self.current_path.is_dir():
            dirs = sorted([d for d in self.current_path.iterdir() if d.is_dir()])
            self.items.extend(dirs)

        self.selected_idx = min(self.selected_idx, max(0, len(self.items) - 1))

    def get_display_height(self) -> int:
        """Get the available height for the file list."""
        h, _ = self.stdscr.getmaxyx()
        return h - 6  # Header (2) + status bar (2) + borders (2)

    def draw(self):
        """Draw the main browser interface."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Title bar
        title = " Checkpoint Upload "
        self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(0, 0, "─" * w)
        self.stdscr.addstr(0, (w - len(title)) // 2, title)
        self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

        # Path display
        rel_path = self.current_path.relative_to(self.runs_path.parent)
        path_str = f" Path: {rel_path}/"
        self.stdscr.addstr(1, 0, path_str[:w-1])
        self.stdscr.addstr(2, 0, "─" * w)

        # File list
        display_h = self.get_display_height()
        list_start = 3

        # Adjust scroll
        if self.selected_idx < self.scroll_offset:
            self.scroll_offset = self.selected_idx
        elif self.selected_idx >= self.scroll_offset + display_h:
            self.scroll_offset = self.selected_idx - display_h + 1

        for i in range(display_h):
            idx = self.scroll_offset + i
            if idx >= len(self.items):
                break

            item = self.items[idx]
            y = list_start + i

            # Selection marker
            if idx == self.selected_idx:
                self.stdscr.attron(curses.A_REVERSE)
                marker = " > "
            else:
                marker = "   "

            if item is None:
                # Parent directory
                line = f"{marker}../"
                self.stdscr.addstr(y, 0, line[:w-1].ljust(w-1))
            else:
                # Directory with stats
                is_selected = item in self.selected_paths
                sel_marker = "[x] " if is_selected else "[ ] "
                name = item.name + "/"
                dates, ckpts, size = get_directory_stats(item)

                if dates > 1:
                    stats = f"{dates} dates, {ckpts} ckpts"
                elif ckpts > 0:
                    stats = f"{ckpts} checkpoints"
                else:
                    stats = ""

                if size > 0:
                    stats += f" ({format_size(size)})"

                # Format line
                name_col = 26
                line = f"{marker}{sel_marker}{name:<{name_col}}{stats}"
                self.stdscr.addstr(y, 0, line[:w-1].ljust(w-1))

            if idx == self.selected_idx:
                self.stdscr.attroff(curses.A_REVERSE)

        # Status bar
        self.stdscr.addstr(h-3, 0, "─" * w)

        # Determine context-appropriate help
        sel_count = len(self.selected_paths)
        sel_info = f" ({sel_count} sel)" if sel_count > 0 else ""
        help_text = f" [Space] Sel  [a] All  [c] Clear  [u] Upload{sel_info}  [Enter] In  [BS] Back  [q] Quit"

        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(h-2, 0, help_text[:w-1])
        self.stdscr.attroff(curses.color_pair(3))

        self.stdscr.refresh()

    def get_input_string(self, prompt: str, default: str = "") -> Optional[str]:
        """Show input dialog and get string from user."""
        h, w = self.stdscr.getmaxyx()
        box_w = min(60, w - 4)
        box_h = 8
        box_y = (h - box_h) // 2
        box_x = (w - box_w) // 2

        # Create window
        win = curses.newwin(box_h, box_w, box_y, box_x)
        win.box()
        win.addstr(0, 2, " Upload Destination ")

        # Prompt
        win.addstr(2, 2, prompt[:box_w-4])

        # Input field
        input_y = 4
        input_x = 2
        input_w = box_w - 4

        value = default
        cursor_pos = len(value)

        curses.curs_set(1)  # Show cursor

        while True:
            # Draw input field
            win.addstr(input_y, input_x, " " * input_w)
            win.addstr(input_y, input_x, f"[{value}]"[:input_w])

            # Help text
            win.addstr(box_h-2, 2, "[Enter] Confirm  [Esc] Cancel")

            win.move(input_y, input_x + 1 + cursor_pos)
            win.refresh()

            key = win.getch()

            if key == 27:  # Escape
                curses.curs_set(0)
                return None
            elif key in (curses.KEY_ENTER, 10, 13):
                curses.curs_set(0)
                return value
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                if cursor_pos > 0:
                    value = value[:cursor_pos-1] + value[cursor_pos:]
                    cursor_pos -= 1
            elif key == curses.KEY_LEFT:
                cursor_pos = max(0, cursor_pos - 1)
            elif key == curses.KEY_RIGHT:
                cursor_pos = min(len(value), cursor_pos + 1)
            elif key == curses.KEY_HOME:
                cursor_pos = 0
            elif key == curses.KEY_END:
                cursor_pos = len(value)
            elif 32 <= key <= 126:  # Printable
                value = value[:cursor_pos] + chr(key) + value[cursor_pos:]
                cursor_pos += 1

        curses.curs_set(0)
        return None

    def show_confirmation(self, lines: list[str]) -> bool:
        """Show confirmation dialog."""
        h, w = self.stdscr.getmaxyx()
        box_w = min(70, w - 4)
        box_h = len(lines) + 6
        box_y = (h - box_h) // 2
        box_x = (w - box_w) // 2

        win = curses.newwin(box_h, box_w, box_y, box_x)
        win.box()
        win.addstr(0, 2, " Confirm Upload ")

        for i, line in enumerate(lines):
            win.addstr(2 + i, 2, line[:box_w-4])

        win.attron(curses.color_pair(3))
        win.addstr(box_h-2, 2, "[y] Yes  [n/Esc] No")
        win.attroff(curses.color_pair(3))

        win.refresh()

        while True:
            key = win.getch()
            if key in (ord('y'), ord('Y')):
                return True
            elif key in (ord('n'), ord('N'), 27):  # n or Escape
                return False

    def show_message(self, title: str, message: str, wait: bool = True):
        """Show a message dialog."""
        h, w = self.stdscr.getmaxyx()
        box_w = min(60, w - 4)
        box_h = 5
        box_y = (h - box_h) // 2
        box_x = (w - box_w) // 2

        win = curses.newwin(box_h, box_w, box_y, box_x)
        win.box()
        win.addstr(0, 2, f" {title} ")
        win.addstr(2, 2, message[:box_w-4])

        if wait:
            win.addstr(box_h-1, 2, " Press any key... ")
            win.refresh()
            win.getch()
        else:
            win.refresh()

    def show_delete_selection(self, paths: list[Path]) -> list[Path]:
        """Show selection dialog for directories to delete. All selected by default."""
        h, w = self.stdscr.getmaxyx()

        # All selected by default
        selected = set(paths)
        cursor_idx = 0
        scroll_offset = 0

        box_w = min(80, w - 4)
        list_h = min(len(paths) + 2, h - 10)
        box_h = list_h + 6
        box_y = (h - box_h) // 2
        box_x = (w - box_w) // 2

        while True:
            win = curses.newwin(box_h, box_w, box_y, box_x)
            win.box()
            win.addstr(0, 2, " Delete Uploaded Files ")

            # Instructions
            win.addstr(1, 2, "Select directories to delete (all selected by default):")

            # Calculate visible range
            visible_h = list_h
            if cursor_idx < scroll_offset:
                scroll_offset = cursor_idx
            elif cursor_idx >= scroll_offset + visible_h:
                scroll_offset = cursor_idx - visible_h + 1

            # Draw list
            for i in range(visible_h):
                idx = scroll_offset + i
                if idx >= len(paths):
                    break

                path = paths[idx]
                is_selected = path in selected
                is_cursor = idx == cursor_idx

                sel_mark = "[x]" if is_selected else "[ ]"
                name = path.name

                line = f" {sel_mark} {name}"

                y = 3 + i
                if is_cursor:
                    win.attron(curses.A_REVERSE)
                win.addstr(y, 2, line[:box_w-4].ljust(box_w-4))
                if is_cursor:
                    win.attroff(curses.A_REVERSE)

            # Status and help
            sel_count = len(selected)
            win.addstr(box_h-3, 2, f"Selected: {sel_count}/{len(paths)} directories")
            win.attron(curses.color_pair(3))
            win.addstr(box_h-2, 2, "[Space] Toggle  [a] All  [c] Clear  [Enter] Delete  [Esc] Cancel")
            win.attroff(curses.color_pair(3))

            win.refresh()

            key = win.getch()

            if key == 27:  # Escape - cancel
                return []
            elif key in (curses.KEY_ENTER, 10, 13):  # Enter - confirm
                return list(selected)
            elif key == curses.KEY_UP:
                cursor_idx = max(0, cursor_idx - 1)
            elif key == curses.KEY_DOWN:
                cursor_idx = min(len(paths) - 1, cursor_idx + 1)
            elif key == ord(' '):  # Space - toggle
                path = paths[cursor_idx]
                if path in selected:
                    selected.discard(path)
                else:
                    selected.add(path)
            elif key == ord('a'):  # Select all
                selected = set(paths)
            elif key == ord('c'):  # Clear all
                selected.clear()

    def run(self) -> Optional[tuple[Path | list[Path], str, str]]:
        """
        Run the browser, return (selected_path(s), mode, pid_visit) or None.
        mode: 'date' for single date dir, 'env' for single env, 'dates'/'envs' for batch
        """
        curses.curs_set(0)

        while True:
            self.draw()
            key = self.stdscr.getch()

            if key == ord('q'):
                return None

            elif key == curses.KEY_UP:
                if self.selected_idx > 0:
                    self.selected_idx -= 1

            elif key == curses.KEY_DOWN:
                if self.selected_idx < len(self.items) - 1:
                    self.selected_idx += 1

            elif key == curses.KEY_PPAGE:  # Page Up
                self.selected_idx = max(0, self.selected_idx - self.get_display_height())

            elif key == curses.KEY_NPAGE:  # Page Down
                self.selected_idx = min(len(self.items) - 1,
                                        self.selected_idx + self.get_display_height())

            elif key in (curses.KEY_BACKSPACE, 127, 8):
                # Backspace = go back
                if self.current_path != self.runs_path:
                    self.current_path = self.current_path.parent
                    self.selected_idx = 0
                    self.scroll_offset = 0
                    self.selected_paths.clear()  # Clear selections when going back
                    self.refresh_items()

            elif key in (curses.KEY_ENTER, 10, 13):
                # Enter = enter directory
                if not self.items:
                    continue

                item = self.items[self.selected_idx]

                if item is None:
                    # Go to parent
                    self.current_path = self.current_path.parent
                    self.selected_idx = 0
                    self.scroll_offset = 0
                    self.selected_paths.clear()
                    self.refresh_items()
                elif item.is_dir():
                    # Enter directory
                    self.current_path = item
                    self.selected_idx = 0
                    self.scroll_offset = 0
                    self.selected_paths.clear()
                    self.refresh_items()

            elif key == ord(' '):
                # Space = toggle selection
                if not self.items:
                    continue
                item = self.items[self.selected_idx]
                if item is not None:  # Can't select parent ".."
                    if item in self.selected_paths:
                        self.selected_paths.discard(item)
                    else:
                        self.selected_paths.add(item)

            elif key == ord('s'):
                # Toggle selection of current item
                if not self.items:
                    continue
                item = self.items[self.selected_idx]
                if item is not None:  # Can't select parent ".."
                    if item in self.selected_paths:
                        self.selected_paths.discard(item)
                    else:
                        self.selected_paths.add(item)

            elif key == ord('a'):
                # Select all directories in current view
                for item in self.items:
                    if item is not None:  # Skip parent ".."
                        self.selected_paths.add(item)

            elif key == ord('c'):
                # Clear all selections
                self.selected_paths.clear()

            elif key == ord('u'):
                # Upload selected directories
                if not self.selected_paths:
                    self.show_message("Info", "No directories selected. Use [s] or [a] to select.")
                    continue

                return self._handle_batch_upload()

        return None

    def _handle_upload(self, path: Path, mode: str) -> Optional[tuple[Path, str, str]]:
        """Handle the upload selection and destination input."""
        # Detect PID from first checkpoint
        detected_pid = None

        if mode == 'date':
            ckpts = get_checkpoints_in_date(path)
            if ckpts:
                detected_pid = detect_pid_from_metadata(ckpts[0])
        else:  # env mode
            dates = get_dates_in_env(path)
            if dates:
                ckpts = get_checkpoints_in_date(dates[0])
                if ckpts:
                    detected_pid = detect_pid_from_metadata(ckpts[0])

        # Get stats for confirmation
        dates, ckpts, size = get_directory_stats(path)

        # Build prompt
        if detected_pid:
            prompt = f"Detected: {detected_pid} | Enter pid/visit:"
            default = detected_pid
        else:
            prompt = "Enter destination (pid/visit):"
            default = ""

        # Get destination
        pid_visit = self.get_input_string(prompt, default)
        if not pid_visit:
            return None

        # Validate format
        if '/' not in pid_visit:
            self.show_message("Error", "Format must be: pid/visit (e.g. 29792292/pre)")
            return None

        # Get env name for preview
        if mode == 'date':
            env_name = path.parent.name
            date_name = path.name
            preview = f"@pid:{pid_visit}/ckpt/{env_name}/{date_name}/..."
        else:
            env_name = path.name
            preview = f"@pid:{pid_visit}/ckpt/{env_name}/..."

        # Confirmation
        lines = [
            f"Destination: {pid_visit}",
            f"Preview: {preview}",
            "",
            f"Mode: {'SINGLE DATE' if mode == 'date' else 'BATCH'} - {dates} date(s), {ckpts} checkpoint(s)",
            f"Total size: {format_size(size)}",
        ]

        if self.show_confirmation(lines):
            return (path, mode, pid_visit)

        return None

    def _handle_batch_upload(self) -> Optional[tuple[list[Path], str, str]]:
        """Handle batch upload of selected directories."""
        # Determine mode based on selected paths
        # Check if selected paths are date directories or env directories
        all_dates = all(get_checkpoints_in_date(p) for p in self.selected_paths)

        if all_dates:
            mode = 'dates'  # Multiple date directories
        else:
            mode = 'envs'  # Multiple env directories

        # Detect PID from first checkpoint in first selected path
        detected_pid = None
        for path in self.selected_paths:
            if mode == 'dates':
                ckpts = get_checkpoints_in_date(path)
                if ckpts:
                    detected_pid = detect_pid_from_metadata(ckpts[0])
                    break
            else:
                dates = get_dates_in_env(path)
                if dates:
                    ckpts = get_checkpoints_in_date(dates[0])
                    if ckpts:
                        detected_pid = detect_pid_from_metadata(ckpts[0])
                        break

        # Calculate total stats
        total_dates = 0
        total_ckpts = 0
        total_size = 0
        for path in self.selected_paths:
            dates, ckpts, size = get_directory_stats(path)
            total_dates += dates
            total_ckpts += ckpts
            total_size += size

        # Build prompt
        if detected_pid:
            prompt = f"Detected: {detected_pid} | Enter pid/visit:"
            default = detected_pid
        else:
            prompt = "Enter destination (pid/visit):"
            default = ""

        # Get destination
        pid_visit = self.get_input_string(prompt, default)
        if not pid_visit:
            return None

        # Validate format
        if '/' not in pid_visit:
            self.show_message("Error", "Format must be: pid/visit (e.g. 29792292/pre)")
            return None

        # Preview
        selected_names = ", ".join(p.name for p in sorted(self.selected_paths)[:3])
        if len(self.selected_paths) > 3:
            selected_names += f" (+{len(self.selected_paths) - 3} more)"

        # Confirmation
        lines = [
            f"Destination: {pid_visit}",
            f"Selected: {selected_names}",
            "",
            f"Mode: BATCH - {len(self.selected_paths)} dirs, {total_dates} date(s), {total_ckpts} checkpoint(s)",
            f"Total size: {format_size(total_size)}",
        ]

        if self.show_confirmation(lines):
            return (list(self.selected_paths), mode, pid_visit)

        return None


class FTPUploader:
    """FTP upload handler."""

    def __init__(self, config: dict, stdscr):
        self.config = config
        self.stdscr = stdscr
        self.ftp: Optional[FTP] = None

    def connect(self) -> bool:
        """Connect to FTP server."""
        try:
            self.ftp = FTP()
            self.ftp.connect(self.config['ip'], self.config['port'])
            self.ftp.login(self.config['username'], self.config['password'])
            return True
        except Exception as e:
            self._show_error(f"FTP connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from FTP server."""
        if self.ftp:
            try:
                self.ftp.quit()
            except Exception:
                pass
            self.ftp = None

    def _show_error(self, message: str):
        """Show error message."""
        h, w = self.stdscr.getmaxyx()
        self.stdscr.addstr(h-1, 0, f"ERROR: {message}"[:w-1])
        self.stdscr.refresh()
        self.stdscr.getch()

    def _mkdirs(self, path: str):
        """Create remote directories recursively."""
        dirs = path.split('/')
        current = ""
        for d in dirs:
            if not d:
                continue
            current += "/" + d
            try:
                self.ftp.mkd(current)
            except Exception:
                pass  # Directory might already exist

    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m"

    def _draw_progress(self, status: str, file_current: int, file_total: int,
                        total_current: int, total_files: int, start_time: float,
                        filename: str = ""):
        """Draw two progress bars - total files and current file with ETA."""
        h, w = self.stdscr.getmaxyx()

        # Clear progress area
        for i in range(6):
            self.stdscr.addstr(h-6+i, 0, " " * (w-1))

        # Status
        self.stdscr.addstr(h-6, 2, status[:w-4])

        bar_w = min(w - 22, 60)  # Max bar width 60 chars (total line ~100)

        # Calculate ETA
        elapsed = time.time() - start_time
        eta_str = ""
        if total_current > 0 and total_files > 0:
            rate = total_current / elapsed  # files per second
            remaining = total_files - total_current
            if rate > 0:
                eta_seconds = remaining / rate
                eta_str = f"ETA: {self._format_time(eta_seconds)}"

        # Total files progress bar
        if total_files > 0:
            filled = int((total_current / total_files) * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = f"{100 * total_current // total_files:3d}%"
            self.stdscr.addstr(h-5, 2, f"Total: [{bar}] {pct} ({total_current}/{total_files})"[:w-2])

        # Current checkpoint file progress bar
        if file_total > 0:
            filled = int((file_current / file_total) * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = f"{100 * file_current // file_total:3d}%"
            self.stdscr.addstr(h-4, 2, f"Ckpt:  [{bar}] {pct}"[:w-2])

        # ETA and elapsed time
        elapsed_str = f"Elapsed: {self._format_time(elapsed)}"
        time_info = f"{elapsed_str}  {eta_str}".strip()
        self.stdscr.addstr(h-3, 2, time_info[:w-4])

        # Current file
        if filename:
            self.stdscr.addstr(h-2, 2, f"File: {filename}"[:w-4])

        self.stdscr.refresh()

    def upload(self, local_paths: Path | list[Path], pid_visit: str, mode: str) -> bool:
        """Upload checkpoints to FTP."""
        if not self.connect():
            return False

        try:
            root = self.config['root']  # /mnt/blue8T/CP/RM

            # Normalize to list of (date_path, env_name) tuples
            date_env_pairs: list[tuple[Path, str]] = []

            if isinstance(local_paths, Path):
                # Single path (legacy mode)
                if mode == 'date':
                    date_env_pairs = [(local_paths, local_paths.parent.name)]
                else:  # env mode
                    dates = get_dates_in_env(local_paths)
                    date_env_pairs = [(d, local_paths.name) for d in dates]
            else:
                # List of paths (batch mode)
                for path in local_paths:
                    if mode == 'dates':
                        # Each path is a date directory
                        date_env_pairs.append((path, path.parent.name))
                    else:  # envs mode
                        # Each path is an env directory
                        dates = get_dates_in_env(path)
                        date_env_pairs.extend((d, path.name) for d in dates)

            total_dates = len(date_env_pairs)
            total_ckpts = sum(len(get_checkpoints_in_date(d)) for d, _ in date_env_pairs)

            # Count total files across all checkpoints
            total_files = 0
            for date_path, _ in date_env_pairs:
                for ckpt_path in get_checkpoints_in_date(date_path):
                    total_files += len([f for f in ckpt_path.iterdir() if f.is_file()])

            ckpt_num = 0
            global_file_num = 0
            start_time = time.time()

            for date_idx, (date_path, env_name) in enumerate(date_env_pairs):
                date_name = date_path.name
                ckpts = get_checkpoints_in_date(date_path)

                for ckpt_idx, ckpt_path in enumerate(ckpts):
                    ckpt_num += 1
                    ckpt_name = ckpt_path.name

                    status = f"Uploading {env_name}/{date_name} - {ckpt_name} ({ckpt_num}/{total_ckpts})"

                    # Remote path: /{root}/{pid}/{visit}/ckpt/{env_name}/{date}/{ckpt}/
                    remote_dir = f"{root}/{pid_visit}/ckpt/{env_name}/{date_name}/{ckpt_name}"
                    self._mkdirs(remote_dir)

                    # Upload all files in checkpoint
                    files = [f for f in ckpt_path.iterdir() if f.is_file()]
                    for file_idx, file_path in enumerate(files):
                        self._draw_progress(status, file_idx, len(files),
                                          global_file_num, total_files, start_time, file_path.name)

                        remote_file = f"{remote_dir}/{file_path.name}"
                        with open(file_path, 'rb') as f:
                            self.ftp.storbinary(f'STOR {remote_file}', f)

                        global_file_num += 1

                    self._draw_progress(status, len(files), len(files),
                                      global_file_num, total_files, start_time, "Done")

            return True

        except Exception as e:
            self._show_error(str(e))
            return False

        finally:
            self.disconnect()


def main(stdscr):
    """Main entry point."""
    # Get runs path from args or default
    if len(sys.argv) > 1:
        runs_path = Path(sys.argv[1])
    else:
        runs_path = Path(__file__).parent.parent / "runs"

    if not runs_path.exists():
        print(f"Error: runs directory not found: {runs_path}")
        return 1

    # Load FTP config
    config_path = Path(__file__).parent.parent / "data" / "rm_config.yaml"
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        return 1

    ftp_config = load_ftp_config(config_path)

    # Create browser
    browser = CheckpointBrowser(stdscr, runs_path)

    # Run browser
    result = browser.run()

    if result:
        paths, mode, pid_visit = result

        # Upload
        uploader = FTPUploader(ftp_config, stdscr)
        success = uploader.upload(paths, pid_visit, mode)

        if success:
            # Ask to delete uploaded files with selection UI
            if isinstance(paths, list):
                delete_targets = paths
            else:
                delete_targets = [paths]

            browser.show_message("Success", "Upload completed!", wait=True)

            # Show selection dialog (all selected by default)
            to_delete = browser.show_delete_selection(delete_targets)

            if to_delete:
                # Delete selected directories
                deleted = 0
                for target in to_delete:
                    try:
                        if target.is_dir():
                            shutil.rmtree(target)
                            deleted += 1
                    except Exception as e:
                        browser.show_message("Error", f"Failed to delete {target.name}: {e}")

                browser.show_message("Done", f"Deleted {deleted} directories")
            else:
                browser.show_message("Done", "Local files kept.")

            return 0
        else:
            return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = curses.wrapper(main)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
