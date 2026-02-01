#!/usr/bin/env python3
"""
Checkpoint Upload Tool - ncurses UI for uploading checkpoints to FTP.

Usage:
    python scripts/upload_ckpt.py [runs_path]

Navigation:
    Arrow keys  - Navigate
    Enter       - Enter directory / Select
    u           - Upload ALL (batch mode)
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
                name_col = 30
                line = f"{marker}{name:<{name_col}}{stats}"
                self.stdscr.addstr(y, 0, line[:w-1].ljust(w-1))

            if idx == self.selected_idx:
                self.stdscr.attroff(curses.A_REVERSE)

        # Status bar
        self.stdscr.addstr(h-3, 0, "─" * w)

        # Determine context-appropriate help
        if self.current_path == self.runs_path:
            help_text = " [Enter] Enter dir  [u] Upload ALL  [q] Quit"
        else:
            help_text = " [Enter] Select  [u] Upload ALL dates  [Backspace] Back  [q] Quit"

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

    def run(self) -> Optional[tuple[Path, str, str]]:
        """
        Run the browser, return (selected_path, mode, pid_visit) or None.
        mode: 'date' for single date dir, 'env' for batch upload
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
                if self.current_path != self.runs_path:
                    self.current_path = self.current_path.parent
                    self.selected_idx = 0
                    self.scroll_offset = 0
                    self.refresh_items()

            elif key in (curses.KEY_ENTER, 10, 13):
                if not self.items:
                    continue

                item = self.items[self.selected_idx]

                if item is None:
                    # Go to parent
                    self.current_path = self.current_path.parent
                    self.selected_idx = 0
                    self.scroll_offset = 0
                    self.refresh_items()
                else:
                    # Check if this is a date directory (contains checkpoints)
                    ckpts = get_checkpoints_in_date(item)
                    if ckpts:
                        # This is a date directory - select for upload
                        return self._handle_upload(item, 'date')
                    else:
                        # Enter directory
                        self.current_path = item
                        self.selected_idx = 0
                        self.scroll_offset = 0
                        self.refresh_items()

            elif key == ord('u'):
                # Upload ALL - batch mode
                if not self.items:
                    continue

                # Determine what we're uploading
                if self.current_path == self.runs_path:
                    # At runs/, need to select an env first
                    item = self.items[self.selected_idx]
                    if item is not None:
                        return self._handle_upload(item, 'env')
                else:
                    # Inside an env directory - upload all dates
                    return self._handle_upload(self.current_path, 'env')

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

    def _draw_progress(self, status: str, current: int, total: int, filename: str = ""):
        """Draw progress bar."""
        h, w = self.stdscr.getmaxyx()

        # Clear progress area
        self.stdscr.addstr(h-4, 0, " " * (w-1))
        self.stdscr.addstr(h-3, 0, " " * (w-1))
        self.stdscr.addstr(h-2, 0, " " * (w-1))

        # Status
        self.stdscr.addstr(h-4, 2, status[:w-4])

        # Progress bar
        bar_w = w - 20
        if total > 0:
            filled = int((current / total) * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = f"{100 * current // total:3d}%"
            self.stdscr.addstr(h-3, 2, f"[{bar}] {pct}")

        # Current file
        if filename:
            self.stdscr.addstr(h-2, 2, f"File: {filename}"[:w-4])

        self.stdscr.refresh()

    def upload(self, local_path: Path, pid_visit: str, mode: str) -> bool:
        """Upload checkpoints to FTP."""
        if not self.connect():
            return False

        try:
            root = self.config['root']  # /mnt/blue8T/CP/RM

            if mode == 'date':
                # Single date directory
                dates = [local_path]
                env_name = local_path.parent.name
            else:
                # All dates in env directory
                dates = get_dates_in_env(local_path)
                env_name = local_path.name

            total_dates = len(dates)
            total_ckpts = sum(len(get_checkpoints_in_date(d)) for d in dates)
            ckpt_num = 0

            for date_idx, date_path in enumerate(dates):
                date_name = date_path.name
                ckpts = get_checkpoints_in_date(date_path)

                for ckpt_idx, ckpt_path in enumerate(ckpts):
                    ckpt_num += 1
                    ckpt_name = ckpt_path.name

                    status = f"Uploading {date_name} ({date_idx+1}/{total_dates}) - {ckpt_name} ({ckpt_num}/{total_ckpts})"

                    # Remote path: /{root}/{pid}/{visit}/ckpt/{env_name}/{date}/{ckpt}/
                    remote_dir = f"{root}/{pid_visit}/ckpt/{env_name}/{date_name}/{ckpt_name}"
                    self._mkdirs(remote_dir)

                    # Upload all files in checkpoint
                    files = [f for f in ckpt_path.iterdir() if f.is_file()]
                    for file_idx, file_path in enumerate(files):
                        self._draw_progress(status, file_idx, len(files), file_path.name)

                        remote_file = f"{remote_dir}/{file_path.name}"
                        with open(file_path, 'rb') as f:
                            self.ftp.storbinary(f'STOR {remote_file}', f)

                    self._draw_progress(status, len(files), len(files), "Done")

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
        path, mode, pid_visit = result

        # Upload
        uploader = FTPUploader(ftp_config, stdscr)
        success = uploader.upload(path, pid_visit, mode)

        if success:
            browser.show_message("Success", "Upload completed successfully!")
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
