#!/usr/bin/env python3
"""Run experiments for all datasets with all methods with interactive CLI.

This script iterates through all dataset directories in configs/ and runs
experiments with each specified method (dpg, dice).

Interactive Controls (when running with --parallel):
  Ctrl+C / d  - Detach: Exit but leave experiments running (can reconnect later)
  q           - Hard quit: Kill all running experiments immediately
  s           - Soft stop: Let running experiments finish, don't start new ones
  p           - Pause/Resume: Toggle starting new experiments
  r           - Refresh display

Usage:
  # Run all datasets with all methods
  python scripts/run_all_experiments.py
  
  # Run specific datasets only
  python scripts/run_all_experiments.py --datasets iris german_credit
  
  # Run specific methods only
  python scripts/run_all_experiments.py --methods dpg
  
  # Skip datasets that have already been processed
  python scripts/run_all_experiments.py --skip-existing
  
  # Dry run (show what would be executed)
  python scripts/run_all_experiments.py --dry-run
  
  # Run in offline mode (no wandb sync)
  python scripts/run_all_experiments.py --offline
  
  # Limit number of datasets (useful for testing)
  python scripts/run_all_experiments.py --limit 3
  
  # Run multiple experiments in parallel
  python scripts/run_all_experiments.py --parallel 4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
import signal
import threading
import queue
import select
import termios
import tty
import fcntl
import yaml
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import experiment status utilities
from utils.experiment_status import (
    PersistentStatus,
    ExperimentStatusInfo,
    get_experiment_status,
    get_all_experiment_statuses,
    get_global_stats,
    read_log_tail,
    get_log_file_path,
    is_process_running,
)

# Default methods to run
DEFAULT_METHODS = ['dpg', 'dice']

# Files to exclude from dataset detection
EXCLUDED_FILES = {'config.yaml', 'sweep_config.yaml'}

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def get_system_stats() -> Tuple[float, float, float, float]:
    """Get CPU and memory usage stats.
    
    Returns:
        Tuple of (cpu_percent, mem_used_gb, mem_total_gb, mem_percent)
    """
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)
        mem_percent = mem.percent
        return cpu_percent, mem_used_gb, mem_total_gb, mem_percent
    else:
        # Fallback: read from /proc on Linux
        try:
            # CPU usage (rough estimate from /proc/stat)
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                fields = line.split()[1:]
                idle = int(fields[3])
                total = sum(int(x) for x in fields[:7])
                # This is a rough estimate, not as accurate as psutil
                cpu_percent = 100.0 * (1 - idle / total) if total > 0 else 0.0
            
            # Memory from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])
                
                mem_total_kb = meminfo.get('MemTotal', 0)
                mem_available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                mem_used_kb = mem_total_kb - mem_available_kb
                
                mem_total_gb = mem_total_kb / (1024 ** 2)
                mem_used_gb = mem_used_kb / (1024 ** 2)
                mem_percent = 100.0 * mem_used_kb / mem_total_kb if mem_total_kb > 0 else 0.0
            
            return cpu_percent, mem_used_gb, mem_total_gb, mem_percent
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    EXTERNAL_RUNNING = "external_running"  # Running from another orchestrator


@dataclass
class Experiment:
    dataset: str
    method: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    return_code: Optional[int] = None
    last_log_line: str = ""
    log_lines: List[str] = field(default_factory=list)
    external_pid: Optional[int] = None  # PID of external process (from another orchestrator)
    
    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.method}"
    
    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def elapsed_str(self) -> str:
        elapsed = self.elapsed_time
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}m"
        else:
            return f"{elapsed/3600:.1f}h"


class ExperimentRunner:
    """Manages parallel experiment execution with interactive controls."""
    
    def __init__(
        self,
        experiments: List[Dict[str, str]],
        max_workers: int = 1,
        verbose: bool = False,
        offline: bool = False,
        overrides: Optional[List[str]] = None,
        output_dir: Optional[pathlib.Path] = None,
        skip_existing: bool = False,
        monitor_only: bool = False,  # New: just monitor, don't start new experiments
    ):
        self.max_workers = max_workers
        self.verbose = verbose
        self.offline = offline
        self.overrides = overrides or []
        self.output_dir = output_dir or REPO_ROOT / 'outputs'
        self.skip_existing = skip_existing
        self.monitor_only = monitor_only
        
        # Thread management (must be initialized before experiment loop)
        self.log_queue: queue.Queue = queue.Queue()
        self.reader_threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        
        # Control flags
        self.stop_requested = False  # Hard stop - kill all experiments
        self.detach_requested = False  # Detach - exit but leave experiments running
        self.soft_stop_requested = False or monitor_only  # Don't start new if monitor_only
        self.paused = False
        
        # Spinner animation
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_idx = 0
        
        # Stats
        self.start_time: Optional[float] = None
        
        # Display state
        self.last_display_lines = 0
        
        # Initialize experiments
        self.experiments: Dict[str, Experiment] = {}
        self.pending_queue: List[str] = []
        
        for exp_dict in experiments:
            exp = Experiment(dataset=exp_dict['dataset'], method=exp_dict['method'])
            self.experiments[exp.key] = exp
            
            # Check status from persistent status file
            persistent_status, status_info = get_experiment_status(
                exp.dataset, exp.method, self.output_dir
            )
            
            if persistent_status == PersistentStatus.FINISHED:
                # Already completed successfully - skip only if skip_existing is True
                if self.skip_existing:
                    exp.status = ExperimentStatus.SKIPPED
                    if status_info and status_info.start_time:
                        exp.start_time = status_info.start_time
                        exp.end_time = status_info.end_time
                else:
                    # Add to queue to re-run
                    self.pending_queue.append(exp.key)
            elif persistent_status == PersistentStatus.RUNNING:
                # Another process is running this experiment - track it
                if status_info and status_info.pid and is_process_running(status_info.pid):
                    exp.status = ExperimentStatus.EXTERNAL_RUNNING
                    exp.external_pid = status_info.pid
                    exp.start_time = status_info.start_time
                    # Start a thread to read its logs
                    self._start_external_log_reader(exp)
                else:
                    # Process died - it's actually an error, add to queue
                    if not monitor_only:
                        self.pending_queue.append(exp.key)
            elif persistent_status == PersistentStatus.ERROR:
                # Previous run errored - we should retry unless monitor_only
                if monitor_only:
                    exp.status = ExperimentStatus.FAILED
                    if status_info:
                        exp.start_time = status_info.start_time
                        exp.end_time = status_info.end_time
                        exp.last_log_line = status_info.error_message or "Error"
                else:
                    self.pending_queue.append(exp.key)
            else:
                # No status or unknown - add to queue unless monitor_only
                if not monitor_only:
                    self.pending_queue.append(exp.key)
    
    def _start_external_log_reader(self, exp: Experiment):
        """Start a thread to read logs from an external experiment's log file."""
        def read_external_logs():
            log_file = get_log_file_path(exp.dataset, exp.method, self.output_dir)
            last_position = 0
            
            while exp.status == ExperimentStatus.EXTERNAL_RUNNING:
                try:
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = f.tell()
                            
                            for line in new_lines:
                                line = line.rstrip()
                                if line:
                                    with self.lock:
                                        exp.last_log_line = line
                                        exp.log_lines.append(line)
                                        if len(exp.log_lines) > 100:
                                            exp.log_lines = exp.log_lines[-100:]
                    
                    # Check if the external process is still running
                    if exp.external_pid and not is_process_running(exp.external_pid):
                        # Process finished - check final status
                        final_status, _ = get_experiment_status(
                            exp.dataset, exp.method, self.output_dir
                        )
                        if final_status == PersistentStatus.FINISHED:
                            exp.status = ExperimentStatus.COMPLETED
                        else:
                            exp.status = ExperimentStatus.FAILED
                        exp.end_time = time.time()
                        break
                    
                    time.sleep(0.5)
                except Exception:
                    time.sleep(1)
        
        thread = threading.Thread(target=read_external_logs, daemon=True)
        thread.start()
        self.reader_threads.append(thread)
    
    def _check_existing_output(self, dataset: str, method: str) -> bool:
        """Check if experiment completed successfully (based on status file)."""
        status, info = get_experiment_status(dataset, method, self.output_dir)
        return status == PersistentStatus.FINISHED
    
    def _build_command(self, exp: Experiment) -> List[str]:
        """Build command for running an experiment."""
        cmd = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'run_experiment.py'),
            '--dataset', exp.dataset,
            '--method', exp.method,
        ]
        
        if self.verbose:
            cmd.append('--verbose')
        
        if self.offline:
            cmd.append('--offline')
        
        for override in self.overrides:
            cmd.extend(['--set', override])
        
        return cmd
    
    def _start_experiment(self, key: str) -> bool:
        """Start a single experiment subprocess."""
        exp = self.experiments[key]
        
        try:
            cmd = self._build_command(exp)
            
            # Start process with pipes for stdout/stderr
            exp.process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            exp.status = ExperimentStatus.RUNNING
            exp.start_time = time.time()
            
            # Start reader thread for this process
            reader_thread = threading.Thread(
                target=self._read_output,
                args=(exp,),
                daemon=True
            )
            reader_thread.start()
            self.reader_threads.append(reader_thread)
            
            return True
            
        except Exception as e:
            exp.status = ExperimentStatus.FAILED
            exp.last_log_line = f"Failed to start: {e}"
            return False
    
    def _read_output(self, exp: Experiment):
        """Read output from experiment process (runs in thread)."""
        try:
            for line in exp.process.stdout:
                line = line.rstrip()
                if line:
                    with self.lock:
                        exp.last_log_line = line
                        exp.log_lines.append(line)
                        # Keep only last 100 lines per experiment
                        if len(exp.log_lines) > 100:
                            exp.log_lines = exp.log_lines[-100:]
                    
                    # Put in queue for display
                    self.log_queue.put((exp.key, line))
        except Exception:
            pass
    
    def _check_completed(self):
        """Check for completed experiments and update their status."""
        for key, exp in self.experiments.items():
            if exp.status == ExperimentStatus.RUNNING and exp.process:
                ret = exp.process.poll()
                if ret is not None:
                    exp.end_time = time.time()
                    exp.return_code = ret
                    if ret == 0:
                        exp.status = ExperimentStatus.COMPLETED
                    else:
                        exp.status = ExperimentStatus.FAILED
    
    def _check_external_completed(self):
        """Check if externally running experiments have completed."""
        for key, exp in self.experiments.items():
            if exp.status == ExperimentStatus.EXTERNAL_RUNNING:
                # Check if the process is still running
                persistent_status, status_info = get_experiment_status(
                    exp.dataset, exp.method, self.output_dir
                )
                if status_info:
                    if not status_info.pid or not is_process_running(status_info.pid):
                        # Process no longer running, check final status
                        if persistent_status == PersistentStatus.FINISHED:
                            exp.status = ExperimentStatus.COMPLETED
                            exp.end_time = status_info.end_time or time.time()
                        elif persistent_status == PersistentStatus.ERROR:
                            exp.status = ExperimentStatus.FAILED
                            exp.end_time = status_info.end_time or time.time()
                        else:
                            # Status file says running but process is dead
                            exp.status = ExperimentStatus.FAILED
                            exp.end_time = time.time()
                else:
                    # No status file found, mark as failed
                    exp.status = ExperimentStatus.FAILED
                    exp.end_time = time.time()
    
    def _get_external_running_count(self) -> int:
        """Get number of externally running experiments."""
        return sum(1 for exp in self.experiments.values() if exp.status == ExperimentStatus.EXTERNAL_RUNNING)
    
    def _get_running_count(self) -> int:
        """Get number of currently running experiments."""
        return sum(1 for exp in self.experiments.values() if exp.status == ExperimentStatus.RUNNING)
    
    def _get_stats(self) -> Dict[str, int]:
        """Get experiment statistics."""
        stats = {s.value: 0 for s in ExperimentStatus}
        for exp in self.experiments.values():
            stats[exp.status.value] += 1
        return stats
    
    def _kill_all_running(self):
        """Kill all running experiment processes."""
        for exp in self.experiments.values():
            if exp.status == ExperimentStatus.RUNNING and exp.process:
                try:
                    exp.process.terminate()
                    exp.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    exp.process.kill()
                except Exception:
                    pass
                exp.status = ExperimentStatus.CANCELLED
                exp.end_time = time.time()
    
    def _cancel_pending(self):
        """Mark all pending experiments as cancelled."""
        for key in self.pending_queue:
            exp = self.experiments[key]
            if exp.status == ExperimentStatus.PENDING:
                exp.status = ExperimentStatus.CANCELLED
        self.pending_queue.clear()
    
    def _detach_from_experiments(self):
        """Detach from running experiments without killing them.
        
        This allows the user to exit the orchestrator while experiments continue
        running in the background. When the orchestrator is started again, it will
        detect these experiments via their status files and reconnect.
        """
        running_count = 0
        for exp in self.experiments.values():
            if exp.status == ExperimentStatus.RUNNING and exp.process:
                # Don't kill the process, just detach
                # The process will continue running independently
                running_count += 1
        
        if running_count > 0:
            print(f"\n{Colors.YELLOW}Detaching from {running_count} running experiment(s).{Colors.RESET}")
            print(f"{Colors.DIM}Experiments will continue running in the background.{Colors.RESET}")
            print(f"{Colors.DIM}Run this script again to reconnect and monitor progress.{Colors.RESET}")
    
    def _generate_interim_report(self):
        """Generate an interim report when detaching."""
        # Compile current results
        results = {
            'success': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.COMPLETED],
            'failed': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.FAILED],
            'skipped': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.SKIPPED],
            'cancelled': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.CANCELLED],
        }
        
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        report_path = REPO_ROOT / 'report.md'
        
        # Generate report with a note that it's interim
        report_lines = []
        report_lines.append("# Experiment Run Report (INTERIM)")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Status:** Detached - experiments still running in background")
        report_lines.append(f"**Duration so far:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        report_lines.append("")
        
        # Count running experiments
        running = sum(1 for exp in self.experiments.values() if exp.status in (ExperimentStatus.RUNNING, ExperimentStatus.EXTERNAL_RUNNING))
        pending = sum(1 for exp in self.experiments.values() if exp.status == ExperimentStatus.PENDING)
        
        report_lines.append(f"**Still running:** {running} experiments")
        report_lines.append(f"**Pending:** {pending} experiments")
        report_lines.append("")
        
        # Rest of the report
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(f"- ✓ **Successful:** {len(results['success'])}")
        report_lines.append(f"- ✗ **Failed:** {len(results['failed'])}")
        report_lines.append(f"- ⊘ **Skipped:** {len(results['skipped'])}")
        report_lines.append(f"- ⊗ **Cancelled:** {len(results['cancelled'])}")
        report_lines.append("")
        
        # Include successful and failed sections (similar to main report)
        if results['success']:
            report_lines.append("## ✓ Successful Runs")
            report_lines.append("")
            for exp_key in sorted(results['success']):
                exp = self.experiments.get(exp_key)
                if exp and exp.end_time:
                    completed_time = datetime.fromtimestamp(exp.end_time).strftime('%Y-%m-%d %H:%M:%S')
                    report_lines.append(f"- `{exp_key}` - {exp.elapsed_str} - {completed_time}")
            report_lines.append("")
        
        if results['failed']:
            report_lines.append("## ✗ Failed Runs")
            report_lines.append("")
            for exp_key in sorted(results['failed']):
                exp = self.experiments.get(exp_key)
                if exp:
                    error_msg = exp.last_log_line if exp.last_log_line else f"Exit code: {exp.return_code}"
                    report_lines.append(f"- `{exp_key}` - {error_msg[:100]}")
            report_lines.append("")
        
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"{Colors.GREEN}✓ Interim report saved to {report_path}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to save interim report: {e}{Colors.RESET}")
    
    def _format_status_line(self, exp: Experiment, width: int = 80) -> str:
        """Format a single experiment status line."""
        status_colors = {
            ExperimentStatus.PENDING: Colors.DIM,
            ExperimentStatus.RUNNING: Colors.CYAN,
            ExperimentStatus.EXTERNAL_RUNNING: Colors.BLUE,
            ExperimentStatus.COMPLETED: Colors.GREEN,
            ExperimentStatus.FAILED: Colors.RED,
            ExperimentStatus.SKIPPED: Colors.YELLOW,
            ExperimentStatus.CANCELLED: Colors.MAGENTA,
        }
        
        status_symbols = {
            ExperimentStatus.PENDING: '○',
            ExperimentStatus.RUNNING: self.spinner_chars[self.spinner_idx],
            ExperimentStatus.EXTERNAL_RUNNING: '⟐',  # Different symbol for external
            ExperimentStatus.COMPLETED: '✓',
            ExperimentStatus.FAILED: '✗',
            ExperimentStatus.SKIPPED: '⊘',
            ExperimentStatus.CANCELLED: '⊗',
        }
        
        color = status_colors.get(exp.status, Colors.WHITE)
        symbol = status_symbols.get(exp.status, '?')
        
        # Format: [symbol] dataset/method (elapsed) - last_log
        key_part = f"{symbol} {exp.key}"
        time_part = f"({exp.elapsed_str})" if exp.status in (ExperimentStatus.RUNNING, ExperimentStatus.EXTERNAL_RUNNING) or exp.end_time else ""
        
        # Add PID for external processes
        if exp.status == ExperimentStatus.EXTERNAL_RUNNING and exp.external_pid:
            time_part += f" [PID:{exp.external_pid}]"
        
        base = f"{color}{key_part}{Colors.RESET} {time_part}"
        
        if exp.status in (ExperimentStatus.RUNNING, ExperimentStatus.EXTERNAL_RUNNING) and exp.last_log_line:
            # Truncate log line to fit
            max_log_len = width - len(exp.key) - 35
            log_line = exp.last_log_line[:max_log_len] if len(exp.last_log_line) > max_log_len else exp.last_log_line
            base += f" {Colors.DIM}- {log_line}{Colors.RESET}"
        
        return base
    
    def _clear_previous_display(self):
        """Clear the previous display by moving cursor up and clearing lines."""
        if self.last_display_lines > 0:
            try:
                # Move cursor up N lines and clear each
                sys.stdout.write(f'\033[{self.last_display_lines}A')
                for _ in range(self.last_display_lines):
                    sys.stdout.write('\033[2K\033[1B')  # Clear line and move down
                sys.stdout.write(f'\033[{self.last_display_lines}A')  # Move back up
                sys.stdout.flush()
            except (BlockingIOError, IOError):
                # Terminal buffer full, skip clearing
                pass
    
    def _print_status(self):
        """Print current status of all experiments."""
        stats = self._get_stats()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Get terminal width
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        
        lines = []
        
        # Header
        lines.append(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")
        
        status_flags = []
        if self.paused:
            status_flags.append(f"{Colors.YELLOW}PAUSED{Colors.RESET}")
        if self.soft_stop_requested:
            status_flags.append(f"{Colors.YELLOW}SOFT STOP{Colors.RESET}")
        if self.stop_requested:
            status_flags.append(f"{Colors.RED}STOPPING{Colors.RESET}")
        
        header = (f"{Colors.BOLD}EXPERIMENT RUNNER{Colors.RESET} | "
                 f"Elapsed: {elapsed:.0f}s | "
                 f"Workers: {self._get_running_count()}/{self.max_workers}")
        if status_flags:
            header += " | " + " ".join(status_flags)
        lines.append(header)
        
        # System resources line
        cpu_pct, mem_used, mem_total, mem_pct = get_system_stats()
        
        # Color code based on usage levels
        cpu_color = Colors.GREEN if cpu_pct < 70 else (Colors.YELLOW if cpu_pct < 90 else Colors.RED)
        mem_color = Colors.GREEN if mem_pct < 70 else (Colors.YELLOW if mem_pct < 90 else Colors.RED)
        
        resource_line = (f"CPU: {cpu_color}{cpu_pct:5.1f}%{Colors.RESET} | "
                        f"RAM: {mem_color}{mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%){Colors.RESET}")
        lines.append(resource_line)
        
        # Get global stats from status files
        global_stats = get_global_stats(self.output_dir)
        
        # Session stats bar (this orchestrator's view)
        stats_line = (f"{Colors.GREEN}Done: {stats['completed']}{Colors.RESET} | "
                     f"{Colors.RED}Failed: {stats['failed']}{Colors.RESET} | "
                     f"{Colors.CYAN}Running: {stats['running']}{Colors.RESET} | "
                     f"{Colors.BLUE}External: {stats.get('external_running', 0)}{Colors.RESET} | "
                     f"{Colors.DIM}Pending: {stats['pending']}{Colors.RESET} | "
                     f"{Colors.YELLOW}Skipped: {stats['skipped']}{Colors.RESET} | "
                     f"{Colors.MAGENTA}Cancelled: {stats['cancelled']}{Colors.RESET}")
        lines.append(stats_line)
        
        # Global stats line (all experiments ever)
        global_line = (f"{Colors.DIM}Global: "
                      f"Finished: {global_stats['finished']} | "
                      f"Running: {global_stats['running']} | "
                      f"Error: {global_stats['error']} | "
                      f"Total tracked: {global_stats['total']}{Colors.RESET}")
        lines.append(global_line)
        lines.append(f"{Colors.DIM}{'─' * width}{Colors.RESET}")
        
        # Running experiments (include both local and external)
        running_exps = [exp for exp in self.experiments.values() 
                       if exp.status in (ExperimentStatus.RUNNING, ExperimentStatus.EXTERNAL_RUNNING)]
        if running_exps:
            lines.append(f"{Colors.BOLD}Running:{Colors.RESET}")
            for exp in running_exps:
                lines.append(f"  {self._format_status_line(exp, width - 2)}")
        else:
            lines.append(f"{Colors.DIM}No experiments running{Colors.RESET}")
        
        # Recently completed (last 3)
        completed_exps = sorted(
            [exp for exp in self.experiments.values() if exp.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)],
            key=lambda e: e.end_time or 0,
            reverse=True
        )[:3]
        if completed_exps:
            lines.append(f"{Colors.BOLD}Recent:{Colors.RESET}")
            for exp in completed_exps:
                lines.append(f"  {self._format_status_line(exp, width - 2)}")
        
        # Controls hint
        lines.append(f"{Colors.DIM}{'─' * width}{Colors.RESET}")
        lines.append(f"{Colors.DIM}[Ctrl+C/d] detach  [q] kill all  [s] soft-stop  [p] pause  [r] refresh{Colors.RESET}")
        
        # Clear previous and print new
        self._clear_previous_display()
        
        output = '\n'.join(lines)
        
        # Handle potential BlockingIOError when terminal buffer is full
        try:
            print(output)
            sys.stdout.flush()
        except BlockingIOError:
            # Terminal buffer full, skip this update
            pass
        except IOError:
            # Other IO errors, skip update
            pass
        
        self.last_display_lines = len(lines)
    
    def _handle_input(self) -> Optional[str]:
        """Check for keyboard input (non-blocking)."""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        except Exception:
            pass
        return None
    
    def run(self) -> Dict[str, List[str]]:
        """Run all experiments with interactive control."""
        self.start_time = time.time()
        
        # Set up terminal for raw input
        old_settings = None
        old_flags = None
        is_tty = sys.stdin.isatty()
        
        if is_tty:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
                fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
            except Exception:
                is_tty = False
        
        # Set up signal handler for Ctrl+C (detach, not kill)
        def signal_handler(signum, frame):
            self.detach_requested = True
        
        old_signal = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Initial display
            print()  # Blank line before status
            self._print_status()
            
            while True:
                # Check for user input
                if is_tty:
                    key = self._handle_input()
                    if key:
                        if key == 'q':
                            self.stop_requested = True  # Hard quit
                        elif key == 'd':
                            self.detach_requested = True  # Detach
                        elif key == 's':
                            self.soft_stop_requested = True
                            self._cancel_pending()
                        elif key == 'p':
                            self.paused = not self.paused
                        # 'r' just triggers refresh below
                
                # Check hard stop flag (kills experiments)
                if self.stop_requested:
                    self._kill_all_running()
                    self._cancel_pending()
                    break
                
                # Check detach flag (exit but leave experiments running)
                if self.detach_requested:
                    self._detach_from_experiments()
                    # Generate interim report before detaching
                    self._generate_interim_report()
                    break
                
                # Check for completed experiments (local processes)
                self._check_completed()
                
                # Check for completed external experiments
                self._check_external_completed()
                
                # Start new experiments if we have capacity (and not in monitor mode)
                running_count = self._get_running_count()
                while (running_count < self.max_workers 
                       and self.pending_queue 
                       and not self.stop_requested 
                       and not self.soft_stop_requested
                       and not self.paused
                       and not self.monitor_only):
                    
                    key = self.pending_queue.pop(0)
                    self._start_experiment(key)
                    running_count += 1
                
                # Check if all done (both local and external)
                stats = self._get_stats()
                external_running = stats.get('external_running', 0)
                if stats['running'] == 0 and stats['pending'] == 0 and external_running == 0:
                    # In monitor mode, keep running to watch for new experiments
                    if not self.monitor_only:
                        break
                
                # Update spinner
                self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
                
                # Update display
                self._print_status()
                
                # Small delay
                time.sleep(0.3)
            
        finally:
            # Restore signal handler
            signal.signal(signal.SIGINT, old_signal)
            
            # Restore terminal settings
            if old_settings:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
            if old_flags is not None:
                try:
                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags)
                except Exception:
                    pass
        
        # Final status
        print()
        self.last_display_lines = 0
        self._print_status()
        
        # Compile results
        results = {
            'success': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.COMPLETED],
            'failed': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.FAILED],
            'skipped': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.SKIPPED],
            'cancelled': [exp.key for exp in self.experiments.values() if exp.status == ExperimentStatus.CANCELLED],
        }
        
        return results


def get_all_datasets(configs_dir: pathlib.Path) -> List[str]:
    """Get all dataset names from the configs directory."""
    datasets = []
    for item in configs_dir.iterdir():
        if item.is_dir() and item.name not in EXCLUDED_FILES:
            config_file = item / 'config.yaml'
            if config_file.exists():
                datasets.append(item.name)
    return sorted(datasets)


def check_existing_output(dataset: str, method: str, output_dir: pathlib.Path) -> bool:
    """Check if output already exists for a dataset/method combination."""
    expected_output = output_dir / f"{dataset}_{method}"
    return expected_output.exists() and any(expected_output.iterdir())


def run_experiment_simple(
    dataset: str,
    method: str,
    verbose: bool = False,
    offline: bool = False,
    overrides: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run a single experiment (simple sequential mode)."""
    experiment_key = f"{dataset}/{method}"
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'run_experiment.py'),
        '--dataset', dataset,
        '--method', method,
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    if offline:
        cmd.append('--offline')
    
    if overrides:
        for override in overrides:
            cmd.extend(['--set', override])
    
    cmd_str = ' '.join(cmd)
    
    if dry_run:
        return {
            'success': True,
            'message': "Dry run - command not executed",
            'experiment_key': experiment_key,
            'cmd': cmd_str
        }
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'message': f"Completed in {elapsed_time:.1f}s" if result.returncode == 0 else f"Failed: {result.stderr[:200]}",
            'experiment_key': experiment_key,
            'elapsed_time': elapsed_time,
            'cmd': cmd_str
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Exception: {str(e)}",
            'experiment_key': experiment_key,
            'cmd': cmd_str
        }


def generate_report(experiments_dict: Dict[str, Experiment], results: Dict[str, List[str]], total_elapsed: float, output_path: pathlib.Path):
    """Generate a markdown report summarizing experiment results."""
    report_lines = []
    
    # Header
    report_lines.append("# Experiment Run Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Duration:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- ✓ **Successful:** {len(results['success'])}")
    report_lines.append(f"- ✗ **Failed:** {len(results['failed'])}")
    report_lines.append(f"- ⊘ **Skipped:** {len(results['skipped'])}")
    if 'cancelled' in results:
        report_lines.append(f"- ⊗ **Cancelled:** {len(results['cancelled'])}")
    report_lines.append(f"- **Total:** {len(results['success']) + len(results['failed']) + len(results['skipped']) + len(results.get('cancelled', []))}")
    report_lines.append("")
    
    # Successful runs
    if results['success']:
        report_lines.append("## ✓ Successful Runs")
        report_lines.append("")
        report_lines.append("| Experiment | Duration | Completed At |")
        report_lines.append("|------------|----------|--------------|")
        
        for exp_key in sorted(results['success']):
            exp = experiments_dict.get(exp_key)
            if exp:
                duration = exp.elapsed_str
                if exp.end_time:
                    completed_time = datetime.fromtimestamp(exp.end_time).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    completed_time = "N/A"
                report_lines.append(f"| `{exp_key}` | {duration} | {completed_time} |")
            else:
                report_lines.append(f"| `{exp_key}` | N/A | N/A |")
        report_lines.append("")
    
    # Failed runs
    if results['failed']:
        report_lines.append("## ✗ Failed Runs")
        report_lines.append("")
        report_lines.append("| Experiment | Duration | Error/Last Log |")
        report_lines.append("|------------|----------|----------------|")
        
        for exp_key in sorted(results['failed']):
            exp = experiments_dict.get(exp_key)
            if exp:
                duration = exp.elapsed_str
                error_msg = exp.last_log_line if exp.last_log_line else f"Exit code: {exp.return_code}"
                # Truncate long error messages
                if len(error_msg) > 100:
                    error_msg = error_msg[:97] + "..."
                # Escape pipe characters in error messages
                error_msg = error_msg.replace('|', '\|')
                report_lines.append(f"| `{exp_key}` | {duration} | {error_msg} |")
            else:
                report_lines.append(f"| `{exp_key}` | N/A | Unknown error |")
        report_lines.append("")
    
    # Skipped runs
    if results['skipped']:
        report_lines.append("## ⊘ Skipped Runs")
        report_lines.append("")
        report_lines.append("These experiments were skipped (already completed in previous runs):")
        report_lines.append("")
        for exp_key in sorted(results['skipped']):
            report_lines.append(f"- `{exp_key}`")
        report_lines.append("")
    
    # Cancelled runs
    if results.get('cancelled'):
        report_lines.append("## ⊗ Cancelled Runs")
        report_lines.append("")
        report_lines.append("These experiments were cancelled during execution:")
        report_lines.append("")
        for exp_key in sorted(results['cancelled']):
            report_lines.append(f"- `{exp_key}`")
        report_lines.append("")
    
    # Write report
    try:
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        return True
    except Exception as e:
        print(f"Warning: Failed to write report to {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for all datasets with all methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to run (default: all datasets in configs/)')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                       help=f'Methods to run (default: {DEFAULT_METHODS})')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments that completed successfully (based on status files)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for each experiment')
    parser.add_argument('--offline', action='store_true',
                       help='Run WandB in offline mode')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit the number of datasets to process')
    parser.add_argument('--set', action='append', default=[], dest='overrides',
                       help='Override config values for all experiments')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue running other experiments even if one fails')
    parser.add_argument('--parallel', type=int, default=None, metavar='N',
                       help='Run N experiments in parallel with interactive CLI')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor mode: only show running experiments, do not start new ones')
    parser.add_argument('--priority-only', action='store_true',
                       help='Only run experiments for datasets in priority_datasets list from config.yaml')
    
    args = parser.parse_args()
    
    configs_dir = REPO_ROOT / 'configs'
    output_dir = REPO_ROOT / 'outputs'
    
    # Load global config to get excluded datasets and priority datasets
    config_file = REPO_ROOT / 'configs' / 'config.yaml'
    excluded_datasets = []
    priority_datasets = []
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                excluded_datasets = config.get('excluded_datasets', [])
                priority_datasets = config.get('priority_datasets', [])
        except Exception:
            # If config loading fails, just use empty list
            excluded_datasets = []
            priority_datasets = []
    
    # Get datasets
    if args.datasets:
        datasets = args.datasets
        all_datasets = set(get_all_datasets(configs_dir))
        invalid_datasets = set(datasets) - all_datasets
        if invalid_datasets:
            print(f"ERROR: Invalid datasets: {invalid_datasets}")
            print(f"Available datasets: {sorted(all_datasets)}")
            return 1
    else:
        datasets = get_all_datasets(configs_dir)
    
    # Filter for priority datasets if --priority-only flag is set
    if args.priority_only:
        if not priority_datasets:
            print("ERROR: --priority-only flag used but no priority_datasets defined in config.yaml")
            return 1
        priority_set = set(priority_datasets)
        original_count = len(datasets)
        datasets = [d for d in datasets if d in priority_set]
        filtered_count = original_count - len(datasets)
        if filtered_count > 0:
            print(f"Priority mode: running only {len(datasets)} priority dataset(s): {sorted(datasets)}")
            print(f"Skipped {filtered_count} non-priority dataset(s)")
        if not datasets:
            print("ERROR: No valid priority datasets found")
            return 1
    
    # Filter out excluded datasets
    if excluded_datasets:
        excluded_set = set(excluded_datasets)
        original_count = len(datasets)
        datasets = [d for d in datasets if d not in excluded_set]
        filtered_count = original_count - len(datasets)
        if filtered_count > 0:
            print(f"Excluded {filtered_count} dataset(s): {sorted(excluded_set & set(datasets + excluded_datasets))}")
    
    if args.limit:
        datasets = datasets[:args.limit]
    
    methods = args.methods
    
    # Build experiments list with iris first (baseline/smallest dataset)
    # Then randomize remaining datasets, keeping all methods together per dataset
    import random
    
    experiments = []
    
    # Add iris experiments first if iris is in the datasets list
    if 'iris' in datasets:
        for method in methods:
            experiments.append({'dataset': 'iris', 'method': method})
    
    # Get remaining datasets (excluding iris) and shuffle them randomly
    remaining_datasets = [d for d in datasets if d != 'iris']
    random.shuffle(remaining_datasets)
    
    # Add experiments for each shuffled dataset with all methods together
    for dataset in remaining_datasets:
        for method in methods:
            experiments.append({'dataset': dataset, 'method': method})
    
    # Print header
    print("=" * 60)
    print("COUNTERFACTUAL EXPERIMENTS BATCH RUNNER")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: {methods}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Parallel workers: {args.parallel if args.parallel else 'sequential'}")
    print(f"Monitor only: {args.monitor}")
    if args.overrides:
        print(f"Config overrides: {args.overrides}")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Run experiments
    experiments_dict = {}  # Will store experiment objects for report generation
    
    if args.parallel and args.parallel > 0 or args.monitor:
        # Interactive parallel mode (or monitor mode)
        runner = ExperimentRunner(
            experiments=experiments,
            max_workers=args.parallel or 0,
            verbose=args.verbose,
            offline=args.offline,
            overrides=args.overrides,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            monitor_only=args.monitor,
        )
        
        # Store experiments for report generation
        experiments_dict = runner.experiments
        
        if args.dry_run:
            print("\n[DRY RUN MODE]")
            # Show status-aware info from the runner
            stats = runner._get_stats()
            print(f"  Pending (will run): {stats['pending']}")
            print(f"  Skipped (finished): {stats['skipped']}")
            print(f"  External running: {stats.get('external_running', 0)}")
            print(f"  Failed (will retry): {stats.get('failed', 0)}")
            print()
            for key, exp in runner.experiments.items():
                status_str = exp.status.value
                if exp.status == ExperimentStatus.PENDING:
                    print(f"  [RUN]  {key}")
                elif exp.status == ExperimentStatus.SKIPPED:
                    print(f"  [SKIP] {key} (already completed)")
                elif exp.status == ExperimentStatus.EXTERNAL_RUNNING:
                    print(f"  [EXT]  {key} (running externally, PID {exp.external_pid})")
                elif exp.status == ExperimentStatus.FAILED:
                    print(f"  [RETRY] {key} (previous run failed)")
            results = {'success': [], 'failed': [], 'skipped': [], 'cancelled': []}
        else:
            results = runner.run()
    else:
        # Sequential mode (original behavior)
        results = {'success': [], 'failed': [], 'skipped': []}
        
        # Track experiments for report generation
        for exp_dict in experiments:
            exp_obj = Experiment(dataset=exp_dict['dataset'], method=exp_dict['method'])
            experiments_dict[exp_obj.key] = exp_obj
        
        for i, exp in enumerate(experiments, 1):
            dataset = exp['dataset']
            method = exp['method']
            experiment_key = f"{dataset}/{method}"
            exp_obj = experiments_dict[experiment_key]
            
            print(f"\n[{i}/{len(experiments)}] {experiment_key}")
            print("-" * 40)
            
            # Use status-based check
            status, _ = get_experiment_status(dataset, method, output_dir)
            if args.skip_existing and status == PersistentStatus.FINISHED:
                print(f"  Skipping (completed successfully)")
                exp_obj.status = ExperimentStatus.SKIPPED
                results['skipped'].append(experiment_key)
                continue
            
            print(f"  Running...")
            exp_obj.start_time = time.time()
            result = run_experiment_simple(
                dataset=dataset,
                method=method,
                verbose=args.verbose,
                offline=args.offline,
                overrides=args.overrides,
                dry_run=args.dry_run
            )
            exp_obj.end_time = time.time()
            
            if result['success']:
                print(f"  ✓ {result['message']}")
                exp_obj.status = ExperimentStatus.COMPLETED
                results['success'].append(experiment_key)
            else:
                print(f"  ✗ {result['message']}")
                exp_obj.status = ExperimentStatus.FAILED
                exp_obj.last_log_line = result.get('message', 'Unknown error')
                results['failed'].append(experiment_key)
                
                if not args.continue_on_error and not args.dry_run:
                    print("\nERROR: Stopping. Use --continue-on-error to keep going.")
                    break
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")
    if 'cancelled' in results:
        print(f"⊗ Cancelled: {len(results['cancelled'])}")
    
    if results['failed']:
        print("\nFailed experiments:")
        for exp in results['failed']:
            print(f"  - {exp}")
    
    print("=" * 60)
    
    # Generate report.md
    if not args.dry_run:
        report_path = REPO_ROOT / 'report.md'
        print(f"\nGenerating report: {report_path}")
        if generate_report(experiments_dict, results, total_elapsed, report_path):
            print(f"✓ Report saved to {report_path}")
        else:
            print(f"✗ Failed to save report")
    
    return 1 if results['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
