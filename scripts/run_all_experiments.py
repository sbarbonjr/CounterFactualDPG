#!/usr/bin/env python3
"""Run experiments for all datasets with all methods with interactive CLI.

This script iterates through all dataset directories in configs/ and runs
experiments with each specified method (dpg, dice).

Interactive Controls (when running with --parallel):
  q / Ctrl+C  - Hard stop: Kill all running experiments immediately
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
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


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
        skip_existing: bool = False
    ):
        self.max_workers = max_workers
        self.verbose = verbose
        self.offline = offline
        self.overrides = overrides or []
        self.output_dir = output_dir or REPO_ROOT / 'outputs'
        self.skip_existing = skip_existing
        
        # Initialize experiments
        self.experiments: Dict[str, Experiment] = {}
        self.pending_queue: List[str] = []
        
        for exp_dict in experiments:
            exp = Experiment(dataset=exp_dict['dataset'], method=exp_dict['method'])
            self.experiments[exp.key] = exp
            
            # Check if should skip
            if skip_existing and self._check_existing_output(exp.dataset, exp.method):
                exp.status = ExperimentStatus.SKIPPED
            else:
                self.pending_queue.append(exp.key)
        
        # Control flags
        self.stop_requested = False
        self.soft_stop_requested = False
        self.paused = False
        
        # Thread management
        self.log_queue: queue.Queue = queue.Queue()
        self.reader_threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        
        # Spinner animation
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_idx = 0
        
        # Stats
        self.start_time: Optional[float] = None
        
        # Display state
        self.last_display_lines = 0
    
    def _check_existing_output(self, dataset: str, method: str) -> bool:
        """Check if output already exists for a dataset/method combination."""
        expected_output = self.output_dir / f"{dataset}_{method}"
        return expected_output.exists() and any(expected_output.iterdir())
    
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
    
    def _format_status_line(self, exp: Experiment, width: int = 80) -> str:
        """Format a single experiment status line."""
        status_colors = {
            ExperimentStatus.PENDING: Colors.DIM,
            ExperimentStatus.RUNNING: Colors.CYAN,
            ExperimentStatus.COMPLETED: Colors.GREEN,
            ExperimentStatus.FAILED: Colors.RED,
            ExperimentStatus.SKIPPED: Colors.YELLOW,
            ExperimentStatus.CANCELLED: Colors.MAGENTA,
        }
        
        status_symbols = {
            ExperimentStatus.PENDING: '○',
            ExperimentStatus.RUNNING: self.spinner_chars[self.spinner_idx],
            ExperimentStatus.COMPLETED: '✓',
            ExperimentStatus.FAILED: '✗',
            ExperimentStatus.SKIPPED: '⊘',
            ExperimentStatus.CANCELLED: '⊗',
        }
        
        color = status_colors.get(exp.status, Colors.WHITE)
        symbol = status_symbols.get(exp.status, '?')
        
        # Format: [symbol] dataset/method (elapsed) - last_log
        key_part = f"{symbol} {exp.key}"
        time_part = f"({exp.elapsed_str})" if exp.status == ExperimentStatus.RUNNING or exp.end_time else ""
        
        base = f"{color}{key_part}{Colors.RESET} {time_part}"
        
        if exp.status == ExperimentStatus.RUNNING and exp.last_log_line:
            # Truncate log line to fit
            max_log_len = width - len(exp.key) - 25
            log_line = exp.last_log_line[:max_log_len] if len(exp.last_log_line) > max_log_len else exp.last_log_line
            base += f" {Colors.DIM}- {log_line}{Colors.RESET}"
        
        return base
    
    def _clear_previous_display(self):
        """Clear the previous display by moving cursor up and clearing lines."""
        if self.last_display_lines > 0:
            # Move cursor up N lines and clear each
            sys.stdout.write(f'\033[{self.last_display_lines}A')
            for _ in range(self.last_display_lines):
                sys.stdout.write('\033[2K\033[1B')  # Clear line and move down
            sys.stdout.write(f'\033[{self.last_display_lines}A')  # Move back up
            sys.stdout.flush()
    
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
        
        # Stats bar
        stats_line = (f"{Colors.GREEN}✓ {stats['completed']}{Colors.RESET} | "
                     f"{Colors.RED}✗ {stats['failed']}{Colors.RESET} | "
                     f"{Colors.CYAN}⟳ {stats['running']}{Colors.RESET} | "
                     f"{Colors.DIM}○ {stats['pending']}{Colors.RESET} | "
                     f"{Colors.YELLOW}⊘ {stats['skipped']}{Colors.RESET} | "
                     f"{Colors.MAGENTA}⊗ {stats['cancelled']}{Colors.RESET}")
        lines.append(stats_line)
        lines.append(f"{Colors.DIM}{'─' * width}{Colors.RESET}")
        
        # Running experiments (always show)
        running_exps = [exp for exp in self.experiments.values() if exp.status == ExperimentStatus.RUNNING]
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
        lines.append(f"{Colors.DIM}[q] quit  [s] soft-stop  [p] pause/resume  [r] refresh{Colors.RESET}")
        
        # Clear previous and print new
        self._clear_previous_display()
        
        output = '\n'.join(lines)
        print(output)
        sys.stdout.flush()
        
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
        
        # Set up signal handler for Ctrl+C
        def signal_handler(signum, frame):
            self.stop_requested = True
        
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
                            self.stop_requested = True
                        elif key == 's':
                            self.soft_stop_requested = True
                            self._cancel_pending()
                        elif key == 'p':
                            self.paused = not self.paused
                        # 'r' just triggers refresh below
                
                # Check stop flag (from signal or key)
                if self.stop_requested:
                    self._kill_all_running()
                    self._cancel_pending()
                    break
                
                # Check for completed experiments
                self._check_completed()
                
                # Start new experiments if we have capacity
                running_count = self._get_running_count()
                while (running_count < self.max_workers 
                       and self.pending_queue 
                       and not self.stop_requested 
                       and not self.soft_stop_requested
                       and not self.paused):
                    
                    key = self.pending_queue.pop(0)
                    self._start_experiment(key)
                    running_count += 1
                
                # Check if all done
                stats = self._get_stats()
                if stats['running'] == 0 and stats['pending'] == 0:
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
                       help='Skip dataset/method combinations that already have output')
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
    
    args = parser.parse_args()
    
    configs_dir = REPO_ROOT / 'configs'
    output_dir = REPO_ROOT / 'outputs'
    
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
    
    if args.limit:
        datasets = datasets[:args.limit]
    
    methods = args.methods
    
    # Build experiments list
    experiments = [
        {'dataset': dataset, 'method': method}
        for dataset in datasets
        for method in methods
    ]
    
    # Print header
    print("=" * 60)
    print("COUNTERFACTUAL EXPERIMENTS BATCH RUNNER")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: {methods}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Parallel workers: {args.parallel if args.parallel else 'sequential'}")
    if args.overrides:
        print(f"Config overrides: {args.overrides}")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Run experiments
    if args.parallel and args.parallel > 0:
        # Interactive parallel mode
        runner = ExperimentRunner(
            experiments=experiments,
            max_workers=args.parallel,
            verbose=args.verbose,
            offline=args.offline,
            overrides=args.overrides,
            output_dir=output_dir,
            skip_existing=args.skip_existing
        )
        
        if args.dry_run:
            print("\n[DRY RUN MODE]")
            for exp in experiments:
                print(f"  Would run: {exp['dataset']}/{exp['method']}")
            results = {'success': [f"{e['dataset']}/{e['method']}" for e in experiments], 
                      'failed': [], 'skipped': [], 'cancelled': []}
        else:
            results = runner.run()
    else:
        # Sequential mode (original behavior)
        results = {'success': [], 'failed': [], 'skipped': []}
        
        for i, exp in enumerate(experiments, 1):
            dataset = exp['dataset']
            method = exp['method']
            experiment_key = f"{dataset}/{method}"
            
            print(f"\n[{i}/{len(experiments)}] {experiment_key}")
            print("-" * 40)
            
            if args.skip_existing and check_existing_output(dataset, method, output_dir):
                print(f"  Skipping (output exists)")
                results['skipped'].append(experiment_key)
                continue
            
            print(f"  Running...")
            result = run_experiment_simple(
                dataset=dataset,
                method=method,
                verbose=args.verbose,
                offline=args.offline,
                overrides=args.overrides,
                dry_run=args.dry_run
            )
            
            if result['success']:
                print(f"  ✓ {result['message']}")
                results['success'].append(experiment_key)
            else:
                print(f"  ✗ {result['message']}")
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
    
    return 1 if results['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
