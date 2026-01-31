"""Experiment status tracking utilities.

This module provides utilities for tracking experiment status across runs,
allowing the orchestrator to:
- Skip only completed experiments (not failed or running)
- Reconnect to running experiments
- Show global experiment statistics
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import fcntl


class PersistentStatus(Enum):
    """Persistent experiment status (saved to disk)."""
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class ExperimentStatusInfo:
    """Information about an experiment's status."""
    dataset: str
    method: str
    status: str  # PersistentStatus value
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentStatusInfo':
        return cls(**data)


def get_status_dir(output_dir: pathlib.Path) -> pathlib.Path:
    """Get the directory for status files."""
    status_dir = output_dir / '.experiment_status'
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir


def get_status_file_path(dataset: str, method: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Get the path to the status file for an experiment."""
    status_dir = get_status_dir(output_dir)
    return status_dir / f"{dataset}_{method}.status.json"


def get_log_file_path(dataset: str, method: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Get the path to the log file for an experiment."""
    status_dir = get_status_dir(output_dir)
    return status_dir / f"{dataset}_{method}.log"


def write_status(
    dataset: str,
    method: str,
    status: PersistentStatus,
    output_dir: pathlib.Path,
    pid: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Write experiment status to file with file locking."""
    status_file = get_status_file_path(dataset, method, output_dir)
    log_file = get_log_file_path(dataset, method, output_dir)
    
    info = ExperimentStatusInfo(
        dataset=dataset,
        method=method,
        status=status.value,
        pid=pid,
        start_time=start_time,
        end_time=end_time,
        error_message=error_message,
        log_file=str(log_file),
    )
    
    # Write with file locking to prevent race conditions
    with open(status_file, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(info.to_dict(), f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_status(dataset: str, method: str, output_dir: pathlib.Path) -> Optional[ExperimentStatusInfo]:
    """Read experiment status from file."""
    status_file = get_status_file_path(dataset, method, output_dir)
    
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return ExperimentStatusInfo.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
        return True
    except OSError:
        return False


def get_experiment_status(
    dataset: str, 
    method: str, 
    output_dir: pathlib.Path
) -> tuple[Optional[PersistentStatus], Optional[ExperimentStatusInfo]]:
    """Get the current status of an experiment.
    
    Returns:
        Tuple of (status, info) where status is None if no status file exists,
        or the actual status. If status file says RUNNING but process is dead,
        returns ERROR status.
    """
    info = read_status(dataset, method, output_dir)
    
    if info is None:
        return None, None
    
    status = PersistentStatus(info.status)
    
    # If marked as running, verify the process is actually running
    if status == PersistentStatus.RUNNING:
        if info.pid and not is_process_running(info.pid):
            # Process died without updating status - mark as error
            return PersistentStatus.ERROR, info
    
    return status, info


def get_all_experiment_statuses(output_dir: pathlib.Path) -> Dict[str, ExperimentStatusInfo]:
    """Get status info for all experiments that have status files.
    
    Returns:
        Dict mapping "dataset/method" to ExperimentStatusInfo
    """
    status_dir = get_status_dir(output_dir)
    statuses = {}
    
    if not status_dir.exists():
        return statuses
    
    for status_file in status_dir.glob("*.status.json"):
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
            info = ExperimentStatusInfo.from_dict(data)
            key = f"{info.dataset}/{info.method}"
            
            # Verify running status
            if info.status == PersistentStatus.RUNNING.value:
                if info.pid and not is_process_running(info.pid):
                    info.status = PersistentStatus.ERROR.value
                    info.error_message = "Process terminated unexpectedly"
            
            statuses[key] = info
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    return statuses


def get_global_stats(output_dir: pathlib.Path) -> Dict[str, int]:
    """Get global statistics across all experiments.
    
    Returns:
        Dict with counts for each status: running, finished, error
    """
    statuses = get_all_experiment_statuses(output_dir)
    
    stats = {
        'running': 0,
        'finished': 0,
        'error': 0,
        'total': len(statuses),
    }
    
    for info in statuses.values():
        if info.status == PersistentStatus.RUNNING.value:
            # Double-check if actually running
            if info.pid and is_process_running(info.pid):
                stats['running'] += 1
            else:
                stats['error'] += 1
        elif info.status == PersistentStatus.FINISHED.value:
            stats['finished'] += 1
        elif info.status == PersistentStatus.ERROR.value:
            stats['error'] += 1
    
    return stats


def read_log_tail(dataset: str, method: str, output_dir: pathlib.Path, lines: int = 50) -> List[str]:
    """Read the last N lines from an experiment's log file."""
    log_file = get_log_file_path(dataset, method, output_dir)
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
    except Exception:
        return []


def append_log(dataset: str, method: str, output_dir: pathlib.Path, line: str) -> None:
    """Append a line to the experiment's log file."""
    log_file = get_log_file_path(dataset, method, output_dir)
    
    with open(log_file, 'a') as f:
        f.write(line + '\n')


def clear_log(dataset: str, method: str, output_dir: pathlib.Path) -> None:
    """Clear the experiment's log file."""
    log_file = get_log_file_path(dataset, method, output_dir)
    
    # Truncate the file
    with open(log_file, 'w') as f:
        pass
