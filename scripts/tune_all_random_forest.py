#!/usr/bin/env python
"""Run RandomForest hyperparameter tuning on all datasets.

This script iterates through all datasets in the configs/ directory
and runs the tune_random_forest.py script for each one.

Usage:
    python scripts/tune_all_random_forest.py
    python scripts/tune_all_random_forest.py --n-iter 50
    python scripts/tune_all_random_forest.py --cv 10 --scoring f1_weighted
    python scripts/tune_all_random_forest.py --start-from iris
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import List


def get_available_datasets(configs_dir: pathlib.Path) -> List[str]:
    """Find all subdirectories in configs/ that contain a config.yaml file.
    
    Args:
        configs_dir: Path to the configs directory
        
    Returns:
        List of dataset names (subdirectory names)
    """
    datasets = []
    for item in configs_dir.iterdir():
        if item.is_dir() and (item / 'config.yaml').exists():
            datasets.append(item.name)
    return sorted(datasets)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    The script accepts arguments that are passed through to tune_random_forest.py.
    The --dataset argument is excluded as it's determined automatically.
    """
    parser = argparse.ArgumentParser(
        description='Run RandomForest hyperparameter tuning on all datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tune_all_random_forest.py
    python scripts/tune_all_random_forest.py --n-iter 50
    python scripts/tune_all_random_forest.py --cv 10 --scoring f1_weighted
    python scripts/tune_all_random_forest.py --n-iter 100 --cv 5 --scoring accuracy --n-jobs -1
    python scripts/tune_all_random_forest.py --start-from iris
        """
    )
    
    parser.add_argument(
        '--start-from',
        type=str,
        help='Start processing from the specified dataset (alphabetically sorted)'
    )
    
    parser.add_argument(
        '--n-iter', '-n',
        type=int,
        help='Number of parameter settings sampled (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        help='Number of cross-validation folds (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--scoring', '-s',
        type=str,
        help='Scoring metric for optimization (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        help='Number of parallel jobs (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        help='Random state for reproducibility (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        help='Proportion of data to use as test set (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results (passed to tune_random_forest.py)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=int,
        help='Verbosity level for RandomizedSearchCV (passed to tune_random_forest.py)'
    )
    
    return parser.parse_args()


def build_command(dataset: str, args: argparse.Namespace) -> List[str]:
    """Build the command to run tune_random_forest.py for a dataset.
    
    Args:
        dataset: Name of the dataset
        args: Parsed command line arguments
        
    Returns:
        List of command arguments
    """
    cmd = ['python', 'scripts/tune_random_forest.py', '--dataset', dataset]
    
    # Add optional arguments if provided
    if args.n_iter is not None:
        cmd.extend(['--n-iter', str(args.n_iter)])
    if args.cv is not None:
        cmd.extend(['--cv', str(args.cv)])
    if args.scoring is not None:
        cmd.extend(['--scoring', args.scoring])
    if args.n_jobs is not None:
        cmd.extend(['--n-jobs', str(args.n_jobs)])
    if args.random_state is not None:
        cmd.extend(['--random-state', str(args.random_state)])
    if args.test_size is not None:
        cmd.extend(['--test-size', str(args.test_size)])
    if args.output is not None:
        cmd.extend(['--output', args.output])
    if args.verbose is not None:
        cmd.extend(['--verbose', str(args.verbose)])
    
    return cmd


def main() -> int:
    """Main function to run tuning on all datasets."""
    args = parse_args()
    
    # Get the repository root
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    configs_dir = repo_root / 'configs'
    
    # Find all available datasets
    datasets = get_available_datasets(configs_dir)
    
    if not datasets:
        print("No datasets found in configs/ directory!")
        return 1
    
    # Filter datasets if --start-from is provided
    if args.start_from is not None:
        if args.start_from not in datasets:
            print(f"Error: Dataset '{args.start_from}' not found in available datasets!")
            print(f"Available datasets: {', '.join(datasets)}")
            return 1
        start_index = datasets.index(args.start_from)
        datasets = datasets[start_index:]
        print(f"Starting from dataset: {args.start_from}")
    
    print("=" * 70)
    print("RandomForest Hyperparameter Tuning - All Datasets")
    print("=" * 70)
    print(f"Found {len(datasets)} dataset(s) to process:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    print("=" * 70)
    
    # Track results
    successful = []
    failed = []
    
    # Process each dataset sequentially
    for idx, dataset in enumerate(datasets, 1):
        print(f"\n[{idx}/{len(datasets)}] Processing dataset: {dataset}")
        print("-" * 70)
        
        # Build command
        cmd = build_command(dataset, args)
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run the tuning script
        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                check=True,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                successful.append(dataset)
                print(f"\n✓ Successfully completed: {dataset}")
            else:
                failed.append(dataset)
                print(f"\n✗ Failed: {dataset} (exit code: {result.returncode})")
                
        except subprocess.CalledProcessError as e:
            failed.append(dataset)
            print(f"\n✗ Failed: {dataset} - {e}")
        except Exception as e:
            failed.append(dataset)
            print(f"\n✗ Failed: {dataset} - Unexpected error: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful datasets:")
        for dataset in successful:
            print(f"  ✓ {dataset}")
    
    if failed:
        print("\nFailed datasets:")
        for dataset in failed:
            print(f"  ✗ {dataset}")
    
    print("=" * 70)
    
    # Return exit code based on whether all datasets succeeded
    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
