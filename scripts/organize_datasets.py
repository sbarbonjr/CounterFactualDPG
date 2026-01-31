#!/usr/bin/env python3
"""
Script to organize datasets from /bases directory into /configs structure.
Each dataset will be moved to /configs/dataset_name/dataset.csv
"""

import os
import shutil
from pathlib import Path

# Define base paths
REPO_ROOT = Path(__file__).parent.parent.resolve()
BASES_DIR = REPO_ROOT / "bases"
CONFIGS_DIR = REPO_ROOT / "configs"

def sanitize_dataset_name(filename: str) -> str:
    """
    Convert filename to a suitable directory name.
    Example: 'abalone_19.csv' -> 'abalone_19'
    """
    return filename.replace('.csv', '')

def organize_datasets():
    """
    Move datasets from /bases to /configs/dataset_name/dataset.csv
    """
    if not BASES_DIR.exists():
        print(f"ERROR: Bases directory not found: {BASES_DIR}")
        return
    
    if not CONFIGS_DIR.exists():
        print(f"Creating configs directory: {CONFIGS_DIR}")
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in bases directory
    csv_files = list(BASES_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {BASES_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {BASES_DIR}")
    print("-" * 80)
    
    moved_count = 0
    skipped_count = 0
    
    for csv_file in sorted(csv_files):
        dataset_name = sanitize_dataset_name(csv_file.name)
        target_dir = CONFIGS_DIR / dataset_name
        target_file = target_dir / "dataset.csv"
        
        # Check if target already exists
        if target_file.exists():
            print(f"SKIP: {csv_file.name} -> {target_file.relative_to(REPO_ROOT)} (already exists)")
            skipped_count += 1
            continue
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        try:
            shutil.move(str(csv_file), str(target_file))
            print(f"MOVED: {csv_file.name} -> {target_file.relative_to(REPO_ROOT)}")
            moved_count += 1
        except Exception as e:
            print(f"ERROR moving {csv_file.name}: {e}")
    
    print("-" * 80)
    print(f"Summary:")
    print(f"  Moved: {moved_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Total: {len(csv_files)}")
    
    # Check if bases directory is now empty
    remaining_files = list(BASES_DIR.glob("*.csv"))
    if not remaining_files:
        print(f"\nAll CSV files have been moved from {BASES_DIR}")
        print(f"You can safely delete the empty /bases directory if desired.")
    else:
        print(f"\n{len(remaining_files)} CSV files remain in {BASES_DIR}")

if __name__ == "__main__":
    print("=" * 80)
    print("Dataset Organization Script")
    print("=" * 80)
    organize_datasets()
