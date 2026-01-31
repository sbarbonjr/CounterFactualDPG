#!/usr/bin/env python3
"""
Clean up dataset configs by removing parameters that match the base defaults.

This script compares each dataset config against configs/config.yaml and removes
any parameters that have the exact same value, making dataset configs leaner.

Usage:
    python scripts/cleanup_dataset_configs.py
    
    # Dry run (show what would be removed without modifying files)
    python scripts/cleanup_dataset_configs.py --dry-run
    
    # Clean a specific dataset only
    python scripts/cleanup_dataset_configs.py --dataset iris
"""

import argparse
import os
from pathlib import Path
import yaml
from copy import deepcopy

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = REPO_ROOT / "configs"
BASE_CONFIG_PATH = CONFIGS_DIR / "config.yaml"


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict, header_comment: str = None):
    """Save a dictionary to a YAML file with optional header comment."""
    with open(path, 'w') as f:
        if header_comment:
            f.write(header_comment)
            if not header_comment.endswith('\n'):
                f.write('\n')
            f.write('\n')
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def values_equal(val1, val2) -> bool:
    """Check if two values are equal, handling nested dicts and lists."""
    if type(val1) != type(val2):
        return False
    if isinstance(val1, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        return all(values_equal(val1[k], val2[k]) for k in val1)
    if isinstance(val1, list):
        if len(val1) != len(val2):
            return False
        return all(values_equal(v1, v2) for v1, v2 in zip(val1, val2))
    return val1 == val2


def remove_matching_defaults(config: dict, defaults: dict, path: str = "") -> tuple[dict, list]:
    """
    Recursively remove keys from config that match defaults.
    
    Args:
        config: The dataset-specific config
        defaults: The base defaults
        path: Current path for logging (e.g., "model.n_estimators")
        
    Returns:
        Tuple of (cleaned_config, list_of_removed_paths)
    """
    if not isinstance(config, dict) or not isinstance(defaults, dict):
        return config, []
    
    cleaned = {}
    removed = []
    
    for key, value in config.items():
        current_path = f"{path}.{key}" if path else key
        
        if key not in defaults:
            # Key doesn't exist in defaults, keep it
            cleaned[key] = value
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            # Recursively clean nested dicts
            nested_cleaned, nested_removed = remove_matching_defaults(value, defaults[key], current_path)
            removed.extend(nested_removed)
            if nested_cleaned:  # Only keep if there are remaining keys
                cleaned[key] = nested_cleaned
            else:
                removed.append(f"{current_path} (empty after cleanup)")
        elif values_equal(value, defaults[key]):
            # Value matches default, remove it
            removed.append(f"{current_path} = {repr(value)}")
        else:
            # Value differs from default, keep it
            cleaned[key] = value
    
    return cleaned, removed


def get_header_comment(dataset_name: str) -> str:
    """Generate a header comment for the config file."""
    return f"""# Configuration for {dataset_name} dataset
# Only dataset-specific overrides are included here.
# Base defaults are loaded from configs/config.yaml
#
# Usage:
#   python scripts/run_experiment.py --dataset {dataset_name} --method dpg
#   python scripts/run_experiment.py --dataset {dataset_name} --method dice"""


def cleanup_config(dataset_dir: Path, base_config: dict, dry_run: bool = False) -> tuple[int, list]:
    """
    Clean up a single dataset config.
    
    Returns:
        Tuple of (number_of_removed_params, list_of_removed_paths)
    """
    dataset_name = dataset_dir.name
    config_path = dataset_dir / "config.yaml"
    
    if not config_path.exists():
        return 0, []
    
    config = load_yaml(config_path)
    original_config = deepcopy(config)
    
    cleaned_config, removed_paths = remove_matching_defaults(config, base_config)
    
    if not removed_paths:
        return 0, []
    
    if dry_run:
        print(f"\n{dataset_name}: Would remove {len(removed_paths)} parameters:")
        for path in removed_paths:
            print(f"  - {path}")
    else:
        # Save cleaned config
        header = get_header_comment(dataset_name)
        save_yaml(config_path, cleaned_config, header)
        print(f"\n{dataset_name}: Removed {len(removed_paths)} parameters")
        for path in removed_paths:
            print(f"  - {path}")
    
    return len(removed_paths), removed_paths


def main():
    parser = argparse.ArgumentParser(
        description="Clean up dataset configs by removing default values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without modifying files'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Clean only a specific dataset'
    )
    
    args = parser.parse_args()
    
    # Load base config
    if not BASE_CONFIG_PATH.exists():
        print(f"ERROR: Base config not found at {BASE_CONFIG_PATH}")
        return 1
    
    base_config = load_yaml(BASE_CONFIG_PATH)
    print(f"Loaded base defaults from {BASE_CONFIG_PATH}")
    
    if args.dry_run:
        print("\n*** DRY RUN - No files will be modified ***")
    
    print("=" * 60)
    
    # Get directories to process
    if args.dataset:
        dirs_to_process = [CONFIGS_DIR / args.dataset]
    else:
        dirs_to_process = sorted([
            d for d in CONFIGS_DIR.iterdir()
            if d.is_dir() and (d / "config.yaml").exists()
        ])
    
    total_removed = 0
    datasets_cleaned = 0
    
    for dataset_dir in dirs_to_process:
        if not dataset_dir.exists():
            print(f"WARNING: Dataset directory not found: {dataset_dir}")
            continue
        
        num_removed, _ = cleanup_config(dataset_dir, base_config, args.dry_run)
        if num_removed > 0:
            total_removed += num_removed
            datasets_cleaned += 1
    
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Datasets processed: {len(dirs_to_process)}")
    print(f"  Datasets cleaned: {datasets_cleaned}")
    print(f"  Total parameters removed: {total_removed}")
    
    if args.dry_run and total_removed > 0:
        print(f"\nRun without --dry-run to apply changes")
    
    return 0


if __name__ == "__main__":
    exit(main())
