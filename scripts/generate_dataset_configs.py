#!/usr/bin/env python3
"""
Generate unified config files for all datasets in configs/ directory.

This script creates a single config.yaml per dataset with:
- Shared dataset/model/experiment settings at the root
- Method-specific settings (dpg, dice, etc.) nested under 'methods'

Usage:
    python scripts/generate_dataset_configs.py
    
    # Force regeneration of existing configs
    python scripts/generate_dataset_configs.py --force
    
    # Migrate from old structure (dpg/, dice/ subfolders)
    python scripts/generate_dataset_configs.py --migrate
"""

import argparse
import os
from pathlib import Path
import yaml
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = REPO_ROOT / "configs"

# Template for unified config
CONFIG_TEMPLATE = {
    'data': {
        'dataset': None,  # Will be filled
        'dataset_path': None,  # Will be filled
        'target_column': None,  # Will be inferred or set to last column
        'test_size': 0.2,
        'random_state': 42,
    },
    'model': {
        'type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42,
    },
    'experiment': {
        'project': 'CounterFactualDPG',
        'tags': [],
    },
    'experiment_params': {
        'seed': 42,
        'num_samples': 5,
        'num_replications': 3,
        'num_combinations_to_test': 1,
        'parallel_replications': True,
        'compute_comprehensive_metrics': True,
    },
    'output': {
        'local_dir': 'outputs',
        'save_visualizations': True,
        'save_visualization_images': True,
    },
    'counterfactual_defaults': {
        'actionability': {},
    },
    'methods': {
        'dpg': {
            'method': 'dpg',
            'population_size': 50,
            'max_generations': 100,
            'mutation_rate': 0.1,
            'diversity_weight': 0.5,
            'repulsion_weight': 4.0,
            'boundary_weight': 15.0,
            'distance_factor': 2.0,
            'sparsity_factor': 1.0,
            'constraints_factor': 3.0,
            'original_escape_weight': 2.0,
            'escape_pressure': 0.5,
            'prioritize_non_overlapping': True,
        },
        'dice': {
            'method': 'dice',
            'total_CFs': 4,
            'proximity_weight': 0.5,
            'diversity_weight': 1.0,
            'generation_method': 'genetic',
        },
    },
}


def deep_copy_template() -> dict:
    """Create a deep copy of the config template."""
    return yaml.safe_load(yaml.dump(CONFIG_TEMPLATE))


def infer_target_column(df: pd.DataFrame) -> str:
    """Infer the target column from a DataFrame."""
    # Common target column names
    common_targets = ['target', 'class', 'label', 'y', 'outcome', 'result', 
                      'credit_risk', 'diagnosis', 'quality', 'type', 'species']
    
    for col in df.columns:
        if col.lower() in common_targets:
            return col
    
    # Check last column - often the target
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 20:  # Likely categorical/target
        return last_col
    
    # Default to last column
    return last_col


def infer_categorical_features(df: pd.DataFrame, target_col: str) -> list:
    """Infer categorical features from DataFrame."""
    categorical = []
    for col in df.columns:
        if col == target_col:
            continue
        # Check if column is object/string type or has few unique values
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical.append(col)
        elif df[col].nunique() <= 10 and df[col].dtype in ['int64', 'int32', 'float64']:
            # Might be encoded categorical
            pass  # Don't auto-add, let the system figure it out
    return categorical


def load_existing_config(config_path: Path) -> dict:
    """Load an existing config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_existing_method_config(base_config: dict, method_name: str, method_config: dict) -> dict:
    """Merge an existing method config into the unified config."""
    # Extract counterfactual-specific settings
    if 'counterfactual' in method_config:
        cf_config = method_config['counterfactual']
        # Merge with defaults
        if method_name in base_config['methods']:
            base_config['methods'][method_name].update(cf_config)
        else:
            base_config['methods'][method_name] = cf_config
        base_config['methods'][method_name]['method'] = method_name
    
    # Merge data settings (only if not set or more specific)
    if 'data' in method_config:
        for key, value in method_config['data'].items():
            if key not in base_config['data'] or base_config['data'][key] is None:
                base_config['data'][key] = value
    
    # Merge model settings
    if 'model' in method_config:
        base_config['model'].update(method_config['model'])
    
    # Merge experiment settings
    if 'experiment' in method_config:
        for key, value in method_config['experiment'].items():
            if key == 'tags':
                # Merge tags
                existing_tags = base_config['experiment'].get('tags', [])
                base_config['experiment']['tags'] = list(set(existing_tags + value))
            elif key not in base_config['experiment']:
                base_config['experiment'][key] = value
    
    # Merge experiment_params
    if 'experiment_params' in method_config:
        base_config['experiment_params'].update(method_config['experiment_params'])
    
    # Merge output settings
    if 'output' in method_config:
        base_config['output'].update(method_config['output'])
    
    return base_config


def generate_config(dataset_dir: Path, migrate: bool = False) -> dict:
    """Generate a unified config for a dataset directory."""
    dataset_name = dataset_dir.name
    dataset_csv = dataset_dir / "dataset.csv"
    
    config = deep_copy_template()
    
    config['data']['dataset'] = dataset_name
    config['data']['dataset_path'] = f"configs/{dataset_name}/dataset.csv"
    config['experiment']['tags'] = [dataset_name]
    config['experiment']['name'] = dataset_name
    
    # Try to infer target column and features from CSV
    if dataset_csv.exists():
        try:
            df = pd.read_csv(dataset_csv, nrows=100)
            target = infer_target_column(df)
            config['data']['target_column'] = target
            print(f"  Inferred target column: {target}")
            
            # Infer categorical features
            categorical = infer_categorical_features(df, target)
            if categorical:
                config['data']['categorical_features'] = categorical
                print(f"  Inferred categorical features: {categorical}")
        except Exception as e:
            print(f"  WARNING: Could not read CSV to infer target: {e}")
            config['data']['target_column'] = 'target'  # Default
    else:
        config['data']['target_column'] = 'target'
    
    # If migrate flag is set, try to merge existing dpg/dice configs
    if migrate:
        dpg_config_path = dataset_dir / "dpg" / "config.yaml"
        dice_config_path = dataset_dir / "dice" / "config.yaml"
        
        if dpg_config_path.exists():
            print(f"  Migrating existing DPG config...")
            existing = load_existing_config(dpg_config_path)
            config = merge_existing_method_config(config, 'dpg', existing)
        
        if dice_config_path.exists():
            print(f"  Migrating existing DICE config...")
            existing = load_existing_config(dice_config_path)
            config = merge_existing_method_config(config, 'dice', existing)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified config files for datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of existing configs'
    )
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Migrate settings from existing dpg/dice subfolders'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Generate config for a specific dataset only'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Unified Dataset Configs")
    print("=" * 60)
    
    generated = 0
    skipped = 0
    migrated = 0
    
    # Get list of directories to process
    if args.dataset:
        dirs_to_process = [CONFIGS_DIR / args.dataset]
    else:
        dirs_to_process = sorted(CONFIGS_DIR.iterdir())
    
    for item in dirs_to_process:
        if not item.is_dir():
            continue
        
        dataset_name = item.name
        config_path = item / "config.yaml"
        dataset_csv = item / "dataset.csv"
        
        # Skip special directories
        if dataset_name in ['sweep_config.yaml', '__pycache__']:
            continue
        
        # Check for dataset.csv or existing method subfolders
        has_dataset = dataset_csv.exists()
        has_old_structure = (item / "dpg").exists() or (item / "dice").exists()
        
        if not has_dataset and not has_old_structure:
            continue
        
        # Skip if unified config already exists (unless --force)
        if config_path.exists() and not args.force:
            print(f"SKIP: {dataset_name} (config.yaml exists, use --force to overwrite)")
            skipped += 1
            continue
        
        # Check if this needs migration
        needs_migration = has_old_structure and args.migrate
        
        if needs_migration:
            print(f"MIGRATE: {dataset_name}")
            migrated += 1
        else:
            print(f"GENERATE: {dataset_name}")
        
        config = generate_config(item, migrate=needs_migration)
        
        # Write config with nice formatting
        with open(config_path, 'w') as f:
            # Add header comment
            f.write(f"# Unified configuration for {dataset_name} dataset\n")
            f.write(f"# Generated by generate_dataset_configs.py\n")
            f.write(f"# Supports methods: dpg, dice (more can be added under 'methods')\n")
            f.write(f"#\n")
            f.write(f"# Usage:\n")
            f.write(f"#   python scripts/run_experiment.py --dataset {dataset_name} --method dpg\n")
            f.write(f"#   python scripts/run_experiment.py --dataset {dataset_name} --method dice\n")
            f.write(f"\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        generated += 1
    
    print("-" * 60)
    print(f"Generated: {generated}")
    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print("=" * 60)
    
    if generated > 0:
        print("\nNext steps:")
        print("1. Review generated configs and adjust target_column if needed")
        print("2. Run experiments with: python scripts/run_experiment.py --dataset <name> --method dpg")


if __name__ == "__main__":
    main()
