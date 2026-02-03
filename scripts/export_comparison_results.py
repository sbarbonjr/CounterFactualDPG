#!/usr/bin/env python3
"""Export DPG vs DiCE comparison results to outputs/comparison_results.

This script replicates the functionality of notebooks/technique_comparison.ipynb,
fetching experiment results from WandB and exporting all results to files
instead of displaying them in a notebook.

Outputs are saved to: outputs/comparison_results/
- comparison.csv: Main comparison table with all datasets
- summary.csv: Cross-dataset summary with win rates
- method_metrics_<dataset>.csv: Per-dataset method-metrics tables
- winner_heatmap.png: Heatmap showing winners per dataset-metric
- winner_heatmap.csv: Winner data as CSV (DPG/DiCE/Tie per dataset-metric)
- comparison_numeric.csv: Numeric values for both DPG and DiCE per dataset-metric
- visualizations/winner_heatmap_small.png: Small heatmap with 8 key metrics only
- visualizations/winner_heatmap_small.csv: Small winner data as CSV
- visualizations/comparison_numeric_small.csv: Small numeric values CSV
- radar_charts.png: Radar charts for datasets
- visualizations/<dataset>/: Per-dataset visualizations including:
  - heatmap_techniques.png: Heatmap comparing DPG vs DiCE counterfactuals
  - pca_comparison.png: PCA plot comparing both methods' counterfactuals (requires dataset/model)
  - radar.png: Radar chart for the dataset
  - bar_*.png: Bar charts for each metric
  - WandB visualizations (comparison, pca_clean, heatmap, etc.)
- metadata.pkl: Saved run metadata for local-only regeneration
- comparison_summary.txt: Console summary output
- rf_model_information.csv: Random Forest model information (train/test split, accuracies, hyperparameters)
- rf_model_summary.txt: Summary statistics of RF models across all datasets
- dataset_overview.tex: LaTeX table with dataset overview (features, samples, classes, accuracies)
- dataset_dpg_constraints.tex: LaTeX table with detailed DPG constraints per dataset (landscape, one dataset per page)
- dpg_constraints/<dataset>_dpg_constraints.json: DPG-learned constraints (min/max bounds) per dataset as JSON

Usage:
    python scripts/export_comparison_results.py                    # Full export (fetches from WandB)
    python scripts/export_comparison_results.py --local-only         # Regenerate images only from disk
    python scripts/export_comparison_results.py --ids <yaml_file>  # Fetch specific run IDs from YAML
"""

import sys
import os
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import tempfile
import json
import pickle
import wandb
from PIL import Image
import shutil
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Export DPG vs DiCE comparison results')
parser.add_argument('--local-only', action='store_true',
                    help='Only regenerate visualizations from existing data, do not fetch from WandB')
parser.add_argument('--ids', type=str, default=None,
                    help='Path to YAML file containing specific run IDs to fetch (disregards min_created_at)')
args = parser.parse_args()

from scripts.compare_techniques import (
    fetch_all_runs,
    create_comparison_table,
    create_method_metrics_table,
    print_comparison_summary,
    plot_heatmap_winners,
    COMPARISON_METRICS,
    TECHNIQUE_COLORS,
    determine_winner,
    filter_to_latest_run_per_combo,
)

from scripts.statistical_analysis import run_full_analysis as run_statistical_analysis

from utils.config_manager import load_config

from CounterFactualVisualizer import (
    heatmap_techniques, 
    plot_pca_with_counterfactuals_comparison,
    plot_sample_and_counterfactual_comparison,
    plot_sample_and_counterfactual_comparison_simple,
    plot_sample_and_counterfactual_comparison_combined,
    plot_ridge_comparison
)
from utils.dataset_loader import load_dataset
from utils.config_manager import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuration (matches notebook hardcoded values)
WANDB_ENTITY = 'mllab-ts-universit-di-trieste'
WANDB_PROJECT = 'CounterFactualDPG'
SELECTED_DATASETS = None  # No filter - fetch all
MIN_CREATED_AT = "2026-01-31T22:00:00" # Fetch runs created after this date
APPLY_EXCLUDED_DATASETS = True

# Dataset to export individual counterfactual comparison plots
DATASET_FOR_CF_COMPARISON = 'diabetes'

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    '_comparison_results')

# Metadata file for storing run information
METADATA_FILE = os.path.join(OUTPUT_DIR, 'metadata.pkl')

# Cache file for storing fetched WandB data (sample + counterfactuals per dataset)
WANDB_CACHE_FILE = os.path.join(OUTPUT_DIR, 'wandb_data_cache.pkl')

# Repository root for loading datasets
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache for loaded datasets and models
_DATASET_CACHE = {}

# Cache for WandB fetched data (populated during fetch, used in local-only mode)
_WANDB_DATA_CACHE = {}


def save_wandb_data_cache(cache_data):
    """Save WandB fetched data (samples and counterfactuals) to disk for local-only mode."""
    with open(WANDB_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"✓ Saved WandB data cache to: {WANDB_CACHE_FILE}")


def load_wandb_data_cache():
    """Load cached WandB data from disk."""
    if not os.path.exists(WANDB_CACHE_FILE):
        return {}
    
    with open(WANDB_CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data


def load_dataset_and_model(dataset_name):
    """Load dataset and train model for a given dataset.
    
    Args:
        dataset_name: Name of the dataset to load
    
    Returns:
        dict with keys:
            - model: Trained RandomForestClassifier
            - dataset: Full features DataFrame
            - target: Target labels array
            - feature_names: List of feature names
            - train_features: Training features DataFrame
            - test_features: Test features DataFrame
            - train_labels: Training labels array
            - test_labels: Test labels array
            - train_accuracy: Model accuracy on training set
            - test_accuracy: Model accuracy on test set
            - test_size: Test set size (proportion)
            - random_state: Random state used for splitting
            - model_params: Model hyperparameters
    """
    # Check cache first
    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]
    
    try:
        # Load dataset config using load_config (to inherit base defaults from configs/config.yaml)
        config_path = os.path.join(REPO_ROOT, 'configs', dataset_name, 'config.yaml')
        if not os.path.exists(config_path):
            print(f"  ⚠ Config not found for {dataset_name}: {config_path}")
            return None
        
        # Use load_config to get merged config (base defaults + dataset-specific)
        config = load_config(config_path, repo_root=REPO_ROOT)
        
        # Set random seed BEFORE loading dataset and splitting (matches run_experiment.py)
        # This ensures reproducibility like the original experiment
        seed = getattr(config.experiment_params, 'seed', 42)
        np.random.seed(seed)
        
        # Load dataset
        dataset_info = load_dataset(config, repo_root=REPO_ROOT)
        
        features_df = dataset_info["features_df"]
        labels = dataset_info["labels"]
        feature_names = dataset_info["feature_names"]
        
        # Split data (same as in run_experiment)
        test_size = getattr(config.data, 'test_size', 0.3)
        random_state = getattr(config.data, 'random_state', 42)
        
        train_features, test_features, train_labels, test_labels = train_test_split(
            features_df,
            labels,
            test_size=test_size,
            random_state=random_state,
        )
        
        # Train model (same configuration as in run_experiment)
        model_type = getattr(config.model, 'type', 'RandomForestClassifier')
        if model_type == "RandomForestClassifier":
            model_config = config.model.to_dict() if hasattr(config.model, 'to_dict') else dict(config.model)
            model_params = {k: v for k, v in model_config.items() if k != 'type' and v is not None}
            model = RandomForestClassifier(**model_params)
        else:
            print(f"  ⚠ Unknown model type for {dataset_name}: {model_type}")
            return None
        
        # Train model
        model.fit(train_features, train_labels)
        
        # Calculate accuracies
        train_accuracy = model.score(train_features, train_labels)
        test_accuracy = model.score(test_features, test_labels)
        
        result = {
            'model': model,
            'dataset': features_df,
            'target': labels,
            'feature_names': feature_names,
            'train_features': train_features,
            'test_features': test_features,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_size': test_size,
            'random_state': random_state,
            'model_params': model_params,
        }
        
        # Cache the result
        _DATASET_CACHE[dataset_name] = result
        
        return result
        
    except Exception as e:
        print(f"  ⚠ Error loading dataset/model for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_included_datasets():
    """Load priority_datasets from main config."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs',
        'config.yaml'
    )
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('priority_datasets', None)
    return None


def load_run_ids_from_yaml(yaml_path):
    """Load run IDs from YAML file.
    
    Args:
        yaml_path: Path to YAML file with structure:
            datasets:
              dataset_name:
                dice: run_id
                dpg: run_id
    
    Returns:
        Dictionary mapping dataset -> technique -> run_id
    """
    if not os.path.exists(yaml_path):
        print(f"❌ YAML file not found: {yaml_path}")
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'datasets' not in data:
        print(f"❌ YAML file missing 'datasets' key: {yaml_path}")
        return None
    
    run_ids = data['datasets']
    print(f"✓ Loaded run IDs from {yaml_path}")
    print(f"  Datasets: {list(run_ids.keys())}")
    
    return run_ids


def fetch_specific_runs(run_ids_dict):
    """Fetch specific runs by ID from WandB.
    
    Args:
        run_ids_dict: Dictionary with structure:
            dataset_name:
              dice: run_id
              dpg: run_id
    
    Returns:
        DataFrame with run data
    """
    api = wandb.Api(timeout=60)
    runs_data = []
    
    for dataset, techniques in run_ids_dict.items():
        for technique, run_id in techniques.items():
            try:
                run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
                
                if run.state != 'finished':
                    print(f"  ⚠ {dataset}/{technique}: Run {run_id} not finished (state: {run.state})")
                    continue
                
                config = run.config
                summary = run.summary._json_dict
                
                # Extract metrics from run
                run_data = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'dataset': dataset,
                    'technique': technique.lower(),
                    'state': run.state,
                }
                
                # Extract per-counterfactual metrics using COMPARISON_METRICS and wandb_keys
                for metric_key, metric_info in COMPARISON_METRICS.items():
                    value = None
                    # Try each possible WandB key for this metric
                    wandb_keys = metric_info.get('wandb_keys', [f'metrics/{metric_key}', f'combo_metrics/{metric_key}'])
                    for wkey in wandb_keys:
                        if wkey in summary:
                            value = summary[wkey]
                            break
                    
                    run_data[metric_key] = value
                
                # Also extract any additional summary metrics as fallback
                for key, value in summary.items():
                    if isinstance(value, (int, float)) and key not in run_data:
                        # Clean up metric names
                        clean_key = key.replace('combo_metrics/', '').replace('metrics/', '')
                        if clean_key not in run_data:
                            run_data[clean_key] = value
                
                runs_data.append(run_data)
                print(f"  ✓ {dataset}/{technique}: Fetched run {run_id}")
                
            except Exception as e:
                print(f"  ❌ {dataset}/{technique}: Error fetching run {run_id}: {e}")
                continue
    
    if not runs_data:
        print("❌ No runs were successfully fetched")
        return pd.DataFrame()
    
    df = pd.DataFrame(runs_data)
    return df


def save_metadata(raw_df):
    """Save metadata including run information for local-only mode."""
    metadata = {
        'raw_df': raw_df,
        'datasets': sorted(raw_df['dataset'].unique()),
        'techniques': sorted(raw_df['technique'].unique()),
        'num_runs': len(raw_df)
    }
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to: {METADATA_FILE}")


def load_metadata():
    """Load metadata from disk for local-only mode."""
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file not found: {METADATA_FILE}")
        print("   Run the script first without --local-only to fetch and save data.")
        return None
    
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Loaded metadata from: {METADATA_FILE}")
    print(f"  Datasets: {metadata['datasets']}")
    print(f"  Techniques: {metadata['techniques']}")
    print(f"  Number of runs: {metadata['num_runs']}")
    return metadata


def load_comparison_from_disk():
    """Load comparison data from disk."""
    comparison_path = os.path.join(OUTPUT_DIR, 'comparison.csv')
    if not os.path.exists(comparison_path):
        print(f"❌ Comparison file not found: {comparison_path}")
        print("   Run the script first without --local-only to generate comparison data.")
        return None
    
    comparison_df = pd.read_csv(comparison_path)
    print(f"✓ Loaded comparison data from: {comparison_path}")
    return comparison_df


def fetch_wandb_visualizations(raw_df, dataset, dataset_viz_dir):
    """Fetch comparison, pca_clean, and heatmap visualizations from WandB."""
    if args.local_only:
        print(f"  ⚠ {dataset}: Skipping WandB visualizations (local-only mode)")
        return []
    
    viz_types_exported = []
    try:
        dataset_runs = raw_df[raw_df['dataset'] == dataset]
        
        dpg_run = dataset_runs[dataset_runs['technique'] == 'dpg'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dpg']) > 0 else None
        dice_run = dataset_runs[dataset_runs['technique'] == 'dice'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dice']) > 0 else None
        
        if dpg_run is None or dice_run is None:
            print(f"  ⚠ {dataset}: Missing DPG or DiCE run, skipping WandB visualizations")
            return []
        
        api = wandb.Api(timeout=60)
        dpg_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dpg_run['run_id']}")
        dice_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dice_run['run_id']}")
        
        # Fetch constraints_overview from DPG run
        constraints_overview = None
        for f in dpg_run_obj.files():
            if f.name.endswith('.png') and 'dpg/constraints_overview' in f.name:
                constraints_overview = f
                with tempfile.TemporaryDirectory() as tmpdir:
                    constraints_overview.download(root=tmpdir, replace=True)
                    img_path = os.path.join(tmpdir, constraints_overview.name)
                    dest_path = os.path.join(dataset_viz_dir, 'constraints_overview.png')
                    shutil.copy(img_path, dest_path)
                    viz_types_exported.append('constraints_overview.png')
                break
        
        # Get visualization images from both runs
        def get_viz_images(run):
            images = {}
            for f in run.files():
                if f.name.endswith('.png') and 'visualizations/' in f.name:
                    basename = os.path.basename(f.name)
                    match = re.match(r'([a-z_]+)_\\d+_', basename)
                    if match:
                        viz_type = match.group(1)
                        if viz_type not in images:
                            images[viz_type] = f
            return images
        
        dpg_images = get_viz_images(dpg_run_obj)
        dice_images = get_viz_images(dice_run_obj)
        
        viz_types_to_fetch = ['comparison', 'pca_clean', 'heatmap', 'standard_deviation']
        
        for viz_type in viz_types_to_fetch:
            if viz_type in dpg_images:
                with tempfile.TemporaryDirectory() as tmpdir:
                    download_path = os.path.join(tmpdir, dpg_images[viz_type].name)
                    dpg_images[viz_type].download(root=tmpdir, replace=True)
                    dest_path = os.path.join(dataset_viz_dir, f'{viz_type}_dpg.png')
                    shutil.copy(download_path, dest_path)
                    viz_types_exported.append(f'{viz_type}_dpg.png')
            
            if viz_type in dice_images:
                with tempfile.TemporaryDirectory() as tmpdir:
                    download_path = os.path.join(tmpdir, dice_images[viz_type].name)
                    dice_images[viz_type].download(root=tmpdir, replace=True)
                    dest_path = os.path.join(dataset_viz_dir, f'{viz_type}_dice.png')
                    shutil.copy(download_path, dest_path)
                    viz_types_exported.append(f'{viz_type}_dice.png')
        
        return viz_types_exported
    except Exception as e:
        print(f"  ⚠ {dataset}: Error fetching WandB visualizations: {e}")
        return []


def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def fetch_and_filter_data():
    """Fetch data from WandB and apply filters."""
    print("\n" + "="*80)
    print("FETCHING DATA FROM WANDB")
    print("="*80)
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")
    
    # Check if --ids flag was provided
    if args.ids:
        print(f"Mode: Fetching specific run IDs from YAML")
        print(f"YAML file: {args.ids}")
        print(f"⚠ Ignoring min_created_at filter")
        
        # Load run IDs from YAML
        run_ids_dict = load_run_ids_from_yaml(args.ids)
        if run_ids_dict is None:
            return pd.DataFrame()
        
        # Fetch specific runs
        raw_df = fetch_specific_runs(run_ids_dict)
        
        if len(raw_df) == 0:
            print("\n❌ No runs fetched from specified IDs")
            return raw_df
        
        print(f"\n✓ Successfully fetched {len(raw_df)} runs from {len(run_ids_dict)} datasets")
        print(f"  Datasets: {sorted(raw_df['dataset'].unique())}")
        print(f"  Techniques: {sorted(raw_df['technique'].unique())}")
        
        return raw_df
    
    # Standard mode: fetch by date filter
    print(f"Mode: Fetching runs by date filter")
    print(f"Datasets filter: {SELECTED_DATASETS or 'All'}")
    print(f"Min created at: {MIN_CREATED_AT}")
    print(f"Apply excluded datasets: {APPLY_EXCLUDED_DATASETS}")
    
    # Load included datasets from config
    included_datasets = load_included_datasets()
    print(f"Included datasets from config: {included_datasets if included_datasets else 'All'}")
    
    # Fetch from WandB
    raw_df = fetch_all_runs(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        datasets=SELECTED_DATASETS,
        limit=300,
        min_created_at=MIN_CREATED_AT
    )
    
    # Apply included datasets filter if specified
    if included_datasets is not None:
        pre_filter_count = len(raw_df)
        raw_df = raw_df[raw_df['dataset'].isin(included_datasets)]
        print(f"Applied included datasets filter: {pre_filter_count} -> {len(raw_df)} runs")
        print(f"Included datasets: {sorted(raw_df['dataset'].unique())}")
    
    # Filter to latest run per combo
    raw_df = filter_to_latest_run_per_combo(raw_df)
    
    return raw_df


def create_comparison_table_small(raw_df):
    """Create comparison table with small=True (7 key metrics)."""
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE (small=True)")
    print("="*80)
    
    comparison_df = create_comparison_table(raw_df, small=True)
    print(f"Comparison table created with {len(comparison_df)} datasets")
    
    return comparison_df


def export_comparison_table(comparison_df):
    """Export main comparison table to CSV."""
    output_path = os.path.join(OUTPUT_DIR, 'comparison.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"✓ Exported comparison table to: {output_path}")


def export_method_metrics_tables(raw_df):
    """Export method-metrics tables for each dataset."""
    print("\n" + "="*80)
    print("EXPORTING METHOD-METRICS TABLES FOR ALL DATASETS")
    print("="*80)
    
    all_datasets = sorted(raw_df['dataset'].unique())
    print(f"Found {len(all_datasets)} datasets")
    
    for dataset in all_datasets:
        # Create method-metrics table (always use small=True like notebook)
        table = create_method_metrics_table(raw_df, dataset=dataset, small=True, styled=False)
        
        if table is not None and len(table) > 0:
            output_path = os.path.join(OUTPUT_DIR, f'method_metrics_{dataset}.csv')
            table.to_csv(output_path)
            print(f"✓ Exported method-metrics table for '{dataset}'")
        else:
            print(f"⚠ No data found for dataset: {dataset}")


def export_summary_statistics(comparison_df):
    """Export cross-dataset summary with win rates."""
    print("\n" + "="*80)
    print("CALCULATING CROSS-DATASET SUMMARY")
    print("="*80)
    
    # Calculate overall win rates
    win_counts = {'dpg': {}, 'dice': {}}
    
    for metric_key, metric_info in COMPARISON_METRICS.items():
        dpg_col = f'{metric_key}_dpg'
        dice_col = f'{metric_key}_dice'
        
        if dpg_col not in comparison_df.columns or dice_col not in comparison_df.columns:
            continue
        
        dpg_wins = 0
        dice_wins = 0
        
        for _, row in comparison_df.iterrows():
            winner = determine_winner(row.get(dpg_col), row.get(dice_col), metric_info['goal'])
            if winner == 'dpg':
                dpg_wins += 1
            elif winner == 'dice':
                dice_wins += 1
        
        win_counts['dpg'][metric_key] = dpg_wins
        win_counts['dice'][metric_key] = dice_wins
    
    # Create summary DataFrame
    summary_data = []
    for metric_key, metric_info in COMPARISON_METRICS.items():
        dpg_w = win_counts['dpg'].get(metric_key, 0)
        dice_w = win_counts['dice'].get(metric_key, 0)
        total = dpg_w + dice_w
        
        summary_data.append({
            'Metric': metric_info['name'],
            'Goal': metric_info['goal'],
            'DPG Wins': dpg_w,
            'DiCE Wins': dice_w,
            'DPG Win Rate': f"{dpg_w/total*100:.1f}%" if total > 0 else "N/A",
            'Total Comparisons': total,
            'Better': 'DPG' if dpg_w > dice_w else ('DiCE' if dice_w > dpg_w else 'Tie'),
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export to CSV
    output_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Exported summary statistics to: {output_path}")
    
    return summary_df


def export_winner_heatmap(comparison_df):
    """Export winner heatmap to PNG and CSV."""
    print("\n" + "="*80)
    print("EXPORTING WINNER HEATMAP")
    print("="*80)
    
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    output_path = os.path.join(viz_dir, 'winner_heatmap.png')
    fig = plot_heatmap_winners(comparison_df, figsize=(24, 12))
    
    if fig:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Exported winner heatmap to: {output_path}")
        plt.close(fig)
    else:
        print("⚠ Could not create winner heatmap")
    
    # Export winner data to CSV
    export_winner_heatmap_csv(comparison_df)


def export_winner_heatmap_small(comparison_df):
    """Export small winner heatmap with only key metrics to PNG and CSV."""
    print("\n" + "="*80)
    print("EXPORTING SMALL WINNER HEATMAP (KEY METRICS ONLY)")
    print("="*80)
    
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Small metrics to include
    small_metrics = [
        'perc_valid_cf_all',
        'perc_actionable_cf_all',
        'plausibility_nbr_cf',
        'distance_mh',
        'avg_nbr_changes',
        'count_diversity_all',
        'accuracy_knn_sklearn',
        'runtime'
    ]
    
    output_path = os.path.join(viz_dir, 'winner_heatmap_small.png')
    fig = plot_heatmap_winners(comparison_df, figsize=(20, 12), metrics_to_include=small_metrics)
    
    if fig:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Exported small winner heatmap to: {output_path}")
        plt.close(fig)
    else:
        print("⚠ Could not create small winner heatmap")
    
    # Export winner data to CSV
    export_winner_heatmap_csv(comparison_df, metrics_to_include=small_metrics, filename_suffix='_small')


def export_radar_chart_for_dataset(comparison_df, dataset, viz_dir):
    """Export radar chart for a specific dataset."""
    from scripts.compare_techniques import plot_radar_chart
    
    safe_name = dataset.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(viz_dir, f'radar_{safe_name}.png')
    
    fig = plot_radar_chart(
        comparison_df,
        dataset,
        output_path=output_path
    )
    
    if fig:
        plt.close(fig)
        return True
    return False


def export_radar_charts(comparison_df):
    """Export radar charts for first 4 datasets."""
    print("\n" + "="*80)
    print("EXPORTING RADAR CHARTS")
    print("="*80)
    
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    available_datasets = sorted(comparison_df['dataset'].unique())
    n_to_show = min(4, len(available_datasets))
    
    print(f"Creating radar charts for first {n_to_show} datasets:")
    for ds in available_datasets[:n_to_show]:
        print(f"  - {ds}")
    
    fig, axes = plt.subplots(1, n_to_show, figsize=(5*n_to_show, 5), subplot_kw=dict(polar=True))
    
    if n_to_show == 1:
        axes = [axes]
    
    for i, dataset in enumerate(available_datasets[:n_to_show]):
        # Get data for this dataset
        row = comparison_df[comparison_df['dataset'] == dataset].iloc[0]
        
        metrics = []
        dpg_values = []
        dice_values = []
        
        for metric_key, metric_info in COMPARISON_METRICS.items():
            dpg_col = f'{metric_key}_dpg'
            dice_col = f'{metric_key}_dice'
            
            if dpg_col in row.index and dice_col in row.index:
                dpg_val = row[dpg_col]
                dice_val = row[dice_col]
                
                if pd.notna(dpg_val) and pd.notna(dice_val):
                    metrics.append(metric_info['name'][:10])  # Truncate for readability
                    max_val = max(abs(dpg_val), abs(dice_val))
                    if max_val > 0:
                        if metric_info['goal'] == 'minimize':
                            dpg_values.append(1 - (dpg_val / (max_val * 1.1)))
                            dice_values.append(1 - (dice_val / (max_val * 1.1)))
                        else:
                            dpg_values.append(dpg_val / (max_val * 1.1))
                            dice_values.append(dice_val / (max_val * 1.1))
                    else:
                        dpg_values.append(0.5)
                        dice_values.append(0.5)
        
        if len(metrics) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            dpg_values += dpg_values[:1]
            dice_values += dice_values[:1]
            angles += angles[:1]
            
            ax = axes[i]
            ax.plot(angles, dpg_values, 'o-', linewidth=2, label='DPG', color=TECHNIQUE_COLORS['dpg'])
            ax.fill(angles, dpg_values, alpha=0.25, color=TECHNIQUE_COLORS['dpg'])
            ax.plot(angles, dice_values, 'o-', linewidth=2, label='DiCE', color=TECHNIQUE_COLORS['dice'])
            ax.fill(angles, dice_values, alpha=0.25, color=TECHNIQUE_COLORS['dice'])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, size=8)
            ax.set_ylim(0, 1)
            ax.set_title(dataset, size=12)
            if i == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.suptitle('Dataset Comparison Profiles (Higher = Better)', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(viz_dir, 'radar_charts.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Exported combined radar charts to: {output_path}")
    plt.close(fig)


def _load_local_viz_data(dataset, technique):
    """Load visualization data from local pkl files for local-only mode."""
    # Root outputs dir: outputs/{dataset}_{technique}/{sample_id}/after_viz_generation.pkl
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_base = os.path.join(root_dir, 'outputs')
    
    # Try exact folder name first
    technique_dir = os.path.join(outputs_base, f"{dataset}_{technique}")
    
    if not os.path.exists(technique_dir):
        # Try without underscores in dataset name
        technique_dir = os.path.join(outputs_base, f"{dataset.replace('_', '')}_{technique}")
    
    if not os.path.exists(technique_dir):
        return None
    
    # Find the most recent sample directory (highest number)
    sample_dirs = []
    for name in os.listdir(technique_dir):
        sample_path = os.path.join(technique_dir, name)
        if os.path.isdir(sample_path):
            try:
                sample_dirs.append((int(name), sample_path))
            except ValueError:
                continue
    
    if not sample_dirs:
        return None
    
    # Get most recent (highest sample ID)
    sample_dirs.sort(key=lambda x: x[0], reverse=True)
    _, sample_dir = sample_dirs[0]
    
    pkl_path = os.path.join(sample_dir, 'after_viz_generation.pkl')
    if not os.path.exists(pkl_path):
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"    Warning: Failed to load {pkl_path}: {e}")
        return None


def export_heatmap_techniques(raw_df, dataset, dataset_viz_dir):
    """Export heatmap comparing DPG vs DiCE counterfactuals."""
    
    # Handle local-only mode by loading from cache or disk
    if args.local_only:
        try:
            # Try to load from WandB data cache first (saved from previous non-local run)
            if dataset in _WANDB_DATA_CACHE:
                cached_data = _WANDB_DATA_CACHE[dataset]
                dpg_sample = cached_data['sample']
                dpg_class = cached_data.get('class', 0)
                dpg_cfs = cached_data['dpg_cfs']
                dice_cfs = cached_data['dice_cfs']
                restrictions = cached_data.get('restrictions')
            else:
                # Fallback to local pkl files
                dpg_data = _load_local_viz_data(dataset, 'dpg')
                dice_data = _load_local_viz_data(dataset, 'dice')
                
                if not dpg_data or not dice_data:
                    missing = []
                    if not dpg_data:
                        missing.append('DPG')
                    if not dice_data:
                        missing.append('DiCE')
                    print(f"  ⚠ {dataset}: Missing local {' and '.join(missing)} data, skipping heatmap_techniques")
                    return False
                
                # Extract sample and counterfactuals from local data
                dpg_sample = dpg_data.get('original_sample')
                feature_names = dpg_data.get('features_names', [])
                restrictions = dpg_data.get('restrictions')
                
                # Get class from sample prediction or default to 0
                dpg_class = 0  # Default, would need model to predict
                
                # Extract counterfactuals from visualizations
                dpg_cfs = []
                for viz in dpg_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dpg_cfs.append(cf)
                
                dice_cfs = []
                for viz in dice_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dice_cfs.append(cf)
            
            if not dpg_sample or not dpg_cfs or not dice_cfs:
                print(f"  ⚠ {dataset}: Missing local data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs) if dpg_cfs else 0}, dice_cfs: {len(dice_cfs) if dice_cfs else 0}")
                return False
            
            # Create heatmap
            fig = heatmap_techniques(
                sample=dpg_sample,
                class_sample=dpg_class,
                cf_list_1=dpg_cfs[:5],
                cf_list_2=dice_cfs[:5],
                technique_names=('DPG', 'DiCE'),
                restrictions=restrictions
            )
            
            if fig:
                output_path = os.path.join(dataset_viz_dir, 'heatmap_techniques.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {dataset}: Exported heatmap_techniques (from local data) and with restrictions: {restrictions is not None}")
                return True
            
            return False
            
        except Exception as e:
            print(f"  ⚠ {dataset}: Error loading local data for heatmap_techniques: {e}")
            return False
    
    # WandB mode
    try:
        dataset_runs = raw_df[raw_df['dataset'] == dataset]
        
        dpg_run = dataset_runs[dataset_runs['technique'] == 'dpg'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dpg']) > 0 else None
        dice_run = dataset_runs[dataset_runs['technique'] == 'dice'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dice']) > 0 else None
        
        # If runs not found in raw_df, fetch them directly from WandB
        api = wandb.Api(timeout=60)
        
        if dpg_run is None or dice_run is None:
            print(f"  → {dataset}: One or both runs missing from raw_df, querying WandB directly...")
            
            # Query WandB for the latest finished runs for this dataset
            try:
                all_runs = api.runs(
                    f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                    order="-created_at",
                    per_page=100,
                    filters={"state": "finished"}
                )
                
                dpg_wandb_run = None
                dice_wandb_run = None
                
                for run in all_runs:
                    if run.state != 'finished':
                        continue
                    
                    config = run.config
                    if 'data' not in config:
                        continue
                    
                    data_config = config['data']
                    run_dataset = data_config.get('dataset_name') or data_config.get('dataset')
                    
                    if run_dataset != dataset:
                        continue
                    
                    # Get technique
                    technique = data_config.get('method')
                    if not technique:
                        run_name_lower = run.name.lower()
                        if '_dpg' in run_name_lower or run_name_lower.endswith('dpg'):
                            technique = 'dpg'
                        elif '_dice' in run_name_lower or run_name_lower.endswith('dice'):
                            technique = 'dice'
                    
                    if technique:
                        technique = technique.lower()
                        if technique == 'dpg' and dpg_wandb_run is None:
                            dpg_wandb_run = run
                        elif technique == 'dice' and dice_wandb_run is None:
                            dice_wandb_run = run
                    
                    # Stop if we found both
                    if dpg_wandb_run and dice_wandb_run:
                        break
                
                # Use the WandB runs if we found them
                if dpg_wandb_run and dpg_run is None:
                    dpg_run_obj = dpg_wandb_run
                    print(f"  ✓ {dataset}: Found DPG run in WandB: {dpg_wandb_run.id}")
                elif dpg_run is not None:
                    dpg_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dpg_run['run_id']}")
                else:
                    dpg_run_obj = None
                
                if dice_wandb_run and dice_run is None:
                    dice_run_obj = dice_wandb_run
                    print(f"  ✓ {dataset}: Found DiCE run in WandB: {dice_wandb_run.id}")
                elif dice_run is not None:
                    dice_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dice_run['run_id']}")
                else:
                    dice_run_obj = None
                
                if dpg_run_obj is None or dice_run_obj is None:
                    missing = []
                    if dpg_run_obj is None:
                        missing.append('DPG')
                    if dice_run_obj is None:
                        missing.append('DiCE')
                    print(f"  ⚠ {dataset}: Could not find {' and '.join(missing)} run(s) in WandB, skipping heatmap_techniques")
                    return False
                    
            except Exception as e:
                print(f"  ⚠ {dataset}: Error querying WandB: {e}")
                return False
        else:
            # Both runs found in raw_df, use them
            dpg_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dpg_run['run_id']}")
            dice_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dice_run['run_id']}")
        
        # Fetch feature names (needed to convert lists to dicts)
        feature_names = dpg_run_obj.config.get('feature_names', [])
        
        # Fetch sample and counterfactuals from runs
        # Try multiple locations for sample data
        dpg_sample = dpg_run_obj.config.get('sample')
        if not dpg_sample:
            dpg_sample = dpg_run_obj.summary.get('sample')
        if not dpg_sample:
            dpg_sample = dpg_run_obj.summary.get('original_sample')
        
        dpg_class = dpg_run_obj.config.get('sample_class')
        if dpg_class is None:
            dpg_class = dpg_run_obj.summary.get('sample_class')
        if dpg_class is None:
            dpg_class = dpg_run_obj.summary.get('original_class')
        
        dpg_cfs = dpg_run_obj.summary.get('final_counterfactuals', [])
        if isinstance(dpg_cfs, str):
            try:
                dpg_cfs = json.loads(dpg_cfs)
            except:
                dpg_cfs = []
        
        dice_cfs = dice_run_obj.summary.get('final_counterfactuals', [])
        if isinstance(dice_cfs, str):
            try:
                dice_cfs = json.loads(dice_cfs)
            except:
                dice_cfs = []
        
        # Convert list format to dict format if needed
        if feature_names:
            # Convert sample from list to dict
            if isinstance(dpg_sample, list):
                dpg_sample = dict(zip(feature_names, dpg_sample))
            
            # Convert counterfactuals from list of lists to list of dicts
            if dpg_cfs and isinstance(dpg_cfs[0], list):
                dpg_cfs = [dict(zip(feature_names, cf)) for cf in dpg_cfs]
            if dice_cfs and isinstance(dice_cfs[0], list):
                dice_cfs = [dict(zip(feature_names, cf)) for cf in dice_cfs]
        
        # Debug: print what we found
        if not dpg_sample or not dpg_cfs or not dice_cfs:
            print(f"  ⚠ {dataset}: Missing required data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs) if dpg_cfs else 0}, dice_cfs: {len(dice_cfs) if dice_cfs else 0}")
            # Print available keys for debugging
            config_keys = list(dpg_run_obj.config.keys())
            summary_keys = list(dpg_run_obj.summary.keys())
            print(f"     Config keys ({len(config_keys)}): {config_keys[:15]}")
            print(f"     Summary keys ({len(summary_keys)}): {summary_keys[:15]}")
            return False
        
        # Fetch restrictions if available
        restrictions = dpg_run_obj.config.get('restrictions')
        
        # Fetch DPG per-class constraints (for visualization)
        dpg_config = dpg_run_obj.config.get('dpg', {})
        constraints = dpg_config.get('constraints') if dpg_config else None
        
        # Cache the fetched data for PCA comparison and future local-only use
        _WANDB_DATA_CACHE[dataset] = {
            'sample': dpg_sample,
            'class': dpg_class,
            'dpg_cfs': dpg_cfs,
            'dice_cfs': dice_cfs,
            'restrictions': restrictions,
            'constraints': constraints  # Add per-class min/max constraints
        }
        
        # Create heatmap comparing techniques
        fig = heatmap_techniques(
            sample=dpg_sample,
            class_sample=dpg_class,
            cf_list_1=dpg_cfs[:5],  # Limit to first 5 CFs per technique
            cf_list_2=dice_cfs[:5],
            technique_names=('DPG', 'DiCE'),
            restrictions=restrictions
        )
        
        if fig:
            output_path = os.path.join(dataset_viz_dir, 'heatmap_techniques.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {dataset}: Exported heatmap_techniques comparison")
            return True
        else:
            print(f"  ⚠ {dataset}: Could not create heatmap_techniques")
            return False
            
    except Exception as e:
        print(f"  ⚠ {dataset}: Error exporting heatmap_techniques: {e}")
        return False


def export_constraints_with_first_cfs(raw_df, dataset, dataset_viz_dir):
    """Export constraints overview combined with first CF from each method.
    
    Overlays the first DPG and DiCE counterfactuals on the constraints overview plot
    using the same markers and colors as the PCA comparison visualization.
    """
    from DPG.dpg import plot_dpg_constraints_overview
    
    # Get sample and counterfactuals from cache or local data
    if args.local_only:
        # Try to load from WandB data cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
            restrictions = cached_data.get('restrictions')
            constraints = cached_data.get('constraints')
        else:
            # Fallback to local pkl files
            try:
                dpg_data = _load_local_viz_data(dataset, 'dpg')
                dice_data = _load_local_viz_data(dataset, 'dice')
                
                if not dpg_data or not dice_data:
                    print(f"  ⚠ {dataset}: Constraints+CFs in local-only mode requires cached data")
                    return False
                
                sample = dpg_data.get('original_sample')
                restrictions = dpg_data.get('restrictions')
                constraints = dpg_data.get('constraints')
                
                dpg_cfs = []
                for viz in dpg_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dpg_cfs.append(cf)
                
                dice_cfs = []
                for viz in dice_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dice_cfs.append(cf)
            except Exception as e:
                print(f"  ⚠ {dataset}: Error loading local data: {e}")
                return False
    else:
        # WandB mode - check cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
            restrictions = cached_data.get('restrictions')
            constraints = cached_data.get('constraints')
        else:
            print(f"  ⚠ {dataset}: Data not in cache, run heatmap generation first")
            return False
    
    if not sample or not dpg_cfs or not dice_cfs or not constraints:
        print(f"  ⚠ {dataset}: Missing required data for constraints+CFs visualization")
        print(f"     Have: sample={bool(sample)}, dpg_cfs={len(dpg_cfs) if dpg_cfs else 0}, dice_cfs={len(dice_cfs) if dice_cfs else 0}, constraints={bool(constraints)}")
        return False
    
    try:
        # Load dataset to get feature names and model for predictions
        dataset_model_info = load_dataset_and_model(dataset)
        if dataset_model_info is None:
            print(f"  ⚠ {dataset}: Could not load dataset/model")
            return False
        
        model = dataset_model_info['model']
        feature_names = dataset_model_info['feature_names']
        
        # Get original and target classes
        sample_df = pd.DataFrame([sample])
        original_class = model.predict(sample_df)[0]
        
        # Predict classes for first CFs
        first_dpg_cf = dpg_cfs[0]
        first_dice_cf = dice_cfs[0]
        dpg_cf_df = pd.DataFrame([first_dpg_cf])
        dice_cf_df = pd.DataFrame([first_dice_cf])
        dpg_cf_class = model.predict(dpg_cf_df)[0]
        dice_cf_class = model.predict(dice_cf_df)[0]
        
        # Determine target class (use DPG CF class as target)
        target_class = dpg_cf_class
        
        # Define class colors (matching PCA comparison)
        class_colors_list = ['purple', 'green', 'orange']
        
        # Create constraints overview plot
        fig = plot_dpg_constraints_overview(
            normalized_constraints=constraints,
            feature_names=feature_names,
            class_colors_list=class_colors_list,
            original_sample=sample,
            original_class=original_class,
            target_class=target_class,
            title=f"{dataset}: Constraints with First Counterfactuals"
        )
        
        if fig is None:
            print(f"  ⚠ {dataset}: Could not create constraints overview")
            return False
        
        # Get the axis from the figure
        ax = fig.axes[0]
        
        # Get feature positions (y-axis) from the plot
        # The plot shows features in order, we need to map feature names to y positions
        features_with_constraints = []
        for feat in feature_names:
            has_constraint = any(
                feat in constraints.get(cname, {})
                for cname in constraints.keys()
            )
            if has_constraint:
                features_with_constraints.append(feat)
        
        n_features = len(features_with_constraints)
        y_positions = np.arange(n_features)
        
        # Create mapping from feature name to y position
        feature_to_y = {feat: y_positions[i] for i, feat in enumerate(features_with_constraints)}
        
        # Overlay markers matching PCA comparison style
        # DPG: triangle down (v), orange edge
        # DiCE: square (s), blue edge
        dpg_color = "#CC0000"  # Orange (matching PCA)
        dice_color = "#006DAC"  # Blue (matching PCA)
        marker_size = 150
        linewidth = 2.5
        
        # Plot first DPG CF values
        for feat in features_with_constraints:
            if feat in first_dpg_cf and feat in feature_to_y:
                y = feature_to_y[feat]
                x = first_dpg_cf[feat]
                ax.scatter(x, y, marker='v', s=marker_size, 
                          edgecolor=dpg_color, facecolor='none',
                          linewidths=linewidth, zorder=10)
        
        # Plot first DiCE CF values
        for feat in features_with_constraints:
            if feat in first_dice_cf and feat in feature_to_y:
                y = feature_to_y[feat]
                x = first_dice_cf[feat]
                ax.scatter(x, y, marker='s', s=marker_size,
                          edgecolor=dice_color, facecolor='none',
                          linewidths=linewidth, zorder=10)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='black', markersize=12,
                   markeredgecolor='black', markeredgewidth=linewidth,
                   label='Original Sample'),
            Line2D([0], [0], marker='v', color='w', 
                   markerfacecolor='none', markersize=12,
                   markeredgecolor=dpg_color, markeredgewidth=linewidth,
                   label='DPG CF #1'),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='none', markersize=12,
                   markeredgecolor=dice_color, markeredgewidth=linewidth,
                   label='DiCE CF #1'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.9)
        
        # Save figure
        output_path = os.path.join(dataset_viz_dir, 'constraints_with_first_cfs.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ {dataset}: Exported constraints overview with first CFs overlaid")
        return True
        
    except Exception as e:
        print(f"  ⚠ {dataset}: Error creating constraints+CFs visualization: {e}")
        traceback.print_exc()
        return False


def export_sample_cf_comparison(raw_df, dataset, dataset_viz_dir):
    """Export individual counterfactual comparison plots for each CF.
    
    Creates combined images showing DPG and DiCE counterfactuals side-by-side
    on the same image, organized by CF index.
    """
    
    # Load dataset and model
    dataset_model_info = load_dataset_and_model(dataset)
    if dataset_model_info is None:
        print(f"  ⚠ {dataset}: Could not load dataset/model, skipping sample CF comparison")
        return False
    
    model = dataset_model_info['model']
    
    # Get sample and counterfactuals from cache or local data
    if args.local_only:
        # Try to load from WandB data cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
            restrictions = cached_data.get('restrictions')
            constraints = cached_data.get('constraints')
        else:
            # Fallback to local pkl files
            try:
                dpg_data = _load_local_viz_data(dataset, 'dpg')
                dice_data = _load_local_viz_data(dataset, 'dice')
                
                if not dpg_data or not dice_data:
                    print(f"  ⚠ {dataset}: Sample CF comparison in local-only mode requires cached data")
                    return False
                
                sample = dpg_data.get('original_sample')
                restrictions = dpg_data.get('restrictions')
                constraints = dpg_data.get('constraints')
                
                dpg_cfs = []
                for viz in dpg_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dpg_cfs.append(cf)
                
                dice_cfs = []
                for viz in dice_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dice_cfs.append(cf)
            except Exception as e:
                print(f"  ⚠ {dataset}: Error loading local data: {e}")
                return False
    else:
        # WandB mode - check cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
            restrictions = cached_data.get('restrictions')
            constraints = cached_data.get('constraints')
        else:
            print(f"  ⚠ {dataset}: Data not in cache, run heatmap generation first")
            return False
    
    if not sample or not dpg_cfs or not dice_cfs:
        print(f"  ⚠ {dataset}: Missing required data for CF comparison")
        return False
    
    sample_df = pd.DataFrame([sample])
    
    # Determine max number of CFs to compare (minimum of available CFs)
    max_cfs = min(len(dpg_cfs), len(dice_cfs), 10)  # Limit to first 10 pairs
    
    # Export combined counterfactual comparison images
    combined_count = 0
    combined_simple_count = 0
    
    for i in range(max_cfs):
        try:
            # Full comparison (3 plots) - combine both methods
            dpg_cf = dpg_cfs[i]
            dice_cf = dice_cfs[i]
            
            # Create combined comparison figure
            fig = plot_sample_and_counterfactual_comparison(
                model=model,
                sample=sample,
                sample_df=sample_df,
                counterfactual=dpg_cf,
                constraints=constraints,
                class_colors_list=None,
                generation=None
            )
            
            if fig:
                output_path = os.path.join(dataset_viz_dir, f'cf_comparison_{i+1}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                # Create DiCE side
                fig_dice = plot_sample_and_counterfactual_comparison(
                    model=model,
                    sample=sample,
                    sample_df=sample_df,
                    counterfactual=dice_cf,
                    constraints=constraints,
                    class_colors_list=None,
                    generation=None
                )
                if fig_dice:
                    output_path_dice = os.path.join(dataset_viz_dir, f'cf_comparison_{i+1}_dice.png')
                    fig_dice.savefig(output_path_dice, dpi=150, bbox_inches='tight')
                    plt.close(fig_dice)
                combined_count += 1
            
            # Simple comparison (1 plot) - create combined bar graph
            fig_simple = plot_sample_and_counterfactual_comparison_combined(
                model=model,
                sample=sample,
                sample_df=sample_df,
                dpg_cf=dpg_cfs[i],
                dice_cf=dice_cfs[i],
                method_names=('DPG', 'DiCE'),
                constraints=constraints,
                restrictions=restrictions,
                class_colors_list=None,
                generation=i+1  # Pass CF index for title
            )
            
            if fig_simple:
                output_path_simple = os.path.join(dataset_viz_dir, f'cf_comparison_simple_{i+1}.png')
                fig_simple.savefig(output_path_simple, dpi=150, bbox_inches='tight')
                plt.close(fig_simple)
                combined_simple_count += 1
        except Exception as e:
            print(f"    Warning: Failed to create combined CF {i+1} comparison: {e}")
    
    if combined_count > 0 or combined_simple_count > 0:
        print(f"  ✓ {dataset}: Exported {combined_simple_count} combined simple comparisons")
        return True
    
    return False
    
    sample_df = pd.DataFrame([sample])
    
    # Export DPG counterfactuals
    dpg_count = 0
    dpg_simple_count = 0
    for i, cf in enumerate(dpg_cfs[:10]):  # Limit to first 10
        try:
            # Full comparison (3 plots)
            fig = plot_sample_and_counterfactual_comparison(
                model=model,
                sample=sample,
                sample_df=sample_df,
                counterfactual=cf,
                constraints=constraints,
                class_colors_list=None,
                generation=None
            )
            
            if fig:
                output_path = os.path.join(dpg_cf_dir, f'cf_comparison_{i+1}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                dpg_count += 1
            
            # Simple comparison (1 plot only)
            fig_simple = plot_sample_and_counterfactual_comparison_simple(
                model=model,
                sample=sample,
                sample_df=sample_df,
                counterfactual=cf,
                constraints=constraints,
                class_colors_list=None,
                generation=None
            )
            
            if fig_simple:
                output_path_simple = os.path.join(dpg_cf_dir, f'cf_comparison_simple_{i+1}.png')
                fig_simple.savefig(output_path_simple, dpi=150, bbox_inches='tight')
                plt.close(fig_simple)
                dpg_simple_count += 1
        except Exception as e:
            print(f"    Warning: Failed to create DPG CF {i+1} comparison: {e}")
    
    # Export DiCE counterfactuals
    dice_count = 0
    dice_simple_count = 0
    for i, cf in enumerate(dice_cfs[:10]):  # Limit to first 10
        try:
            # Full comparison (3 plots)
            fig = plot_sample_and_counterfactual_comparison(
                model=model,
                sample=sample,
                sample_df=sample_df,
                counterfactual=cf,
                constraints=constraints,
                class_colors_list=None,
                generation=None
            )
            
            if fig:
                output_path = os.path.join(dice_cf_dir, f'cf_comparison_{i+1}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                dice_count += 1
            
            # Simple comparison (1 plot only)
            fig_simple = plot_sample_and_counterfactual_comparison_simple(
                model=model,
                sample=sample,
                sample_df=sample_df,
                counterfactual=cf,
                constraints=constraints,
                class_colors_list=None,
                generation=None
            )
            
            if fig_simple:
                output_path_simple = os.path.join(dice_cf_dir, f'cf_comparison_simple_{i+1}.png')
                fig_simple.savefig(output_path_simple, dpi=150, bbox_inches='tight')
                plt.close(fig_simple)
                dice_simple_count += 1
        except Exception as e:
            print(f"    Warning: Failed to create DiCE CF {i+1} comparison: {e}")
    
    if dpg_count > 0 or dice_count > 0:
        print(f"  ✓ {dataset}: Exported {dpg_count} DPG + {dice_count} DiCE counterfactual comparisons")
        print(f"  ✓ {dataset}: Exported {dpg_simple_count} DPG + {dice_simple_count} DiCE simple comparisons")
        return True
    
    return False


def export_pca_comparison(raw_df, dataset, dataset_viz_dir):
    """Export PCA comparison plot with counterfactuals from both DPG and DiCE."""
    
    # Load dataset and model
    dataset_model_info = load_dataset_and_model(dataset)
    if dataset_model_info is None:
        print(f"  ⚠ {dataset}: Could not load dataset/model, skipping PCA comparison")
        return False
    
    model = dataset_model_info['model']
    full_dataset = dataset_model_info['dataset']
    target = dataset_model_info['target']
    
    # Handle local-only mode by loading from cache or disk
    if args.local_only:
        # Try to load from WandB data cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            dpg_sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
        else:
            # Fallback to local pkl files
            try:
                dpg_data = _load_local_viz_data(dataset, 'dpg')
                dice_data = _load_local_viz_data(dataset, 'dice')
                
                if not dpg_data or not dice_data:
                    print(f"  ⚠ {dataset}: PCA comparison in local-only mode requires cached data")
                    print(f"     Tip: Run without --local-only flag first to fetch and cache data from WandB")
                    return False
                
                # Extract sample and counterfactuals from local data
                dpg_sample = dpg_data.get('original_sample')
                
                # Extract counterfactuals from visualizations  
                dpg_cfs = []
                for viz in dpg_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dpg_cfs.append(cf)
                
                dice_cfs = []
                for viz in dice_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dice_cfs.append(cf)
                
            except Exception as e:
                print(f"  ⚠ {dataset}: Error loading local data for PCA comparison: {e}")
                return False
        
        if not dpg_sample or not dpg_cfs or not dice_cfs:
            print(f"  ⚠ {dataset}: Missing required data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs) if dpg_cfs else 0}, dice_cfs: {len(dice_cfs) if dice_cfs else 0}")
            return False
        
        # Convert counterfactuals to DataFrames
        try:
            dpg_cfs_df = pd.DataFrame(dpg_cfs[:5])  # Limit to first 5
            dice_cfs_df = pd.DataFrame(dice_cfs[:5])
            
            # Validate that CF features match model's expected features
            expected_features = set(dataset_model_info['feature_names'])
            cf_features = set(dpg_cfs_df.columns)
            if cf_features != expected_features:
                missing_in_cf = expected_features - cf_features
                extra_in_cf = cf_features - expected_features
                print(f"  ⚠ {dataset}: Feature mismatch between model and cached CFs, skipping PCA comparison")
                print(f"     Expected features (first 5): {list(expected_features)[:5]}")
                print(f"     CF features (first 5): {list(cf_features)[:5]}")
                if missing_in_cf:
                    print(f"     Missing in CF: {list(missing_in_cf)[:5]}...")
                if extra_in_cf:
                    print(f"     Extra in CF: {list(extra_in_cf)[:5]}...")
                print(f"     This usually means the WandB run has mismatched data. Check run configs.")
                return False
            
            # Predict classes for counterfactuals
            dpg_cf_classes = model.predict(dpg_cfs_df)
            dice_cf_classes = model.predict(dice_cfs_df)

            print(f"  → {dataset}: Creating PCA comparison plot from cached local data...")
            print(f"     predictions: DPG CF classes: {dpg_cf_classes}, DiCE CF classes: {dice_cf_classes}")
            
            # Create PCA comparison plot
            fig = plot_pca_with_counterfactuals_comparison(
                model=model,
                dataset=full_dataset,
                target=target,
                sample=dpg_sample,
                counterfactuals_df_1=dpg_cfs_df,
                cf_predicted_classes_1=dpg_cf_classes,
                counterfactuals_df_2=dice_cfs_df,
                cf_predicted_classes_2=dice_cf_classes,
                method_1_name='DPG',
                method_2_name='DiCE'
            )
            
            if fig:
                output_path = os.path.join(dataset_viz_dir, 'pca_comparison.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {dataset}: Exported PCA comparison (from cached data)")
                return True
            else:
                return False
        except Exception as e:
            print(f"  ⚠ {dataset}: Error creating PCA comparison: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # WandB mode - check if data is already cached first (from export_heatmap_techniques)
    if dataset in _WANDB_DATA_CACHE:
        cached_data = _WANDB_DATA_CACHE[dataset]
        dpg_sample = cached_data['sample']
        dpg_cfs = cached_data['dpg_cfs']
        dice_cfs = cached_data['dice_cfs']
        
        # Convert counterfactuals to DataFrames
        dpg_cfs_df = pd.DataFrame(dpg_cfs[:5])  # Limit to first 5
        dice_cfs_df = pd.DataFrame(dice_cfs[:5])
        
        # Validate that CF features match model's expected features
        expected_features = set(dataset_model_info['feature_names'])
        cf_features = set(dpg_cfs_df.columns)
        if cf_features != expected_features:
            missing_in_cf = expected_features - cf_features
            extra_in_cf = cf_features - expected_features
            print(f"  ⚠ {dataset}: Feature mismatch between model and cached CFs, skipping PCA comparison")
            print(f"     Expected features (first 5): {list(expected_features)[:5]}")
            print(f"     CF features (first 5): {list(cf_features)[:5]}")
            if missing_in_cf:
                print(f"     Missing in CF: {list(missing_in_cf)[:5]}...")
            if extra_in_cf:
                print(f"     Extra in CF: {list(extra_in_cf)[:5]}...")
            print(f"     This usually means the WandB run has mismatched data. Check run configs.")
            return False
        
        # Predict classes for counterfactuals
        dpg_cf_classes = model.predict(dpg_cfs_df)
        dice_cf_classes = model.predict(dice_cfs_df)
        
        # Create PCA comparison plot
        fig = plot_pca_with_counterfactuals_comparison(
            model=model,
            dataset=full_dataset,
            target=target,
            sample=dpg_sample,
            counterfactuals_df_1=dpg_cfs_df,
            cf_predicted_classes_1=dpg_cf_classes,
            counterfactuals_df_2=dice_cfs_df,
            cf_predicted_classes_2=dice_cf_classes,
            method_1_name='DPG',
            method_2_name='DiCE'
        )
        
        if fig:
            output_path = os.path.join(dataset_viz_dir, 'pca_comparison.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {dataset}: Exported PCA comparison (from cache)")
            return True
        return False
    
    # Fetch from WandB if not cached
    try:
        dataset_runs = raw_df[raw_df['dataset'] == dataset]
        
        dpg_run = dataset_runs[dataset_runs['technique'] == 'dpg'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dpg']) > 0 else None
        dice_run = dataset_runs[dataset_runs['technique'] == 'dice'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dice']) > 0 else None
        
        if dpg_run is None or dice_run is None:
            print(f"  ⚠ {dataset}: Missing DPG or DiCE run data, skipping PCA comparison")
            return False
        
        api = wandb.Api(timeout=60)
        dpg_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dpg_run['run_id']}")
        dice_run_obj = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{dice_run['run_id']}")
        
        # Fetch feature names (needed to convert lists to dicts)
        feature_names = dpg_run_obj.config.get('feature_names', [])
        
        # Fetch sample
        dpg_sample = dpg_run_obj.config.get('sample')
        if not dpg_sample:
            dpg_sample = dpg_run_obj.summary.get('sample')
        if not dpg_sample:
            dpg_sample = dpg_run_obj.summary.get('original_sample')
        
        # Get counterfactuals from both runs
        dpg_cfs = dpg_run_obj.summary.get('final_counterfactuals', [])
        if isinstance(dpg_cfs, str):
            try:
                dpg_cfs = json.loads(dpg_cfs)
            except:
                dpg_cfs = []
        
        dice_cfs = dice_run_obj.summary.get('final_counterfactuals', [])
        if isinstance(dice_cfs, str):
            try:
                dice_cfs = json.loads(dice_cfs)
            except:
                dice_cfs = []
        
        # Convert list format to dict format if needed
        if feature_names:
            # Convert sample from list to dict
            if isinstance(dpg_sample, list):
                dpg_sample = dict(zip(feature_names, dpg_sample))
            
            # Convert counterfactuals from list of lists to list of dicts
            if dpg_cfs and isinstance(dpg_cfs[0], list):
                dpg_cfs = [dict(zip(feature_names, cf)) for cf in dpg_cfs]
            if dice_cfs and isinstance(dice_cfs[0], list):
                dice_cfs = [dict(zip(feature_names, cf)) for cf in dice_cfs]
        
        if not dpg_sample or not dpg_cfs or not dice_cfs:
            print(f"  ⚠ {dataset}: Missing required data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs) if dpg_cfs else 0}, dice_cfs: {len(dice_cfs) if dice_cfs else 0}")
            return False
        
        # Cache the data for future use (e.g., if heatmap_techniques hasn't cached it yet)
        if dataset not in _WANDB_DATA_CACHE:
            _WANDB_DATA_CACHE[dataset] = {
                'sample': dpg_sample,
                'dpg_cfs': dpg_cfs,
                'dice_cfs': dice_cfs,
            }
        
        # Convert counterfactuals to DataFrames
        dpg_cfs_df = pd.DataFrame(dpg_cfs[:5])  # Limit to first 5
        dice_cfs_df = pd.DataFrame(dice_cfs[:5])
        
        # Predict classes for counterfactuals
        dpg_cf_classes = model.predict(dpg_cfs_df)
        dice_cf_classes = model.predict(dice_cfs_df)
        
        # Create PCA comparison plot
        fig = plot_pca_with_counterfactuals_comparison(
            model=model,
            dataset=full_dataset,
            target=target,
            sample=dpg_sample,
            counterfactuals_df_1=dpg_cfs_df,
            cf_predicted_classes_1=dpg_cf_classes,
            counterfactuals_df_2=dice_cfs_df,
            cf_predicted_classes_2=dice_cf_classes,
            method_1_name='DPG',
            method_2_name='DiCE'
        )
        
        if fig:
            output_path = os.path.join(dataset_viz_dir, 'pca_comparison.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {dataset}: Exported PCA comparison")
            return True
        else:
            print(f"  ⚠ {dataset}: Could not create PCA comparison")
            return False
            
    except Exception as e:
        print(f"  ⚠ {dataset}: Error exporting PCA comparison: {e}")
        return False


def export_ridge_comparison(raw_df, dataset, dataset_viz_dir):
    """Export ridge plot comparing feature distributions from DPG and DiCE counterfactuals.
    
    Creates a ridge plot (joy plot) where each row is a feature, showing the distribution
    of values for counterfactuals from both DPG and DiCE methods, with the original sample
    marked as a reference point.
    
    Args:
        raw_df: DataFrame with run metadata
        dataset: Name of the dataset
        dataset_viz_dir: Output directory for visualizations
    
    Returns:
        bool: True if export succeeded, False otherwise
    """
    # Handle local-only mode by loading from cache or disk
    if args.local_only:
        # Try to load from WandB data cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
        else:
            # Fallback to local pkl files
            try:
                dpg_data = _load_local_viz_data(dataset, 'dpg')
                dice_data = _load_local_viz_data(dataset, 'dice')
                
                if not dpg_data or not dice_data:
                    print(f"  ⚠ {dataset}: Ridge plot in local-only mode requires cached data")
                    return False
                
                sample = dpg_data.get('original_sample')
                
                dpg_cfs = []
                for viz in dpg_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dpg_cfs.append(cf)
                
                dice_cfs = []
                for viz in dice_data.get('visualizations', []):
                    for cf_data in viz.get('counterfactuals', []):
                        cf = cf_data.get('counterfactual')
                        if cf:
                            dice_cfs.append(cf)
            except Exception as e:
                print(f"  ⚠ {dataset}: Error loading local data for ridge plot: {e}")
                return False
    else:
        # WandB mode - check cache first
        if dataset in _WANDB_DATA_CACHE:
            cached_data = _WANDB_DATA_CACHE[dataset]
            sample = cached_data['sample']
            dpg_cfs = cached_data['dpg_cfs']
            dice_cfs = cached_data['dice_cfs']
        else:
            print(f"  ⚠ {dataset}: Data not in cache, run heatmap generation first")
            return False
    
    if not sample or not dpg_cfs or not dice_cfs:
        print(f"  ⚠ {dataset}: Missing required data for ridge plot")
        return False
    
    # Need at least 2 counterfactuals from each method for meaningful KDE
    if len(dpg_cfs) < 2 or len(dice_cfs) < 2:
        print(f"  ⚠ {dataset}: Not enough counterfactuals for ridge plot (DPG: {len(dpg_cfs)}, DiCE: {len(dice_cfs)})")
        return False
    
    # Get DPG constraints from cache
    dpg_constraints = None
    if dataset in _WANDB_DATA_CACHE:
        dpg_constraints = _WANDB_DATA_CACHE[dataset].get('constraints')
    
    # Load dataset for distribution
    dataset_model_info = load_dataset_and_model(dataset)
    if dataset_model_info is None:
        print(f"  ⚠ {dataset}: Could not load dataset for ridge plot")
        return False
    
    dataset_df = dataset_model_info['dataset']
    
    try:
        # Create ridge plot
        fig = plot_ridge_comparison(
            sample=sample,
            cf_list_1=dpg_cfs,
            cf_list_2=dice_cfs,
            technique_names=('DPG', 'DiCE'),
            method_1_color="#CC0000",  # Orange for DPG
            method_2_color="#006DAC",  # Blue for DiCE
            dataset_df=dataset_df,
            constraints=dpg_constraints
        )
        
        if fig:
            output_path = os.path.join(dataset_viz_dir, 'ridge_comparison.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {dataset}: Exported ridge comparison plot")
            return True
        else:
            print(f"  ⚠ {dataset}: Could not create ridge plot")
            return False
            
    except Exception as e:
        print(f"  ⚠ {dataset}: Error exporting ridge plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_winner_heatmap_csv(comparison_df, metrics_to_include=None, filename_suffix=''):
    """Export winner heatmap data to CSV.
    
    Creates a CSV where rows are datasets and columns are metrics.
    Values are 'DPG', 'DiCE', or 'Tie' indicating which technique won.
    """
    # Filter metrics if specified
    if metrics_to_include:
        metrics_to_use = {k: v for k, v in COMPARISON_METRICS.items() 
                         if k in metrics_to_include}
    else:
        metrics_to_use = COMPARISON_METRICS
    
    # Create winner matrix
    winners_data = []
    
    for _, row in comparison_df.iterrows():
        dataset = row['dataset']
        winners_row = {'Dataset': dataset}
        
        for metric_key, metric_info in metrics_to_use.items():
            dpg_col = f'{metric_key}_dpg'
            dice_col = f'{metric_key}_dice'
            
            if dpg_col in row.index and dice_col in row.index:
                dpg_val = row[dpg_col]
                dice_val = row[dice_col]
                
                winner = determine_winner(dpg_val, dice_val, metric_info['goal'])
                # Convert to uppercase for display
                if winner == 'dpg':
                    winners_row[metric_info['name']] = 'DPG'
                elif winner == 'dice':
                    winners_row[metric_info['name']] = 'DiCE'
                else:
                    winners_row[metric_info['name']] = 'Tie'
            else:
                winners_row[metric_info['name']] = 'N/A'
        
        winners_data.append(winners_row)
    
    # Create DataFrame
    winners_df = pd.DataFrame(winners_data)
    winners_df = winners_df.set_index('Dataset')
    
    # Export to CSV
    output_path = os.path.join(OUTPUT_DIR, f'winner_heatmap{filename_suffix}.csv')
    winners_df.to_csv(output_path)
    print(f"✓ Exported winner heatmap data to: {output_path}")
    
    return winners_df


def export_comparison_numeric_csv(comparison_df, metrics_to_include=None, filename_suffix=''):
    """Export numeric comparison data to CSV.
    
    Creates a CSV where rows are datasets and columns are metrics.
    Each metric column has two sub-columns: DPG and DiCE with actual numeric values.
    """
    # Filter metrics if specified
    if metrics_to_include:
        metrics_to_use = {k: v for k, v in COMPARISON_METRICS.items() 
                         if k in metrics_to_include}
    else:
        metrics_to_use = COMPARISON_METRICS
    
    # Create numeric data matrix
    numeric_data = []
    
    for _, row in comparison_df.iterrows():
        dataset = row['dataset']
        numeric_row = {'Dataset': dataset}
        
        for metric_key, metric_info in metrics_to_use.items():
            dpg_col = f'{metric_key}_dpg'
            dice_col = f'{metric_key}_dice'
            
            if dpg_col in row.index and dice_col in row.index:
                dpg_val = row[dpg_col]
                dice_val = row[dice_col]
                
                # Store numeric values, handling NaN
                numeric_row[f"{metric_info['name']}_DPG"] = \
                    f"{dpg_val:.4f}" if pd.notna(dpg_val) else 'N/A'
                numeric_row[f"{metric_info['name']}_DiCE"] = \
                    f"{dice_val:.4f}" if pd.notna(dice_val) else 'N/A'
            else:
                numeric_row[f"{metric_info['name']}_DPG"] = 'N/A'
                numeric_row[f"{metric_info['name']}_DiCE"] = 'N/A'
        
        numeric_data.append(numeric_row)
    
    # Create DataFrame
    numeric_df = pd.DataFrame(numeric_data)
    numeric_df = numeric_df.set_index('Dataset')
    
    # Export to CSV
    output_path = os.path.join(OUTPUT_DIR, f'comparison_numeric{filename_suffix}.csv')
    numeric_df.to_csv(output_path)
    print(f"✓ Exported numeric comparison data to: {output_path}")
    
    return numeric_df


def export_dataset_visualizations(comparison_df, raw_df):
    """Export dataset-specific visualizations organized by dataset."""
    print("\n" + "="*80)
    print("EXPORTING DATASET-SPECIFIC VISUALIZATIONS")
    print("="*80)
    
    viz_base_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_base_dir, exist_ok=True)
    
    available_datasets = sorted(comparison_df['dataset'].unique())
    
    for dataset in available_datasets:
        # Create subdirectory for this dataset
        safe_name = dataset.replace('/', '_').replace(' ', '_')
        dataset_viz_dir = os.path.join(viz_base_dir, safe_name)
        os.makedirs(dataset_viz_dir, exist_ok=True)
        
        viz_files = []
        
        # Export heatmap comparing DPG vs DiCE counterfactuals
        export_heatmap_techniques(raw_df, dataset, dataset_viz_dir)
        
        # Export constraints overview with first CFs (only for DATASET_FOR_CF_COMPARISON)
        if dataset == DATASET_FOR_CF_COMPARISON:
            export_constraints_with_first_cfs(raw_df, dataset, dataset_viz_dir)
        
        # Export PCA comparison (loads dataset and model automatically)
        export_pca_comparison(raw_df, dataset, dataset_viz_dir)
        
        # Export individual counterfactual comparisons for specified dataset
        if dataset == DATASET_FOR_CF_COMPARISON:
            export_sample_cf_comparison(raw_df, dataset, dataset_viz_dir)
        
        # Export ridge comparison plot for specified dataset
        if dataset == DATASET_FOR_CF_COMPARISON:
            export_ridge_comparison(raw_df, dataset, dataset_viz_dir)
        
        # Fetch WandB visualizations (comparison, pca_clean, heatmap)
        wandb_viz = fetch_wandb_visualizations(raw_df, dataset, dataset_viz_dir)
        if wandb_viz:
            viz_files.extend(wandb_viz)
        
        # Export radar chart for this dataset
        # from scripts.compare_techniques import plot_radar_chart
        # radar_path = os.path.join(dataset_viz_dir, f'radar.png')
        # fig = plot_radar_chart(comparison_df, dataset, figsize=(8, 8))
        # if fig:
        #     fig.savefig(radar_path, dpi=150, bbox_inches='tight')
        #     plt.close(fig)
        #     viz_files.append('radar.png')
        
        # Export bar charts for each metric (small metrics only)
        small_metrics = {
            # 'perc_valid_cf_all',
            # 'perc_actionable_cf_all',
            # 'plausibility_nbr_cf',
            # 'distance_mh',
            # 'avg_nbr_changes',
            # 'count_diversity_all',
            # 'accuracy_knn_sklearn',
            # 'runtime'
        }
        
        from scripts.compare_techniques import plot_grouped_bar_chart
        metrics_exported = []
        for metric_key in small_metrics:
            if f'{metric_key}_dpg' in comparison_df.columns and f'{metric_key}_dice' in comparison_df.columns:
                metric_info = COMPARISON_METRICS.get(metric_key)
                if metric_info:
                    bar_path = os.path.join(dataset_viz_dir, f'bar_{metric_key}.png')
                    fig = plot_grouped_bar_chart(
                        comparison_df,
                        metric_key,
                        output_path=bar_path,
                        figsize=(10, 6)
                    )
                    if fig:
                        plt.close(fig)
                        viz_files.append(f'bar_{metric_key}.png')
                        metrics_exported.append(metric_key)
        
        if metrics_exported:
            viz_files.extend([f'bar_{m}.png' for m in metrics_exported])
        
        # Count total visualizations (check directory)
        total_viz = len([f for f in os.listdir(dataset_viz_dir) if f.endswith('.png')])
        print(f"  ✓ {dataset}: {total_viz} visualizations exported")


def export_comparison_summary(comparison_df):
    """Export console comparison summary to text file."""
    print("\n" + "="*80)
    print("EXPORTING COMPARISON SUMMARY TO TEXT FILE")
    print("="*80)
    
    output_path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
    
    # Capture stdout to write to file
    from io import StringIO
    import contextlib
    
    with open(output_path, 'w') as f:
        with contextlib.redirect_stdout(f):
            print_comparison_summary(comparison_df)
    
    print(f"✓ Exported comparison summary to: {output_path}")


def count_actionability_constraints(dataset_name):
    """Count the number of actionability restrictions for a dataset.
    
    Only counts non-actionable constraints (non_decreasing, non_increasing, no_change).
    Features with 'actionable' or None are not counted as restrictions.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Number of restrictions or None if not found
    """
    try:
        config_path = os.path.join(REPO_ROOT, 'configs', dataset_name, 'config.yaml')
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate to actionability constraints
        actionability = config.get('methods', {}).get('_default', {}).get('actionability', {})
        
        if not actionability:
            return 0
        
        # Count only non-actionable constraints (restrictions)
        # Exclude 'actionable' and None as they represent no restriction
        restriction_types = ['non_decreasing', 'non_increasing', 'no_change']
        count = sum(1 for v in actionability.values() 
                   if v is not None and v in restriction_types)
        return count
        
    except Exception as e:
        return None


def get_dpg_constraints_data(dataset_name):
    """Get raw DPG constraints data for a dataset.
    
    Extracts the DPG-learned constraints (min/max bounds per feature per class)
    from multiple sources. Tries:
    1. Local filesystem (outputs/{dataset}_dpg/)
    2. WandB data cache (if available)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of constraints or None if not found
    """
    try:
        constraints = None
        
        # Try source 1: Local filesystem
        outputs_base = os.path.join(REPO_ROOT, 'outputs')
        dpg_dir = os.path.join(outputs_base, f"{dataset_name}_dpg")
        
        if os.path.exists(dpg_dir):
            # Find the most recent sample directory (highest number)
            sample_dirs = []
            for name in os.listdir(dpg_dir):
                sample_path = os.path.join(dpg_dir, name)
                if os.path.isdir(sample_path):
                    try:
                        sample_dirs.append((int(name), sample_path))
                    except ValueError:
                        continue
            
            if sample_dirs:
                # Get most recent (highest sample ID)
                sample_dirs.sort(key=lambda x: x[0], reverse=True)
                _, sample_dir = sample_dirs[0]
                
                # Load constraints from JSON
                constraints_path = os.path.join(sample_dir, 'dpg_constraints_normalized.json')
                if os.path.exists(constraints_path):
                    with open(constraints_path, 'r') as f:
                        constraints = json.load(f)
        
        # Try source 2: WandB data cache
        if not constraints and dataset_name in _WANDB_DATA_CACHE:
            cached_constraints = _WANDB_DATA_CACHE[dataset_name].get('constraints')
            if cached_constraints:
                constraints = cached_constraints
        
        return constraints
        
    except Exception as e:
        print(f"    Warning: Error loading DPG constraints for {dataset_name}: {e}")
        return None


def get_dpg_constraints_formatted(dataset_name):
    """Get DPG constraints formatted for LaTeX table display.
    
    Extracts the DPG-learned constraints (min/max bounds per feature per class)
    from multiple sources and formats them compactly. Tries:
    1. Local filesystem (outputs/{dataset}_dpg/)
    2. WandB data cache (if available)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Formatted string for LaTeX table or "None" if no constraints
    """
    try:
        constraints = None
        
        # Try source 1: Local filesystem
        outputs_base = os.path.join(REPO_ROOT, 'outputs')
        dpg_dir = os.path.join(outputs_base, f"{dataset_name}_dpg")
        
        if os.path.exists(dpg_dir):
            # Find the most recent sample directory (highest number)
            sample_dirs = []
            for name in os.listdir(dpg_dir):
                sample_path = os.path.join(dpg_dir, name)
                if os.path.isdir(sample_path):
                    try:
                        sample_dirs.append((int(name), sample_path))
                    except ValueError:
                        continue
            
            if sample_dirs:
                # Get most recent (highest sample ID)
                sample_dirs.sort(key=lambda x: x[0], reverse=True)
                _, sample_dir = sample_dirs[0]
                
                # Load constraints from JSON
                constraints_path = os.path.join(sample_dir, 'dpg_constraints_normalized.json')
                if os.path.exists(constraints_path):
                    with open(constraints_path, 'r') as f:
                        constraints = json.load(f)
        
        # Try source 2: WandB data cache
        if not constraints and dataset_name in _WANDB_DATA_CACHE:
            cached_constraints = _WANDB_DATA_CACHE[dataset_name].get('constraints')
            if cached_constraints:
                constraints = cached_constraints
        
        if not constraints:
            return "None"
        
        # Extract features that have min/max bounds across all classes
        # Format: "Feature: [min, max]" for features with non-trivial bounds
        features_with_bounds = {}
        all_features = set()
        
        for class_name, class_constraints in constraints.items():
            for feature, bounds in class_constraints.items():
                all_features.add(feature)
                if feature not in features_with_bounds:
                    features_with_bounds[feature] = {
                        'min': [],
                        'max': []
                    }
                
                min_val = bounds.get('min')
                max_val = bounds.get('max')
                
                if min_val is not None:
                    features_with_bounds[feature]['min'].append(min_val)
                if max_val is not None:
                    features_with_bounds[feature]['max'].append(max_val)
        
        # Format constraints compactly - only show features with bounds
        constraints_list = []
        for feature in sorted(all_features):
            bounds = features_with_bounds[feature]
            
            # Skip features with no bounds
            if not bounds['min'] and not bounds['max']:
                continue
            
            # Clean feature name for LaTeX
            feature_clean = feature.replace('_', '\\_')
            
            # Simplify - just count how many features have bounds
            constraints_list.append(feature_clean)
        
        if not constraints_list:
            return "None"
        
        # For compact display, just show count of constrained features
        num_constrained = len(constraints_list)
        
        # If few features, list them; otherwise just show count
        if num_constrained <= 4:
            # List features with line breaks for readability
            if num_constrained <= 2:
                return ", ".join(constraints_list)
            else:
                result = []
                for i, feature in enumerate(constraints_list):
                    result.append(feature)
                    if (i + 1) % 2 == 0 and i < num_constrained - 1:
                        result.append("\\newline ")
                    elif i < num_constrained - 1:
                        result.append(", ")
                return "".join(result)
        else:
            # Show count + first few features
            first_few = ", ".join(constraints_list[:3])
            return f"{num_constrained} features\\newline ({first_few}, ...)"
        
    except Exception as e:
        print(f"    Warning: Error loading DPG constraints for {dataset_name}: {e}")
        return "None"


def export_dpg_constraints(comparison_df):
    """Export DPG constraints for all datasets as JSON and CSV files.
    
    Exports the DPG-learned constraints (min/max bounds per feature per class)
    to individual files per dataset. Tries multiple sources:
    1. Local filesystem (outputs/{dataset}_dpg/)
    2. WandB data cache (if available)
    3. WandB API (if not in local-only mode)
    """
    print("\n" + "="*80)
    print("EXPORTING DPG CONSTRAINTS")
    print("="*80)
    
    datasets = sorted(comparison_df['dataset'].unique()) if len(comparison_df) > 0 else []
    
    if len(datasets) == 0:
        print("⚠ No datasets found, skipping DPG constraints export")
        return
    
    # Create constraints subdirectory
    constraints_dir = os.path.join(OUTPUT_DIR, 'dpg_constraints')
    os.makedirs(constraints_dir, exist_ok=True)
    
    constraints_found = 0
    constraints_not_found = 0
    
    for dataset in datasets:
        constraints = None
        source = None
        
        # Try source 1: Local filesystem
        outputs_base = os.path.join(REPO_ROOT, 'outputs')
        dpg_dir = os.path.join(outputs_base, f"{dataset}_dpg")
        
        if os.path.exists(dpg_dir):
            # Find the most recent sample directory (highest number)
            sample_dirs = []
            for name in os.listdir(dpg_dir):
                sample_path = os.path.join(dpg_dir, name)
                if os.path.isdir(sample_path):
                    try:
                        sample_dirs.append((int(name), sample_path))
                    except ValueError:
                        continue
            
            if sample_dirs:
                # Get most recent (highest sample ID)
                sample_dirs.sort(key=lambda x: x[0], reverse=True)
                sample_id, sample_dir = sample_dirs[0]
                
                # Load constraints from JSON
                constraints_path = os.path.join(sample_dir, 'dpg_constraints_normalized.json')
                if os.path.exists(constraints_path):
                    try:
                        with open(constraints_path, 'r') as f:
                            constraints = json.load(f)
                        source = f"local filesystem (sample {sample_id})"
                    except Exception as e:
                        print(f"  ⚠ {dataset}: Error reading local constraints: {e}")
        
        # Try source 2: WandB data cache
        if not constraints and dataset in _WANDB_DATA_CACHE:
            cached_constraints = _WANDB_DATA_CACHE[dataset].get('constraints')
            if cached_constraints:
                constraints = cached_constraints
                source = "WandB cache"
        
        # Try source 3: WandB API (if not in local-only mode)
        if not constraints and not args.local_only:
            try:
                # Get DPG run for this dataset
                dataset_runs = comparison_df[comparison_df['dataset'] == dataset]
                dpg_runs = dataset_runs[dataset_runs['technique'] == 'dpg']
                
                if len(dpg_runs) > 0:
                    run_id = dpg_runs.iloc[0]['run_id']
                    api = wandb.Api()
                    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
                    
                    dpg_config = run.config.get('dpg', {})
                    api_constraints = dpg_config.get('constraints')
                    
                    if api_constraints:
                        constraints = api_constraints
                        source = "WandB API"
            except Exception as e:
                print(f"  ⚠ {dataset}: Error fetching from WandB API: {e}")
        
        # Export constraints if found
        if not constraints:
            print(f"  ✗ {dataset}: No constraints found")
            constraints_not_found += 1
            continue
        
        try:
            if not isinstance(constraints, dict) or len(constraints) == 0:
                print(f"  ✗ {dataset}: Empty or invalid constraints")
                constraints_not_found += 1
                continue
            
            # Export as JSON
            json_output_path = os.path.join(constraints_dir, f"{dataset}_dpg_constraints.json")
            with open(json_output_path, 'w') as f:
                json.dump(constraints, f, indent=2)
            
            # Export as CSV (flattened format)
            csv_output_path = os.path.join(constraints_dir, f"{dataset}_dpg_constraints.csv")
            csv_rows = []
            
            for class_name, class_constraints in constraints.items():
                if isinstance(class_constraints, dict):
                    for feature, bounds in class_constraints.items():
                        if isinstance(bounds, dict):
                            csv_rows.append({
                                'class': class_name,
                                'feature': feature,
                                'min': bounds.get('min'),
                                'max': bounds.get('max')
                            })
            
            if csv_rows:
                df_constraints = pd.DataFrame(csv_rows)
                df_constraints.to_csv(csv_output_path, index=False)
            
            num_features = len(set(row['feature'] for row in csv_rows))
            print(f"  ✓ {dataset}: Exported {num_features} constrained features from {source}")
            constraints_found += 1
            
        except Exception as e:
            print(f"  ✗ {dataset}: Error exporting constraints: {e}")
            constraints_not_found += 1
            continue
    
    print(f"\n✓ Exported DPG constraints for {constraints_found} datasets")
    print(f"  Directory: {constraints_dir}")
    if constraints_not_found > 0:
        print(f"  ⚠ {constraints_not_found} datasets had no constraints available")


def export_model_information(raw_df, comparison_df):
    """Export Random Forest model information for each dataset.
    
    Exports information about:
    - Train/test split configuration
    - Train and test accuracies
    - Model hyperparameters
    - Dataset statistics
    
    Works in both online and local-only modes.
    """
    print("\n" + "="*80)
    print("EXPORTING RANDOM FOREST MODEL INFORMATION")
    print("="*80)
    
    datasets = sorted(comparison_df['dataset'].unique()) if len(comparison_df) > 0 else []
    if len(datasets) == 0:
        datasets = sorted(raw_df['dataset'].unique()) if len(raw_df) > 0 else []
    
    if len(datasets) == 0:
        print("⚠ No datasets found, skipping model information export")
        return
    
    model_info_data = []
    
    for dataset in datasets:
        print(f"  Processing {dataset}...")
        
        # Load dataset and train model (or get from cache)
        dataset_model_info = load_dataset_and_model(dataset)
        
        if dataset_model_info is None:
            print(f"  ⚠ {dataset}: Could not load dataset/model")
            continue
        
        # Extract model parameters for display
        model_params = dataset_model_info.get('model_params', {})
        
        # Create info record
        info = {
            'Dataset': dataset,
            'Train Size': len(dataset_model_info['train_features']),
            'Test Size': len(dataset_model_info['test_features']),
            'Total Size': len(dataset_model_info['dataset']),
            'Test Size %': f"{dataset_model_info['test_size'] * 100:.1f}%",
            'Train Accuracy': f"{dataset_model_info['train_accuracy']:.4f}",
            'Test Accuracy': f"{dataset_model_info['test_accuracy']:.4f}",
            'Random State': dataset_model_info['random_state'],
            'Num Features': len(dataset_model_info['feature_names']),
            'Num Classes': len(np.unique(dataset_model_info['target'])),
        }
        
        # Add key RF hyperparameters
        for param_name in ['n_estimators', 'max_depth', 'min_samples_split', 
                          'min_samples_leaf', 'max_features', 'random_state']:
            if param_name in model_params:
                info[f'RF_{param_name}'] = model_params[param_name]
        
        model_info_data.append(info)
        print(f"  ✓ {dataset}: Train Acc={info['Train Accuracy']}, Test Acc={info['Test Accuracy']}")
    
    # Create DataFrame
    model_info_df = pd.DataFrame(model_info_data)
    
    # Export to CSV
    output_path = os.path.join(OUTPUT_DIR, 'rf_model_information.csv')
    model_info_df.to_csv(output_path, index=False)
    print(f"\n✓ Exported RF model information to: {output_path}")
    print(f"  {len(model_info_data)} datasets processed")
    
    # Export LaTeX table - First, the summary table
    latex_path = os.path.join(OUTPUT_DIR, 'dataset_overview.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("  \\centering\n")
        f.write(f"  \\caption{{Overview of the {len(model_info_data)} benchmark datasets employed in the experimental evaluation, ")
        f.write("including the number of features, samples, classes, train/test split, accuracy (Acc), and number of actionability rules per dataset}\n")
        f.write("  \\label{tab:datasets}\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{lrrrlrrp{1.8cm}}\n")
        f.write("    \\toprule\n")
        f.write("    Dataset & Features & Samples & Classes & Train/Test & Train Acc. & Test Acc. & Actionability Rules \\\\\n")
        f.write("    \\midrule\n")
        
        # Sort datasets alphabetically
        sorted_data = sorted(model_info_data, key=lambda x: x['Dataset'])
        
        for info in sorted_data:
            dataset = info['Dataset']
            num_features = info['Num Features']
            total_samples = info['Total Size']
            num_classes = info['Num Classes']
            train_size = info['Train Size']
            test_size = info['Test Size']
            train_acc = info['Train Accuracy']
            test_acc = info['Test Accuracy']
            
            # Format train/test split as ratio
            train_test_split = f"{train_size}/{test_size}"
            
            # Count actionability restrictions (only non-actionable constraints)
            num_restrictions = count_actionability_constraints(dataset)
            restrictions_str = str(num_restrictions) if num_restrictions is not None and num_restrictions > 0 else "None"
            
            # Format numbers with thousands separator
            total_samples_str = f"{total_samples:,}"
            
            # Clean dataset name for LaTeX (replace underscores)
            dataset_latex = dataset.replace('_', '\\_')
            
            f.write(f"    {dataset_latex} & {num_features} & {total_samples_str} & {num_classes} & "
                   f"{train_test_split} & {train_acc} & {test_acc} & {restrictions_str} \\\\\n")
        
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}%\n")
        f.write("  }\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Exported LaTeX table to: {latex_path}")
    
    # Export detailed DPG constraints table in landscape mode, one dataset per page
    latex_constraints_path = os.path.join(OUTPUT_DIR, 'dataset_dpg_constraints.tex')
    with open(latex_constraints_path, 'w') as f:
        # Write header once
        f.write("% DPG Constraints for all datasets - one dataset per page\n")
        f.write("% Each table shows the min/max bounds learned by DPG for each feature and class\n\n")
        
        for info in sorted_data:
            dataset = info['Dataset']
            dataset_latex = dataset.replace('_', '\\_')
            
            # Get DPG constraints for this dataset
            constraints = get_dpg_constraints_data(dataset)
            
            if not constraints:
                continue
            
            # Start landscape page for this dataset
            f.write("\\begin{landscape}\n")
            f.write("\\begin{table}[p]\n")
            f.write("  \\centering\n")
            f.write(f"  \\caption{{DPG-learned constraints (min/max bounds) for {dataset_latex} dataset. ")
            f.write("These constraints define valid regions in the feature space for each class ")
            f.write("and are used to guide counterfactual generation.}\n")
            f.write(f"  \\label{{tab:dpg-constraints-{dataset}}}\n")
            f.write("  \\small\n")
            
            # Build table with classes as columns
            classes = sorted(constraints.keys())
            num_classes = len(classes)
            
            # Column specification: Feature name + (min, max) pair for each class
            col_spec = "l" + "rr" * num_classes
            f.write(f"  \\begin{{tabular}}{{{col_spec}}}\n")
            f.write("    \\toprule\n")
            
            # Header row 1: Feature and class labels
            header1 = "    Feature"
            for class_label in classes:
                class_latex = str(class_label).replace('_', '\\_')
                header1 += f" & \\multicolumn{{2}}{{c}}{{{class_latex}}}"
            header1 += " \\\\\n"
            f.write(header1)
            
            # Header row 2: Min/Max labels under each class
            header2 = "    "
            for _ in classes:
                header2 += " & Min & Max"
            header2 += " \\\\\n"
            f.write(header2)
            f.write("    \\midrule\n")
            
            # Collect all features across all classes
            all_features = set()
            for class_constraints in constraints.values():
                all_features.update(class_constraints.keys())
            
            # Write data rows (one per feature)
            for feature in sorted(all_features):
                feature_latex = feature.replace('_', '\\_')
                row = f"    {feature_latex}"
                
                for class_label in classes:
                    class_constraints = constraints[class_label]
                    bounds = class_constraints.get(feature, {})
                    
                    min_val = bounds.get('min')
                    max_val = bounds.get('max')
                    
                    # Format values nicely
                    min_str = f"{min_val:.2f}" if min_val is not None else "---"
                    max_str = f"{max_val:.2f}" if max_val is not None else "---"
                    
                    row += f" & {min_str} & {max_str}"
                
                row += " \\\\\n"
                f.write(row)
            
            f.write("    \\bottomrule\n")
            f.write("  \\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("\\end{landscape}\n")
            f.write("\\clearpage\n\n")
    
    print(f"✓ Exported DPG constraints table to: {latex_constraints_path}")
    
    # Export individual LaTeX files per dataset
    individual_constraints_dir = os.path.join(OUTPUT_DIR, 'dpg_constraints_tex')
    os.makedirs(individual_constraints_dir, exist_ok=True)
    
    print(f"  Exporting individual .tex files per dataset...")
    
    for info in sorted_data:
        dataset = info['Dataset']
        dataset_latex = dataset.replace('_', '\\_')
        safe_name = dataset.replace('/', '_').replace(' ', '_')
        
        # Get DPG constraints for this dataset
        constraints = get_dpg_constraints_data(dataset)
        
        if not constraints:
            continue
        
        # Write individual LaTeX file for this dataset
        individual_path = os.path.join(individual_constraints_dir, f"{safe_name}_dpg_constraints.tex")
        with open(individual_path, 'w') as f:
            f.write(f"% DPG Constraints for {dataset} dataset\n")
            f.write("% This file shows min/max bounds learned by DPG for each feature and class\n")
            f.write("% These constraints define valid regions in the feature space for each class\n")
            f.write("% and are used to guide counterfactual generation.\n\n")
            
            # Start landscape page for this dataset
            f.write("\\begin{landscape}\n")
            f.write("\\begin{table}[ht]\n")
            f.write("  \\centering\n")
            f.write(f"  \\caption{{DPG-learned constraints (min/max bounds) for {dataset_latex} dataset. ")
            f.write("These constraints define valid regions in the feature space for each class ")
            f.write("and are used to guide counterfactual generation.}\n")
            f.write(f"  \\label{{tab:dpg-constraints-{dataset}}}\n")
            f.write("  \\small\n")
            
            # Build table with classes as columns
            classes = sorted(constraints.keys())
            num_classes = len(classes)
            
            # Column specification: Feature name + (min, max) pair for each class
            col_spec = "l" + "rr" * num_classes
            f.write(f"  \\begin{{tabular}}{{{col_spec}}}\n")
            f.write("    \\toprule\n")
            
            # Header row 1: Feature and class labels
            header1 = "    Feature"
            for class_label in classes:
                class_latex = str(class_label).replace('_', '\\_')
                header1 += f" & \\multicolumn{{2}}{{c}}{{{class_latex}}}"
            header1 += " \\\\\n"
            f.write(header1)
            
            # Header row 2: Min/Max labels under each class
            header2 = "    "
            for _ in classes:
                header2 += " & Min & Max"
            header2 += " \\\\\n"
            f.write(header2)
            f.write("    \\midrule\n")
            
            # Collect all features across all classes
            all_features = set()
            for class_constraints in constraints.values():
                all_features.update(class_constraints.keys())
            
            # Write data rows (one per feature)
            for feature in sorted(all_features):
                feature_latex = feature.replace('_', '\\_')
                row = f"    {feature_latex}"
                
                for class_label in classes:
                    class_constraints = constraints[class_label]
                    bounds = class_constraints.get(feature, {})
                    
                    min_val = bounds.get('min')
                    max_val = bounds.get('max')
                    
                    # Format values nicely
                    min_str = f"{min_val:.2f}" if min_val is not None else "---"
                    max_str = f"{max_val:.2f}" if max_val is not None else "---"
                    
                    row += f" & {min_str} & {max_str}"
                
                row += " \\\\\n"
                f.write(row)
            
            f.write("    \\bottomrule\n")
            f.write("  \\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("\\end{landscape}\n")
        
        print(f"    ✓ {dataset}: {individual_path}")
    
    print(f"\n✓ Exported individual DPG constraints .tex files to: {individual_constraints_dir}")
    
    # Also export a summary statistics file
    summary_path = os.path.join(OUTPUT_DIR, 'rf_model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANDOM FOREST MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Datasets: {len(model_info_data)}\n\n")
        
        # Convert accuracy strings back to float for statistics
        train_accs = [float(info['Train Accuracy']) for info in model_info_data]
        test_accs = [float(info['Test Accuracy']) for info in model_info_data]
        
        f.write("Train Accuracy Statistics:\n")
        f.write(f"  Mean:   {np.mean(train_accs):.4f}\n")
        f.write(f"  Median: {np.median(train_accs):.4f}\n")
        f.write(f"  Std:    {np.std(train_accs):.4f}\n")
        f.write(f"  Min:    {np.min(train_accs):.4f}\n")
        f.write(f"  Max:    {np.max(train_accs):.4f}\n\n")
        
        f.write("Test Accuracy Statistics:\n")
        f.write(f"  Mean:   {np.mean(test_accs):.4f}\n")
        f.write(f"  Median: {np.median(test_accs):.4f}\n")
        f.write(f"  Std:    {np.std(test_accs):.4f}\n")
        f.write(f"  Min:    {np.min(test_accs):.4f}\n")
        f.write(f"  Max:    {np.max(test_accs):.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED MODEL INFORMATION\n")
        f.write("="*80 + "\n\n")
        
        for info in model_info_data:
            f.write(f"Dataset: {info['Dataset']}\n")
            f.write(f"  Train/Test Split: {info['Train Size']}/{info['Test Size']} "
                   f"({info['Test Size %']})\n")
            f.write(f"  Accuracies: Train={info['Train Accuracy']}, "
                   f"Test={info['Test Accuracy']}\n")
            f.write(f"  Features: {info['Num Features']}, Classes: {info['Num Classes']}\n")
            
            # Show RF params if present
            rf_params = {k: v for k, v in info.items() if k.startswith('RF_')}
            if rf_params:
                f.write(f"  RF Params: ")
                param_strs = [f"{k.replace('RF_', '')}={v}" for k, v in rf_params.items()]
                f.write(", ".join(param_strs) + "\n")
            f.write("\n")
    
    print(f"✓ Exported RF model summary to: {summary_path}")
    
    return model_info_df


def main():
    """Main execution flow."""
    global _WANDB_DATA_CACHE
    
    print("\n" + "="*80)
    print("DPG vs DiCE COMPARISON RESULTS EXPORT")
    print("="*80)
    print("This script replicates notebooks/technique_comparison.ipynb")
    print("and exports all results to files.")
    if args.local_only:
        print("\n🏠 LOCAL-ONLY MODE: Regenerating visualizations from disk")
    print()
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Load WandB data cache if in local-only mode
    if args.local_only:
        _WANDB_DATA_CACHE = load_wandb_data_cache()
        if _WANDB_DATA_CACHE:
            print(f"✓ Loaded WandB data cache with {len(_WANDB_DATA_CACHE)} datasets")
    
    if args.local_only:
        # Load data from disk
        metadata = load_metadata()
        if metadata is None:
            # If metadata doesn't exist but comparison.csv does, create minimal metadata
            comparison_df = load_comparison_from_disk()
            if comparison_df is None:
                return
            # Create minimal raw_df from comparison data
            raw_df = pd.DataFrame()
            # Cannot reconstruct full raw_df, but we have enough for visualizations
            print(f"\n✓ Loaded comparison.csv (metadata.pkl not available)")
        else:
            comparison_df = load_comparison_from_disk()
            if comparison_df is None:
                return
            
            raw_df = metadata['raw_df']
            print(f"\n✓ Loaded {len(raw_df)} runs from disk")
        
        # Apply --ids filter in local-only mode
        if args.ids:
            print(f"\n🔍 Applying --ids filter in local-only mode")
            run_ids_dict = load_run_ids_from_yaml(args.ids)
            if run_ids_dict is not None:
                datasets_to_include = list(run_ids_dict.keys())
                pre_filter_count = len(comparison_df)
                comparison_df = comparison_df[comparison_df['dataset'].isin(datasets_to_include)]
                if len(raw_df) > 0:
                    raw_df = raw_df[raw_df['dataset'].isin(datasets_to_include)]
                print(f"✓ Applied --ids filter: {pre_filter_count} -> {len(comparison_df)} datasets")
                print(f"  Processing datasets: {sorted(comparison_df['dataset'].unique())}")
        # Apply priority_datasets filter in local-only mode (only if --ids not specified)
        elif APPLY_EXCLUDED_DATASETS:
            included_datasets = load_included_datasets()
            if included_datasets is not None:
                pre_filter_count = len(comparison_df)
                comparison_df = comparison_df[comparison_df['dataset'].isin(included_datasets)]
                if len(raw_df) > 0:
                    raw_df = raw_df[raw_df['dataset'].isin(included_datasets)]
                print(f"✓ Applied priority_datasets filter: {pre_filter_count} -> {len(comparison_df)} datasets")
                print(f"  Processing datasets: {sorted(comparison_df['dataset'].unique())}")
    else:
        # Fetch and filter data from WandB
        raw_df = fetch_and_filter_data()
        
        if len(raw_df) == 0:
            print("\n❌ No data found. Exiting.")
            return
        
        print(f"\n✓ Successfully fetched {len(raw_df)} runs")
        print(f"  Datasets: {sorted(raw_df['dataset'].unique())}")
        print(f"  Techniques: {sorted(raw_df['technique'].unique())}")
        
        # Create comparison table (small=True)
        comparison_df = create_comparison_table_small(raw_df)
    
    # Export all results
    # Always export these to reflect any dataset filtering (even in local-only mode)
    export_comparison_table(comparison_df)
    export_method_metrics_tables(raw_df)
    export_summary_statistics(comparison_df)
    
    if not args.local_only:
        save_metadata(raw_df)
    
    # Always regenerate visualizations
    export_winner_heatmap(comparison_df)
    export_comparison_numeric_csv(comparison_df)
    export_winner_heatmap_small(comparison_df)
    export_comparison_numeric_csv(comparison_df, metrics_to_include=[
        'perc_valid_cf_all',
        'perc_actionable_cf_all',
        'plausibility_nbr_cf',
        'distance_mh',
        'avg_nbr_changes',
        'count_diversity_all',
        'accuracy_knn_sklearn',
        'runtime'
    ], filename_suffix='_small')


    # export_radar_charts(comparison_df)
    export_dataset_visualizations(comparison_df, raw_df)
    
    # Export Random Forest model information
    export_model_information(raw_df, comparison_df)
    
    # Export DPG constraints for all datasets
    export_dpg_constraints(comparison_df)
    
    # Run statistical analysis (Wilcoxon tests, LaTeX tables, etc.)
    print("\n" + "="*80)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*80)
    small_csv_path = os.path.join(OUTPUT_DIR, 'comparison_numeric_small.csv')
    if os.path.exists(small_csv_path):
        stats_output_dir = os.path.join(OUTPUT_DIR, 'statistics')
        run_statistical_analysis(
            csv_path=small_csv_path,
            output_dir=stats_output_dir,
            bad_datasets=None,  # Uses default exclusions from statistics module
        )
    else:
        print(f"⚠ Skipping statistical analysis: {small_csv_path} not found")
    
    # Always export comparison summary (even in local-only mode, to reflect filtered datasets)
    export_comparison_summary(comparison_df)
    
    if not args.local_only:
        # Save WandB data cache for future local-only use
        if _WANDB_DATA_CACHE:
            save_wandb_data_cache(_WANDB_DATA_CACHE)
    
    print("\n" + "="*80)
    print("✓ ALL RESULTS EXPORTED SUCCESSFULLY")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            print(f"  - {filename}")
    print()


if __name__ == '__main__':
    main()
