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

from CounterFactualVisualizer import heatmap_techniques, plot_pca_with_counterfactuals_comparison
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

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    '_comparison_results')

# Metadata file for storing run information
METADATA_FILE = os.path.join(OUTPUT_DIR, 'metadata.pkl')

# Repository root for loading datasets
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache for loaded datasets and models
_DATASET_CACHE = {}


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
    """
    # Check cache first
    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]
    
    try:
        # Load dataset config
        config_path = os.path.join(REPO_ROOT, 'configs', dataset_name, 'config.yaml')
        if not os.path.exists(config_path):
            print(f"  ⚠ Config not found for {dataset_name}: {config_path}")
            return None
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to DictConfig object for easier access
        config = DictConfig(config_dict)
        
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
        
        result = {
            'model': model,
            'dataset': features_df,
            'target': labels,
            'feature_names': feature_names,
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
    
    # Handle local-only mode by loading from disk
    if args.local_only:
        try:
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
                print(f"  ⚠ {dataset}: Missing local data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs)}, dice_cfs: {len(dice_cfs)}")
                return False
            
            # Create heatmap
            fig = heatmap_techniques(
                sample=dpg_sample,
                class_sample=dpg_class,
                cf_list_1=dpg_cfs[:5],
                cf_list_2=dice_cfs[:5],
                technique_names=('DPG', 'DiCE'),
                restrictions=None
            )
            
            if fig:
                output_path = os.path.join(dataset_viz_dir, 'heatmap_techniques.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {dataset}: Exported heatmap_techniques (from local data)")
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
    
    # Handle local-only mode by loading from disk
    if args.local_only:
        try:
            dpg_data = _load_local_viz_data(dataset, 'dpg')
            dice_data = _load_local_viz_data(dataset, 'dice')
            
            if not dpg_data or not dice_data:
                missing = []
                if not dpg_data:
                    missing.append('DPG')
                if not dice_data:
                    missing.append('DiCE')
                print(f"  ⚠ {dataset}: Missing local {' and '.join(missing)} data, skipping PCA comparison")
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
            
            if not dpg_sample or not dpg_cfs or not dice_cfs:
                print(f"  ⚠ {dataset}: Missing local data - sample: {bool(dpg_sample)}, dpg_cfs: {len(dpg_cfs)}, dice_cfs: {len(dice_cfs)}")
                return False
            
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
                method_2_name='DiCE',
                method_1_color='#1f77b4',  # Blue for DPG
                method_2_color='#ff7f0e'   # Orange for DiCE
            )
            
            if fig:
                output_path = os.path.join(dataset_viz_dir, 'pca_comparison.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {dataset}: Exported PCA comparison (from local data)")
                return True
            
            return False
            
        except Exception as e:
            print(f"  ⚠ {dataset}: Error loading local data for PCA comparison: {e}")
            return False
    
    # WandB mode
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
            method_2_name='DiCE',
            method_1_color='#1f77b4',  # Blue for DPG
            method_2_color='#ff7f0e'   # Orange for DiCE
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
        
        # Export PCA comparison (loads dataset and model automatically)
        export_pca_comparison(raw_df, dataset, dataset_viz_dir)
        
        # Fetch WandB visualizations (comparison, pca_clean, heatmap)
        wandb_viz = fetch_wandb_visualizations(raw_df, dataset, dataset_viz_dir)
        if wandb_viz:
            viz_files.extend(wandb_viz)
        
        # Export radar chart for this dataset
        from scripts.compare_techniques import plot_radar_chart
        radar_path = os.path.join(dataset_viz_dir, f'radar.png')
        fig = plot_radar_chart(comparison_df, dataset, figsize=(8, 8))
        if fig:
            fig.savefig(radar_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            viz_files.append('radar.png')
        
        # Export bar charts for each metric (small metrics only)
        small_metrics = {
            'perc_valid_cf_all',
            'perc_actionable_cf_all',
            'plausibility_nbr_cf',
            'distance_mh',
            'avg_nbr_changes',
            'count_diversity_all',
            'accuracy_knn_sklearn',
            'runtime'
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


def main():
    """Main execution flow."""
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
        
        # Apply priority_datasets filter in local-only mode
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
    if not args.local_only:
        export_comparison_table(comparison_df)
        export_method_metrics_tables(raw_df)
        export_summary_statistics(comparison_df)
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
    export_radar_charts(comparison_df)
    export_dataset_visualizations(comparison_df, raw_df)
    
    if not args.local_only:
        export_comparison_summary(comparison_df)
    else:
        # Load and print summary from file
        summary_path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
        if os.path.exists(summary_path):
            print("\n" + "="*80)
            print("CONSOLE SUMMARY (from saved file)")
            print("="*80)
            with open(summary_path, 'r') as f:
                print(f.read())
    
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
