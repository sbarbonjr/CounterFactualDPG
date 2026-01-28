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
- radar_charts.png: Radar charts for datasets
- comparison_summary.txt: Console summary output

Usage:
    python scripts/export_comparison_results.py
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import tempfile
import wandb
from PIL import Image
import shutil
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from CounterFactualVisualizer import heatmap_techniques

# Configuration (matches notebook hardcoded values)
WANDB_ENTITY = 'mllab-ts-universit-di-trieste'
WANDB_PROJECT = 'CounterFactualDPG'
SELECTED_DATASETS = None  # No filter - fetch all
MIN_CREATED_AT = "2026-01-26T22:00:00"
APPLY_EXCLUDED_DATASETS = True

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    '_comparison_results'
)


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


def fetch_wandb_visualizations(raw_df, dataset, dataset_viz_dir):
    """Fetch comparison, pca_clean, and heatmap visualizations from WandB."""
    viz_types_exported = []
    try:
        dataset_runs = raw_df[raw_df['dataset'] == dataset]
        
        dpg_run = dataset_runs[dataset_runs['technique'] == 'dpg'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dpg']) > 0 else None
        dice_run = dataset_runs[dataset_runs['technique'] == 'dice'].iloc[0] if len(dataset_runs[dataset_runs['technique'] == 'dice']) > 0 else None
        
        if dpg_run is None or dice_run is None:
            print(f"  ⚠ {dataset}: Missing DPG or DiCE run, skipping WandB visualizations")
            return
        
        api = wandb.Api()
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
                    match = re.match(r'([a-z_]+)_\d+_', basename)
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
        limit=500,
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
    """Export winner heatmap to PNG."""
    print("\n" + "="*80)
    print("EXPORTING WINNER HEATMAP")
    print("="*80)
    
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    output_path = os.path.join(viz_dir, 'winner_heatmap.png')
    fig = plot_heatmap_winners(comparison_df, figsize=(16, 12))
    
    if fig:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Exported winner heatmap to: {output_path}")
        plt.close(fig)
    else:
        print("⚠ Could not create winner heatmap")


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
            'accuracy_knn_sklearn'
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
        
        # Fetch WandB visualizations (comparison, pca_clean, heatmap)
        fetch_wandb_visualizations(comparison_df, dataset, dataset_viz_dir)
        
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
            'accuracy_knn_sklearn'
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
    print()
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Fetch and filter data
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
    export_comparison_table(comparison_df)
    export_method_metrics_tables(raw_df)
    export_summary_statistics(comparison_df)
    export_winner_heatmap(comparison_df)
    export_radar_charts(comparison_df)
    export_dataset_visualizations(comparison_df, raw_df)
    export_comparison_summary(comparison_df)
    
    # Print summary to console as well
    print("\n" + "="*80)
    print("CONSOLE SUMMARY")
    print("="*80)
    print_comparison_summary(comparison_df)
    
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
