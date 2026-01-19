#!/usr/bin/env python3
"""Compare DPG vs DiCE experiment results from Weights & Biases.

This script fetches all runs from WandB, aggregates metrics by dataset and technique,
and generates visualizations for easy comparison.

Usage:
    # Generate comparison report (default: prints summary table)
    python scripts/compare_techniques.py --project counterfactual-dpg
    
    # Export aggregated results to CSV
    python scripts/compare_techniques.py --project counterfactual-dpg --output comparison.csv
    
    # Generate HTML visualization report
    python scripts/compare_techniques.py --project counterfactual-dpg --html report.html
    
    # Generate all visualizations as images
    python scripts/compare_techniques.py --project counterfactual-dpg --plots outputs/comparison_plots/
    
    # Compare specific datasets only
    python scripts/compare_techniques.py --project counterfactual-dpg --datasets iris german_credit heart_disease_uci
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Visualizations will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. HTML reports will be disabled.")

logger = logging.getLogger(__name__)

# Default WandB settings
DEFAULT_ENTITY = 'mllab-ts-universit-di-trieste'
DEFAULT_PROJECT = 'CounterFactualDPG'

# Key metrics for comparison (from RECOMMENDED_METRICS)
# Note: Only using 'metrics/combination/' prefix for aggregated metrics across all CFs
# The 'name' field now uses the original wandb key names for consistency in tables and visualizations
COMPARISON_METRICS = {
    'perc_valid_cf': {
        'name': 'perc_valid_cf',
        'goal': 'maximize',
        'description': 'Percentage of valid counterfactuals',
        'format': '.2%',
        'wandb_keys': ['metrics/combination/perc_valid_cf'],
    },
    'plausibility_sum': {
        'name': 'plausibility_sum',
        'goal': 'minimize',
        'description': 'Sum of distances to nearest training samples',
        'format': '.3f',
        'wandb_keys': ['metrics/combination/plausibility_sum', 'metrics/combination/plausibility_nbr_cf'],
    },
    'distance_l2': {
        'name': 'distance_l2',
        'goal': 'minimize',
        'description': 'Mean Euclidean distance from original',
        'format': '.3f',
        'wandb_keys': ['metrics/combination/distance_l2'],
    },
    'distance_mad': {
        'name': 'distance_mad',
        'goal': 'minimize',
        'description': 'Mean MAD-normalized distance',
        'format': '.3f',
        'wandb_keys': ['metrics/combination/distance_mad'],
    },
    'avg_nbr_changes_per_cf': {
        'name': 'avg_nbr_changes_per_cf',
        'goal': 'minimize',
        'description': 'Average number of feature changes',
        'format': '.2f',
        'wandb_keys': ['metrics/combination/avg_nbr_changes_per_cf'],
    },
    'diversity_l2': {
        'name': 'diversity_l2',
        'goal': 'maximize',
        'description': 'Mean pairwise diversity (L2)',
        'format': '.3f',
        'wandb_keys': ['metrics/combination/diversity_l2'],
    },
    'perc_valid_actionable_cf': {
        'name': 'perc_valid_actionable_cf',
        'goal': 'maximize',
        'description': 'Percentage of valid and actionable CFs',
        'format': '.2%',
        'wandb_keys': ['metrics/combination/perc_valid_actionable_cf'],
    },
    'accuracy_knn_sklearn': {
        'name': 'accuracy_knn_sklearn',
        'goal': 'maximize',
        'description': 'KNN classifier accuracy on CFs',
        'format': '.2%',
        'wandb_keys': ['metrics/combination/accuracy_knn_sklearn', 'metrics/combination/accuracy_knn_dist'],
    },
    'delta': {
        'name': 'delta',
        'goal': 'maximize',
        'description': 'Mean prediction probability change',
        'format': '.3f',
        'wandb_keys': ['metrics/combination/delta'],
    },
}

# Technique colors for consistent visualization
TECHNIQUE_COLORS = {
    'dpg': '#2ecc71',   # Green
    'dice': '#3498db',  # Blue
}


def fetch_all_runs(
    project: str = DEFAULT_PROJECT,
    entity: str = DEFAULT_ENTITY,
    techniques: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch runs from WandB and organize by dataset/technique.
    
    Args:
        project: WandB project name
        entity: WandB entity (team/user)
        techniques: Optional list of techniques to include (default: ['dpg', 'dice'])
        datasets: Optional list of datasets to include (default: all)
        limit: Maximum number of recent runs to fetch (default: 10)
        
    Returns:
        DataFrame with columns: dataset, technique, replication, and all metrics
        
    Note:
        Only runs with state='finished' and properly configured data.dataset and data.method
        will be included. Runs missing these fields are skipped.
    """
    techniques = techniques or ['dpg', 'dice']
    api = wandb.Api()
    api.flush()  # Clear any cached data
    
    print(f"Fetching up to {limit} runs from {entity}/{project}...")
    
    # Order by created_at descending to get most recent first
    runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=limit)
    
    data = []
    for i, run in enumerate(runs):
        if i >= limit:
            break
        
        if run.state != 'finished':
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        # Get dataset from config
        if 'data' not in config:
            continue
        
        data_config = config['data']
        dataset_name = data_config.get('dataset_name') or data_config.get('dataset')
        
        if not dataset_name:
            continue
        
        # Get method from config, or infer from run name (e.g., "iris_dice" -> "dice")
        technique = data_config.get('method')
        if not technique:
            # Try to infer from run name
            run_name_lower = run.name.lower()
            if '_dpg' in run_name_lower or run_name_lower.endswith('dpg'):
                technique = 'dpg'
            elif '_dice' in run_name_lower or run_name_lower.endswith('dice'):
                technique = 'dice'
            else:
                continue
        
        technique = technique.lower()
        
        # Filter by requested techniques and datasets
        if technique not in techniques:
            continue
        if datasets and dataset_name not in datasets:
            continue
        
        # Extract metrics
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'dataset': dataset_name,
            'technique': technique,
            'state': run.state,
        }
        
        # Add replication info if available
        if 'experiment_params' in config and 'replications' in config['experiment_params']:
            row['replications'] = config['experiment_params']['replications']
        
        # Extract per-counterfactual metrics using wandb_keys
        for metric_key, metric_info in COMPARISON_METRICS.items():
            value = None
            # Try each possible WandB key for this metric
            wandb_keys = metric_info.get('wandb_keys', [f'metrics/{metric_key}', f'combo_metrics/{metric_key}'])
            for wkey in wandb_keys:
                if wkey in summary:
                    value = summary[wkey]
                    break
            
            row[metric_key] = value
        
        # Also extract any additional summary metrics
        for key, value in summary.items():
            if isinstance(value, (int, float)) and key not in row:
                # Clean up metric names
                clean_key = key.replace('combo_metrics/', '').replace('metrics/', '')
                if clean_key not in row:
                    row[clean_key] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No runs found matching criteria!")
        return df
    
    print(f"Fetched {len(df)} runs across {df['dataset'].nunique()} datasets")
    print(f"Techniques: {df['technique'].unique().tolist()}")
    
    return df


def aggregate_by_dataset_technique(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by dataset and technique.
    
    Computes mean and std for each metric across all runs of the same
    dataset-technique combination.
    
    Args:
        df: DataFrame from fetch_all_runs()
        
    Returns:
        DataFrame with aggregated statistics
    """
    if len(df) == 0:
        return df
    
    # Get numeric metric columns
    metric_cols = [col for col in COMPARISON_METRICS.keys() if col in df.columns]
    
    # Group by dataset and technique
    agg_funcs = {col: ['mean', 'std', 'count'] for col in metric_cols}
    
    grouped = df.groupby(['dataset', 'technique']).agg(agg_funcs)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    return grouped


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a side-by-side comparison table for DPG vs DiCE.
    
    Args:
        df: DataFrame from fetch_all_runs()
        
    Returns:
        Pivot table with datasets as rows, metrics as columns, techniques compared
    """
    if len(df) == 0:
        return df
    
    # Aggregate first
    agg_df = aggregate_by_dataset_technique(df)
    
    # Get available metrics
    metric_cols = [col for col in COMPARISON_METRICS.keys() if f"{col}_mean" in agg_df.columns]
    
    # Create comparison data
    datasets = agg_df['dataset'].unique()
    techniques = agg_df['technique'].unique()
    
    comparison_data = []
    for dataset in sorted(datasets):
        row = {'dataset': dataset}
        
        for metric in metric_cols:
            for technique in techniques:
                mask = (agg_df['dataset'] == dataset) & (agg_df['technique'] == technique)
                subset = agg_df[mask]
                
                if len(subset) > 0:
                    mean_val = subset[f'{metric}_mean'].values[0]
                    std_val = subset[f'{metric}_std'].values[0]
                    count = subset[f'{metric}_count'].values[0]
                    
                    if pd.notna(mean_val):
                        row[f'{metric}_{technique}'] = mean_val
                        row[f'{metric}_{technique}_std'] = std_val if pd.notna(std_val) else 0
                        row[f'{metric}_{technique}_n'] = count
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def create_method_metrics_table(df: pd.DataFrame, dataset: Optional[str] = None, styled: bool = True) -> pd.DataFrame:
    """Create a table with methods as rows and metrics as columns.
    
    Args:
        df: DataFrame from fetch_all_runs()
        dataset: Optional dataset name to filter. If None, aggregates across all datasets.
        styled: If True, returns a styled DataFrame with color highlighting for best values.
        
    Returns:
        DataFrame (or Styler) with methods as rows (dpg, dice) and metrics as columns.
        Best values are highlighted in green when styled=True.
        
    Example output:
        |          | perc_valid_cf | plausibility_sum | distance_l2 | ...
        |----------|---------------|------------------|-------------|----
        | dpg      | 0.95          | 23.5             | 2.1         | ...
        | dice     | 0.98          | 18.2             | 1.8         | ...
    """
    if len(df) == 0:
        return df
    
    # Filter by dataset if specified
    if dataset is not None:
        df = df[df['dataset'] == dataset]
        if len(df) == 0:
            print(f"No data found for dataset: {dataset}")
            return pd.DataFrame()
    
    # Get metric columns
    metric_cols = [col for col in COMPARISON_METRICS.keys() if col in df.columns]
    
    # Aggregate by technique only (across all datasets if dataset is None)
    agg_data = df.groupby('technique')[metric_cols].mean()
    
    # Reset index to make technique a column, then set it back as index for display
    result = agg_data.reset_index()
    result = result.set_index('technique')
    
    # Build rename map and track goals for styling
    rename_map = {}
    col_goals = {}  # Map renamed column -> goal
    for col in metric_cols:
        if col in COMPARISON_METRICS:
            info = COMPARISON_METRICS[col]
            arrow = '↑' if info['goal'] == 'maximize' else '↓'
            new_name = f"{info['name']} {arrow}"
            rename_map[col] = new_name
            col_goals[new_name] = info['goal']
    
    result = result.rename(columns=rename_map)
    
    if not styled:
        return result
    
    # Apply styling: highlight best value per column based on goal
    def highlight_best(s):
        """Highlight the best value in a column based on its optimization goal."""
        col_name = s.name
        goal = col_goals.get(col_name, 'maximize')
        
        if goal == 'maximize':
            is_best = s == s.max()
        else:  # minimize
            is_best = s == s.min()
        
        return ['font-weight: bold' if v else '' for v in is_best]
    
    return result.style.apply(highlight_best, axis=0).format('{:.4f}')


def determine_winner(dpg_val: float, dice_val: float, goal: str) -> str:
    """Determine which technique is better for a metric.
    
    Args:
        dpg_val: DPG metric value
        dice_val: DiCE metric value
        goal: 'maximize' or 'minimize'
        
    Returns:
        'dpg', 'dice', or 'tie'
    """
    if pd.isna(dpg_val) or pd.isna(dice_val):
        return 'na'
    
    if goal == 'maximize':
        if dpg_val > dice_val:
            return 'dpg'
        elif dice_val > dpg_val:
            return 'dice'
    else:  # minimize
        if dpg_val < dice_val:
            return 'dpg'
        elif dice_val < dpg_val:
            return 'dice'
    
    return 'tie'


def print_comparison_summary(comparison_df: pd.DataFrame):
    """Print a formatted comparison summary to console.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
    """
    if len(comparison_df) == 0:
        print("No data to display")
        return
    
    print("\n" + "=" * 100)
    print("DPG vs DiCE COMPARISON SUMMARY")
    print("=" * 100)
    
    # Count wins per technique per metric
    wins = {'dpg': {}, 'dice': {}}
    
    for metric_key, metric_info in COMPARISON_METRICS.items():
        dpg_col = f'{metric_key}_dpg'
        dice_col = f'{metric_key}_dice'
        
        if dpg_col not in comparison_df.columns or dice_col not in comparison_df.columns:
            continue
        
        print(f"\n{metric_info['name']} ({metric_info['description']})")
        print(f"  Goal: {metric_info['goal']}")
        print("-" * 80)
        print(f"  {'Dataset':<30} {'DPG':>15} {'DiCE':>15} {'Winner':>10}")
        print("-" * 80)
        
        dpg_wins = 0
        dice_wins = 0
        
        for _, row in comparison_df.iterrows():
            dataset = row['dataset']
            dpg_val = row.get(dpg_col, np.nan)
            dice_val = row.get(dice_col, np.nan)
            
            winner = determine_winner(dpg_val, dice_val, metric_info['goal'])
            
            if winner == 'dpg':
                dpg_wins += 1
                winner_str = '← DPG'
            elif winner == 'dice':
                dice_wins += 1
                winner_str = 'DiCE →'
            else:
                winner_str = 'tie' if winner == 'tie' else 'N/A'
            
            dpg_str = f"{dpg_val:.4f}" if pd.notna(dpg_val) else "N/A"
            dice_str = f"{dice_val:.4f}" if pd.notna(dice_val) else "N/A"
            
            print(f"  {dataset:<30} {dpg_str:>15} {dice_str:>15} {winner_str:>10}")
        
        wins['dpg'][metric_key] = dpg_wins
        wins['dice'][metric_key] = dice_wins
        
        print("-" * 80)
        print(f"  {'TOTAL WINS':<30} {dpg_wins:>15} {dice_wins:>15}")
    
    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL WIN COUNTS BY METRIC")
    print("=" * 100)
    print(f"\n  {'Metric':<25} {'DPG Wins':>12} {'DiCE Wins':>12} {'Better':<10}")
    print("-" * 60)
    
    total_dpg = 0
    total_dice = 0
    
    for metric_key, metric_info in COMPARISON_METRICS.items():
        dpg_w = wins['dpg'].get(metric_key, 0)
        dice_w = wins['dice'].get(metric_key, 0)
        total_dpg += dpg_w
        total_dice += dice_w
        
        if dpg_w > dice_w:
            better = 'DPG'
        elif dice_w > dpg_w:
            better = 'DiCE'
        else:
            better = 'Tie'
        
        print(f"  {metric_info['name']:<25} {dpg_w:>12} {dice_w:>12} {better:<10}")
    
    print("-" * 60)
    print(f"  {'TOTAL':<25} {total_dpg:>12} {total_dice:>12}")
    print("\n")


def plot_grouped_bar_chart(
    comparison_df: pd.DataFrame,
    metric_key: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[plt.Figure]:
    """Create grouped bar chart comparing DPG vs DiCE for a specific metric.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
        metric_key: Metric to visualize
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    metric_info = COMPARISON_METRICS.get(metric_key, {'name': metric_key, 'goal': 'maximize'})
    
    dpg_col = f'{metric_key}_dpg'
    dice_col = f'{metric_key}_dice'
    
    if dpg_col not in comparison_df.columns or dice_col not in comparison_df.columns:
        print(f"Metric {metric_key} not found in data")
        return None
    
    # Sort by dataset name
    df = comparison_df.sort_values('dataset')
    
    datasets = df['dataset'].values
    dpg_values = df[dpg_col].values
    dice_values = df[dice_col].values
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, dpg_values, width, label='DPG', color=TECHNIQUE_COLORS['dpg'], alpha=0.8)
    bars2 = ax.bar(x + width/2, dice_values, width, label='DiCE', color=TECHNIQUE_COLORS['dice'], alpha=0.8)
    
    # Add error bars if std is available
    dpg_std_col = f'{metric_key}_dpg_std'
    dice_std_col = f'{metric_key}_dice_std'
    
    if dpg_std_col in df.columns:
        ax.errorbar(x - width/2, dpg_values, yerr=df[dpg_std_col].values, 
                    fmt='none', color='black', capsize=3, alpha=0.5)
    if dice_std_col in df.columns:
        ax.errorbar(x + width/2, dice_values, yerr=df[dice_std_col].values,
                    fmt='none', color='black', capsize=3, alpha=0.5)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(metric_info['name'], fontsize=12)
    ax.set_title(f"{metric_info['name']}: DPG vs DiCE\n({metric_info.get('description', '')})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add goal indicator
    goal_text = f"Goal: {metric_info['goal']}"
    ax.annotate(goal_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_radar_chart(
    comparison_df: pd.DataFrame,
    dataset: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> Optional[plt.Figure]:
    """Create radar/spider chart comparing DPG vs DiCE for a single dataset.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
        dataset: Dataset name to visualize
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    # Filter for specific dataset
    row = comparison_df[comparison_df['dataset'] == dataset]
    if len(row) == 0:
        print(f"Dataset {dataset} not found")
        return None
    row = row.iloc[0]
    
    # Get available metrics
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
                metrics.append(metric_info['name'])
                
                # Normalize values for radar chart (0-1 scale based on max)
                max_val = max(abs(dpg_val), abs(dice_val))
                if max_val > 0:
                    # For minimize goals, invert so higher is better
                    if metric_info['goal'] == 'minimize':
                        dpg_values.append(1 - (dpg_val / (max_val * 1.1)))
                        dice_values.append(1 - (dice_val / (max_val * 1.1)))
                    else:
                        dpg_values.append(dpg_val / (max_val * 1.1))
                        dice_values.append(dice_val / (max_val * 1.1))
                else:
                    dpg_values.append(0.5)
                    dice_values.append(0.5)
    
    if len(metrics) < 3:
        print(f"Not enough metrics for radar chart (need at least 3, got {len(metrics)})")
        return None
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    
    # Complete the loop
    dpg_values += dpg_values[:1]
    dice_values += dice_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    ax.plot(angles, dpg_values, 'o-', linewidth=2, label='DPG', color=TECHNIQUE_COLORS['dpg'])
    ax.fill(angles, dpg_values, alpha=0.25, color=TECHNIQUE_COLORS['dpg'])
    
    ax.plot(angles, dice_values, 'o-', linewidth=2, label='DiCE', color=TECHNIQUE_COLORS['dice'])
    ax.fill(angles, dice_values, alpha=0.25, color=TECHNIQUE_COLORS['dice'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)
    ax.set_ylim(0, 1)
    
    ax.set_title(f"DPG vs DiCE Profile: {dataset}\n(Higher = Better after normalization)", size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_heatmap_winners(
    comparison_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    """Create heatmap showing which technique wins for each dataset-metric pair.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    datasets = sorted(comparison_df['dataset'].unique())
    metrics = list(COMPARISON_METRICS.keys())
    
    # Create winner matrix: 1 = DPG wins, -1 = DiCE wins, 0 = tie/na
    winner_matrix = np.zeros((len(datasets), len(metrics)))
    
    for i, dataset in enumerate(datasets):
        row = comparison_df[comparison_df['dataset'] == dataset]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        for j, metric_key in enumerate(metrics):
            dpg_col = f'{metric_key}_dpg'
            dice_col = f'{metric_key}_dice'
            
            if dpg_col in row.index and dice_col in row.index:
                dpg_val = row[dpg_col]
                dice_val = row[dice_col]
                goal = COMPARISON_METRICS[metric_key]['goal']
                
                winner = determine_winner(dpg_val, dice_val, goal)
                if winner == 'dpg':
                    winner_matrix[i, j] = 1
                elif winner == 'dice':
                    winner_matrix[i, j] = -1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: green for DPG, blue for DiCE, white for tie
    from matplotlib.colors import LinearSegmentedColormap
    colors = [TECHNIQUE_COLORS['dice'], 'white', TECHNIQUE_COLORS['dpg']]
    cmap = LinearSegmentedColormap.from_list('technique_cmap', colors)
    
    im = ax.imshow(winner_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    metric_labels = [COMPARISON_METRICS[m]['name'] for m in metrics]
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(datasets)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(metrics)):
            val = winner_matrix[i, j]
            text = 'DPG' if val > 0 else ('DiCE' if val < 0 else '-')
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    ax.set_title('Winner Heatmap: DPG vs DiCE\n(Green = DPG wins, Blue = DiCE wins)', fontsize=14)
    
    # Legend
    dpg_patch = mpatches.Patch(color=TECHNIQUE_COLORS['dpg'], label='DPG wins')
    dice_patch = mpatches.Patch(color=TECHNIQUE_COLORS['dice'], label='DiCE wins')
    tie_patch = mpatches.Patch(color='white', ec='black', label='Tie / N/A')
    ax.legend(handles=[dpg_patch, dice_patch, tie_patch], loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_all_metrics_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str,
):
    """Generate all comparison plots and save to directory.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmap of winners
    plot_heatmap_winners(
        comparison_df,
        output_path=os.path.join(output_dir, 'winner_heatmap.png')
    )
    
    # 2. Bar charts for each metric
    for metric_key in COMPARISON_METRICS:
        plot_grouped_bar_chart(
            comparison_df,
            metric_key,
            output_path=os.path.join(output_dir, f'bar_{metric_key}.png')
        )
    
    # 3. Radar charts for each dataset
    for dataset in comparison_df['dataset'].unique():
        safe_name = dataset.replace('/', '_').replace(' ', '_')
        plot_radar_chart(
            comparison_df,
            dataset,
            output_path=os.path.join(output_dir, f'radar_{safe_name}.png')
        )
    
    print(f"\nAll plots saved to: {output_dir}")


def generate_html_report(
    comparison_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_path: str,
):
    """Generate interactive HTML report using Plotly.
    
    Args:
        comparison_df: DataFrame from create_comparison_table()
        raw_df: DataFrame from fetch_all_runs()
        output_path: Path to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("plotly not available. Install with: pip install plotly")
        return
    
    datasets = sorted(comparison_df['dataset'].unique())
    metrics = list(COMPARISON_METRICS.keys())
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[COMPARISON_METRICS[m]['name'] for m in metrics],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    
    for idx, metric_key in enumerate(metrics):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        dpg_col = f'{metric_key}_dpg'
        dice_col = f'{metric_key}_dice'
        
        if dpg_col not in comparison_df.columns or dice_col not in comparison_df.columns:
            continue
        
        df = comparison_df.sort_values('dataset')
        
        # DPG bars
        fig.add_trace(
            go.Bar(
                name='DPG',
                x=df['dataset'],
                y=df[dpg_col],
                marker_color=TECHNIQUE_COLORS['dpg'],
                showlegend=(idx == 0),
                legendgroup='dpg',
            ),
            row=row, col=col
        )
        
        # DiCE bars
        fig.add_trace(
            go.Bar(
                name='DiCE',
                x=df['dataset'],
                y=df[dice_col],
                marker_color=TECHNIQUE_COLORS['dice'],
                showlegend=(idx == 0),
                legendgroup='dice',
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text='DPG vs DiCE Comparison Across All Datasets',
        barmode='group',
        height=300 * n_rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axes to rotate labels
    fig.update_xaxes(tickangle=45)
    
    # Save to HTML
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DPG vs DiCE experiment results from WandB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--entity', type=str, default=DEFAULT_ENTITY,
                        help='WandB entity (username or team)')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help='WandB project name')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to compare (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for aggregated results')
    parser.add_argument('--html', type=str, default=None,
                        help='Output HTML file for interactive report')
    parser.add_argument('--plots', type=str, default=None,
                        help='Output directory for plot images')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console summary output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)
    
    # Fetch all runs
    raw_df = fetch_all_runs(
        project=args.project,
        entity=args.entity,
        datasets=args.datasets,
    )
    
    if len(raw_df) == 0:
        print("No data to analyze. Check your WandB project settings.")
        return
    
    # Create comparison table
    comparison_df = create_comparison_table(raw_df)
    
    # Print summary unless quiet
    if not args.quiet:
        print_comparison_summary(comparison_df)
    
    # Export to CSV if requested
    if args.output:
        comparison_df.to_csv(args.output, index=False)
        print(f"Comparison data saved to: {args.output}")
    
    # Generate HTML report if requested
    if args.html:
        generate_html_report(comparison_df, raw_df, args.html)
    
    # Generate plots if requested
    if args.plots:
        plot_all_metrics_comparison(comparison_df, args.plots)


if __name__ == '__main__':
    main()
