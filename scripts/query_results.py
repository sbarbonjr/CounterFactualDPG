"""Query and compare experiment results from Weights & Biases.

This utility script allows you to query, compare, and export experiment results
logged to WandB for analysis and reporting.

Usage:
  # List all runs in a project
  python scripts/query_results.py list --project CounterFactualDPG
  
  # Compare specific runs by ID
  python scripts/query_results.py compare --runs run1_id run2_id --metric sample/success_rate
  
  # Export all runs to CSV
  python scripts/query_results.py export --project CounterFactualDPG --output results.csv
  
  # Get best run by metric
  python scripts/query_results.py best --project CounterFactualDPG --metric experiment/overall_success_rate
"""

import argparse
import logging
import sys
from typing import List, Optional

import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

logger = logging.getLogger(__name__)


def list_runs(project: str, entity: str = 'mllab-ts-universit-di-trieste', tags: Optional[List[str]] = None, limit: int = 50):
    """List all runs in a WandB project.
    
    Args:
        project: WandB project name
        entity: WandB entity (username/team)
        tags: Optional list of tags to filter by
        limit: Maximum number of runs to return
    """
    api = wandb.Api()
    
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    
    runs = api.runs(f"{entity}/{project}", filters=filters)
    
    print(f"\n{'='*80}")
    print(f"Runs in project: {entity}/{project}")
    if tags:
        print(f"Filtered by tags: {tags}")
    print(f"{'='*80}\n")
    
    data = []
    for i, run in enumerate(runs):
        if i >= limit:
            break
        
        # Extract key metrics
        summary = run.summary._json_dict
        config = run.config
        
        data.append({
            'ID': run.id,
            'Name': run.name,
            'State': run.state,
            'Tags': ', '.join(run.tags),
            'Success Rate': summary.get('experiment/overall_success_rate', 'N/A'),
            'Valid CFs': summary.get('experiment/total_valid_counterfactuals', 'N/A'),
            'Created': run.created_at,
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"\nShowing {len(df)} of {len(list(runs))} total runs\n")
    
    return df


def compare_runs(project: str, run_ids: List[str], entity: str = 'mllab-ts-universit-di-trieste', metrics: Optional[List[str]] = None):
    """Compare specific runs by their metrics.
    
    Args:
        project: WandB project name
        run_ids: List of run IDs to compare
        entity: WandB entity (username/team)
        metrics: Optional list of specific metrics to compare. If None, shows all.
    """
    api = wandb.Api()
    
    print(f"\n{'='*80}")
    print(f"Comparing {len(run_ids)} runs")
    print(f"{'='*80}\n")
    
    comparison_data = []
    
    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            
            summary = run.summary._json_dict
            config = run.config
            
            row = {
                'Run ID': run.id,
                'Name': run.name,
                'State': run.state,
            }
            
            # Add config parameters
            if 'counterfactual' in config:
                cf_config = config['counterfactual']
                row['Population'] = cf_config.get('population_size', 'N/A')
                row['Generations'] = cf_config.get('max_generations', 'N/A')
                row['Diversity Wt'] = cf_config.get('diversity_weight', 'N/A')
                row['Repulsion Wt'] = cf_config.get('repulsion_weight', 'N/A')
            
            # Add metrics
            if metrics:
                for metric in metrics:
                    row[metric] = summary.get(metric, 'N/A')
            else:
                # Default metrics
                row['Success Rate'] = summary.get('experiment/overall_success_rate', 'N/A')
                row['Valid CFs'] = summary.get('experiment/total_valid_counterfactuals', 'N/A')
                row['Total Reps'] = summary.get('experiment/total_replications', 'N/A')
            
            comparison_data.append(row)
            
        except Exception as e:
            logger.error(f"Error fetching run {run_id}: {e}")
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()
    
    return df


def export_runs(project: str, output_file: str, entity: str = 'mllab-ts-universit-di-trieste', tags: Optional[List[str]] = None):
    """Export all runs to a CSV file.
    
    Args:
        project: WandB project name
        output_file: Path to output CSV file
        entity: WandB entity (username/team)
        tags: Optional list of tags to filter by
    """
    api = wandb.Api()
    
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    
    runs = api.runs(f"{entity}/{project}", filters=filters)
    
    data = []
    for run in runs:
        summary = run.summary._json_dict
        config = run.config
        
        row = {
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'tags': ', '.join(run.tags),
            'created_at': run.created_at,
        }
        
        # Add all config parameters with prefix
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    row[f'config_{key}_{subkey}'] = subvalue
            else:
                row[f'config_{key}'] = value
        
        # Add all summary metrics
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                row[f'metric_{key.replace("/", "_")}'] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"\nExported {len(df)} runs to {output_file}")
    print(f"Columns: {len(df.columns)}")
    print(f"Sample columns: {list(df.columns[:10])}\n")
    
    return df


def get_best_run(project: str, metric: str, entity: str = 'mllab-ts-universit-di-trieste', maximize: bool = True):
    """Get the best run according to a specific metric.
    
    Args:
        project: WandB project name
        metric: Metric to optimize for
        entity: WandB entity (username/team)
        maximize: If True, get run with highest metric; if False, get lowest
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    best_run = None
    best_value = float('-inf') if maximize else float('inf')
    
    for run in runs:
        summary = run.summary._json_dict
        value = summary.get(metric)
        
        if value is None:
            continue
        
        if maximize and value > best_value:
            best_value = value
            best_run = run
        elif not maximize and value < best_value:
            best_value = value
            best_run = run
    
    if best_run is None:
        print(f"\nNo runs found with metric: {metric}\n")
        return None
    
    print(f"\n{'='*80}")
    print(f"Best run for metric: {metric} ({'maximize' if maximize else 'minimize'})")
    print(f"{'='*80}\n")
    print(f"Run ID: {best_run.id}")
    print(f"Name: {best_run.name}")
    print(f"State: {best_run.state}")
    print(f"Tags: {', '.join(best_run.tags)}")
    print(f"{metric}: {best_value}")
    print(f"Created: {best_run.created_at}")
    print(f"\nConfig:")
    for key, value in best_run.config.items():
        print(f"  {key}: {value}")
    print()
    
    return best_run


def main():
    parser = argparse.ArgumentParser(
        description="Query and analyze WandB experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all runs in a project')
    list_parser.add_argument('--entity', type=str, default='mllab-ts-universit-di-trieste', help='WandB entity (username or team)')
    list_parser.add_argument('--project', type=str, default='CounterFactualDPG', help='WandB project name')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--limit', type=int, default=50, help='Maximum number of runs to show')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare specific runs')
    compare_parser.add_argument('--entity', type=str, default='mllab-ts-universit-di-trieste', help='WandB entity (username or team)')
    compare_parser.add_argument('--project', type=str, default='CounterFactualDPG', help='WandB project name')
    compare_parser.add_argument('--runs', nargs='+', required=True, help='Run IDs to compare')
    compare_parser.add_argument('--metrics', nargs='+', help='Specific metrics to compare')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export runs to CSV')
    export_parser.add_argument('--entity', type=str, default='mllab-ts-universit-di-trieste', help='WandB entity (username or team)')
    export_parser.add_argument('--project', type=str, default='CounterFactualDPG', help='WandB project name')
    export_parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    export_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    
    # Best command
    best_parser = subparsers.add_parser('best', help='Get best run by metric')
    best_parser.add_argument('--entity', type=str, default='mllab-ts-universit-di-trieste', help='WandB entity (username or team)')
    best_parser.add_argument('--project', type=str, default='CounterFactualDPG', help='WandB project name')
    best_parser.add_argument('--metric', type=str, required=True, help='Metric to optimize')
    best_parser.add_argument('--minimize', action='store_true', help='Minimize metric (default: maximize)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Execute command
    if args.command == 'list':
        list_runs(args.project, args.entity, args.tags, args.limit)
    
    elif args.command == 'compare':
        compare_runs(args.project, args.runs, args.entity, args.metrics)
    
    elif args.command == 'export':
        export_runs(args.project, args.output, args.entity, args.tags)
    
    elif args.command == 'best':
        get_best_run(args.project, args.metric, args.entity, maximize=not args.minimize)


if __name__ == '__main__':
    main()
