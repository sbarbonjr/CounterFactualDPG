#!/usr/bin/env python3
"""WandB Sweep Runner for DPG Counterfactual Generation.

This script runs a single experiment as part of a WandB hyperparameter sweep,
optimizing the genetic algorithm parameters for counterfactual quality.

Usage:
  # Run as part of a sweep (called by wandb agent)
  python scripts/run_sweep.py --dataset iris --method dpg
  
  # Run with specific target metric
  python scripts/run_sweep.py --dataset iris --target-metric perc_actionable_cf_all
  python scripts/run_sweep.py --dataset iris --target-metric perc_valid_cf_all
  
  # Initialize a new sweep
  python scripts/run_sweep.py --init-sweep --dataset iris --target-metric perc_valid_cf_all
  
  # Run sweep agent (after initialization)
  python scripts/run_sweep.py --run-agent --sweep-id <sweep_id> --count 10

Available Target Metrics (9 from Guidotti's paper):
  Metric Name               | Goal     | Description
  --------------------------|----------|---------------------------------------------
  plausibility_nbr_cf       | minimize | Implausibility - distance to nearest real sample (DEFAULT)
  perc_actionable_cf_all    | maximize | Actionability - respects constraints
  perc_valid_cf_all         | maximize | Size - percentage of valid CFs
  distance_mh               | minimize | Dissimilarity_dist - MAD+Hamming distance
  avg_nbr_changes           | minimize | Dissimilarity_count - proportion of features changed
  diversity_mh              | maximize | Diversity_dist - pairwise CF diversity
  count_diversity_all       | maximize | Diversity_count - feature diversity
  accuracy_knn_sklearn      | maximize | Discriminative Power (dipo) - 1NN accuracy
  runtime                   | minimize | Execution time in seconds

Note: Metrics are logged under 'metrics/combination/' prefix in WandB.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("ERROR: wandb is required for sweeps. Install with: pip install wandb")
    sys.exit(1)

from utils.config_manager import load_config, apply_overrides, deep_merge_dicts, DictConfig

# Define target metrics with their optimization goals (from Guidotti's paper)
# All metrics are logged under 'metrics/combination/' prefix in WandB
TARGET_METRICS = {
    # Size (Validity) - perc_valid_cf_all = |C|/k
    'perc_valid_cf_all': {'goal': 'maximize', 'description': 'Size - percentage of valid CFs', 'wandb_key': 'metrics/combination/perc_valid_cf_all'},
    
    # Actionability - perc_actionable_cf_all = |{c ∈ C | aA(c,x)}|/k
    'perc_actionable_cf_all': {'goal': 'maximize', 'description': 'Actionability - respects constraints', 'wandb_key': 'metrics/combination/perc_actionable_cf_all'},
    
    # Implausibility - plausibility_nbr_cf = (1/|C|)Σ min d(c, x) (DEFAULT)
    'plausibility_nbr_cf': {'goal': 'minimize', 'description': 'Implausibility - distance to nearest real sample', 'wandb_key': 'metrics/combination/plausibility_nbr_cf'},
    
    # Dissimilarity_dist - distance_mh = (1/|C|)Σ d(x, c) using MAD+Hamming
    'distance_mh': {'goal': 'minimize', 'description': 'Dissimilarity_dist - MAD+Hamming distance', 'wandb_key': 'metrics/combination/distance_mh'},
    
    # Dissimilarity_count - avg_nbr_changes = (1/|C|m)ΣΣ 1_{ci≠xi}
    'avg_nbr_changes': {'goal': 'minimize', 'description': 'Dissimilarity_count - proportion of features changed', 'wandb_key': 'metrics/combination/avg_nbr_changes'},
    
    # Diversity_dist - diversity_mh = (1/|C|²)ΣΣ d(c, c')
    'diversity_mh': {'goal': 'maximize', 'description': 'Diversity_dist - pairwise CF diversity', 'wandb_key': 'metrics/combination/diversity_mh'},
    
    # Diversity_count - count_diversity_all = (1/|C|²m)ΣΣΣ 1_{ci≠c'i}
    'count_diversity_all': {'goal': 'maximize', 'description': 'Diversity_count - feature diversity', 'wandb_key': 'metrics/combination/count_diversity_all'},
    
    # Discriminative Power (dipo) - accuracy_knn_sklearn = 1NN accuracy
    'accuracy_knn_sklearn': {'goal': 'maximize', 'description': 'Discriminative Power - 1NN accuracy', 'wandb_key': 'metrics/combination/accuracy_knn_sklearn'},
    
    # Runtime
    'runtime': {'goal': 'minimize', 'description': 'Execution time in seconds', 'wandb_key': 'metrics/combination/runtime'},
}

# 9 metrics for sweep optimization (from Guidotti's paper summary table)
RECOMMENDED_METRICS = [
    'plausibility_nbr_cf',     # Implausibility (DEFAULT)
    'perc_actionable_cf_all',  # Actionability
    'perc_valid_cf_all',       # Size
    'distance_mh',             # Dissimilarity_dist
    'avg_nbr_changes',         # Dissimilarity_count
    'diversity_mh',            # Diversity_dist
    'count_diversity_all',     # Diversity_count
    'accuracy_knn_sklearn',    # Discriminative Power
    'runtime',                 # Runtime
]


def get_sweep_config(dataset: str, target_metric: str, entity: Optional[str] = None) -> Dict[str, Any]:
    """Generate sweep configuration for a specific metric.
    
    Args:
        dataset: Dataset name
        target_metric: Metric to optimize
        entity: WandB entity (team/user)
        
    Returns:
        Sweep configuration dict
    """
    if target_metric not in TARGET_METRICS:
        raise ValueError(f"Unknown metric: {target_metric}. Available: {list(TARGET_METRICS.keys())}")
    
    metric_info = TARGET_METRICS[target_metric]
    # Use the wandb_key for the sweep metric name (e.g., 'metrics/combination/perc_valid_cf_all')
    wandb_metric_key = metric_info['wandb_key']
    
    return {
        'program': 'scripts/run_sweep.py',
        'method': 'random',
        'name': f'dpg_sweep_{dataset}_{target_metric}',
        'metric': {
            'name': wandb_metric_key,
            'goal': metric_info['goal'],
        },
        'parameters': {
            'population_size': {
                'distribution': 'int_uniform',
                'min': 20,
                'max': 100,
            },
            'max_generations': {
                'distribution': 'int_uniform',
                'min': 30,
                'max': 200,
            },
            'mutation_rate': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.5,
            },
            'diversity_weight': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 3.0,
            },
            'repulsion_weight': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 10.0,
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 3,
            's': 2,
        },
        'command': [
            '${env}',
            '.venv/bin/python',
            '${program}',
            '--dataset', dataset,
            '--method', 'dpg',
            '--target-metric', target_metric,
            '${args}',
        ],
    }


def run_single_sweep_experiment(
    dataset: str,
    method: str = 'dpg',
    target_metric: str = 'plausibility_nbr_cf',
    offline: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment as part of a sweep.
    
    This function is called by wandb.agent() for each sweep run.
    It reads hyperparameters from wandb.config and runs the experiment.
    
    Args:
        dataset: Dataset name
        method: Method to use (dpg or dice)
        target_metric: Metric being optimized
        offline: Whether to run in offline mode
        
    Returns:
        Dict with experiment results
    """
    # Initialize wandb run (sweep agent handles this, but we need the config)
    run = wandb.run
    
    if run is None:
        print("ERROR: No active wandb run. This script should be called by wandb agent.")
        return {'error': 'no_wandb_run'}
    
    # Get hyperparameters from sweep
    sweep_config = dict(wandb.config)
    
    print(f"\n{'='*60}")
    print(f"SWEEP EXPERIMENT: {dataset}/{method}")
    print(f"Target Metric: {target_metric}")
    print(f"Hyperparameters from sweep:")
    for key, value in sweep_config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Load base configuration
    config_path = REPO_ROOT / 'configs' / dataset / 'config.yaml'
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return {'error': 'config_not_found'}
    
    config = load_config(str(config_path), method=method, repo_root=str(REPO_ROOT))
    
    # Apply sweep hyperparameters to config
    # These go under methods._default or methods.dpg
    overrides = []
    for param, value in sweep_config.items():
        if param in ['population_size', 'max_generations', 'mutation_rate', 
                     'diversity_weight', 'repulsion_weight', 'boundary_weight',
                     'distance_factor', 'sparsity_factor', 'constraints_factor',
                     'original_escape_weight', 'escape_pressure']:
            overrides.append(f"methods._default.{param}={value}")
    
    if overrides:
        config = apply_overrides(config, overrides)
    
    # Import and run experiment
    from scripts.run_experiment import run_experiment
    
    # Run the experiment
    try:
        results = run_experiment(config, wandb_run=run)
        
        # Extract the target metric from results
        # Metrics are already logged under 'metrics/combination/' by run_experiment
        # The sweep config uses wandb_key (e.g., 'metrics/combination/perc_valid_cf_all')
        # so WandB can pick it up automatically from the logged metrics
        
        if results and 'aggregated_metrics' in results:
            agg_metrics = results['aggregated_metrics']
            # Look for the metric in aggregated_metrics (without the prefix)
            target_value = agg_metrics.get(target_metric, None)
            
            if target_value is not None:
                # Get the wandb_key for this metric
                metric_info = TARGET_METRICS.get(target_metric, {})
                wandb_key = metric_info.get('wandb_key', f'metrics/combination/{target_metric}')
                
                # Log the target metric explicitly for sweep optimization
                # This ensures the sweep can find it at the expected path
                wandb.log({wandb_key: target_value})
                wandb.summary[wandb_key] = target_value
                print(f"\n✓ Target metric '{target_metric}' ({wandb_key}): {target_value}")
            else:
                print(f"\n⚠ Target metric '{target_metric}' not found in aggregated_metrics")
                print(f"  Available metrics: {list(agg_metrics.keys())[:10]}...")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def init_sweep(
    dataset: str,
    target_metric: str = 'plausibility_nbr_cf',
    project: str = 'CounterFactualDPG',
    entity: Optional[str] = None,
) -> str:
    """Initialize a new WandB sweep.
    
    Args:
        dataset: Dataset name
        target_metric: Metric to optimize
        project: WandB project name
        entity: WandB entity (team/user)
        
    Returns:
        Sweep ID
    """
    sweep_config = get_sweep_config(dataset, target_metric, entity)
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )
    
    print(f"\n{'='*60}")
    print(f"SWEEP INITIALIZED")
    print(f"{'='*60}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Dataset: {dataset}")
    print(f"Target Metric: {target_metric} ({TARGET_METRICS[target_metric]['goal']})")
    print(f"Project: {project}")
    if entity:
        print(f"Entity: {entity}")
    print(f"\nTo run the sweep agent:")
    print(f"  wandb agent {entity+'/' if entity else ''}{project}/{sweep_id}")
    print(f"\nOr use this script:")
    print(f"  python scripts/run_sweep.py --run-agent --sweep-id {sweep_id}")
    print(f"{'='*60}\n")
    
    return sweep_id


def run_agent(
    sweep_id: str,
    project: str = 'CounterFactualDPG',
    entity: Optional[str] = None,
    count: Optional[int] = None,
    dataset: str = 'iris',
    target_metric: str = 'plausibility_nbr_cf',
) -> None:
    """Run a WandB sweep agent.
    
    Args:
        sweep_id: ID of the sweep to run
        project: WandB project name
        entity: WandB entity (team/user)
        count: Maximum number of runs (None for unlimited)
        dataset: Dataset name (for the experiment function)
        target_metric: Target metric being optimized
    """
    def sweep_train():
        """Wrapper function for sweep agent."""
        # Initialize the wandb run - agent provides the config
        run = wandb.init()
        try:
            run_single_sweep_experiment(
                dataset=dataset,
                method='dpg',
                target_metric=target_metric,
            )
        finally:
            wandb.finish()
    
    sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    
    print(f"\n{'='*60}")
    print(f"RUNNING SWEEP AGENT")
    print(f"{'='*60}")
    print(f"Sweep: {sweep_path}")
    print(f"Max runs: {count if count else 'unlimited'}")
    print(f"Dataset: {dataset}")
    print(f"Target Metric: {target_metric}")
    print(f"{'='*60}\n")
    
    wandb.agent(
        sweep_id,
        function=sweep_train,
        project=project,
        entity=entity,
        count=count,
    )


def main():
    parser = argparse.ArgumentParser(
        description="WandB Sweep Runner for DPG Counterfactual Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--init-sweep', action='store_true',
                           help='Initialize a new sweep and print sweep ID')
    mode_group.add_argument('--run-agent', action='store_true',
                           help='Run a sweep agent for an existing sweep')
    mode_group.add_argument('--list-metrics', action='store_true',
                           help='List all available target metrics')
    
    # Sweep configuration
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='Sweep ID (required for --run-agent)')
    parser.add_argument('--count', type=int, default=None,
                       help='Maximum number of runs for agent')
    
    # Experiment configuration
    parser.add_argument('--dataset', type=str, default='iris',
                       help='Dataset to use (default: iris)')
    parser.add_argument('--method', type=str, default='dpg',
                       help='Method to use (default: dpg)')
    parser.add_argument('--target-metric', type=str, default='plausibility_nbr_cf',
                       help='Metric to optimize (default: plausibility_nbr_cf)')
    
    # WandB configuration
    parser.add_argument('--project', type=str, default='CounterFactualDPG',
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='WandB entity (team/user)')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode')
    
    # Sweep parameter overrides (passed by wandb agent)
    parser.add_argument('--population_size', type=int, default=None)
    parser.add_argument('--max_generations', type=int, default=None)
    parser.add_argument('--mutation_rate', type=float, default=None)
    parser.add_argument('--diversity_weight', type=float, default=None)
    parser.add_argument('--repulsion_weight', type=float, default=None)
    
    args = parser.parse_args()
    
    # List metrics mode
    if args.list_metrics:
        print("\nAvailable Target Metrics for Sweep Optimization:")
        print("=" * 70)
        print(f"{'Metric Name':<28} {'Goal':<10} {'Description'}")
        print("-" * 70)
        for metric in RECOMMENDED_METRICS:
            info = TARGET_METRICS[metric]
            marker = "★" if metric == 'plausibility_nbr_cf' else " "
            print(f"{marker} {metric:<26} {info['goal']:<10} {info['description']}")
        print("-" * 70)
        print("★ = Default metric")
        print("\nAll available metrics:")
        for metric, info in TARGET_METRICS.items():
            if metric not in RECOMMENDED_METRICS:
                print(f"  {metric:<26} {info['goal']:<10} {info['description']}")
        return 0
    
    # Validate target metric
    if args.target_metric not in TARGET_METRICS:
        print(f"ERROR: Unknown target metric '{args.target_metric}'")
        print(f"Available metrics: {', '.join(TARGET_METRICS.keys())}")
        return 1
    
    # Initialize sweep mode
    if args.init_sweep:
        init_sweep(
            dataset=args.dataset,
            target_metric=args.target_metric,
            project=args.project,
            entity=args.entity,
        )
        return 0
    
    # Run agent mode
    if args.run_agent:
        if not args.sweep_id:
            print("ERROR: --sweep-id is required when using --run-agent")
            return 1
        run_agent(
            sweep_id=args.sweep_id,
            project=args.project,
            entity=args.entity,
            count=args.count,
            dataset=args.dataset,
            target_metric=args.target_metric,
        )
        return 0
    
    # Default: run as sweep experiment (called by wandb agent)
    # Check if we're being called by wandb agent (wandb.run exists)
    if wandb.run is not None:
        # Running as part of a sweep
        run_single_sweep_experiment(
            dataset=args.dataset,
            method=args.method,
            target_metric=args.target_metric,
            offline=args.offline,
        )
    else:
        # Not in a sweep - initialize a run manually with provided params
        print("Starting standalone sweep experiment run...")
        
        # Build config from CLI args
        sweep_params = {}
        if args.population_size is not None:
            sweep_params['population_size'] = args.population_size
        if args.max_generations is not None:
            sweep_params['max_generations'] = args.max_generations
        if args.mutation_rate is not None:
            sweep_params['mutation_rate'] = args.mutation_rate
        if args.diversity_weight is not None:
            sweep_params['diversity_weight'] = args.diversity_weight
        if args.repulsion_weight is not None:
            sweep_params['repulsion_weight'] = args.repulsion_weight
        
        if not sweep_params:
            # No params provided, show help
            print("\nNo sweep parameters provided. Use one of:")
            print("  --init-sweep    : Initialize a new sweep")
            print("  --run-agent     : Run an existing sweep")
            print("  --list-metrics  : Show available target metrics")
            print("\nOr provide hyperparameters manually:")
            print("  --population_size, --max_generations, --mutation_rate, etc.")
            return 1
        
        # Initialize wandb run
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            config=sweep_params,
            name=f"sweep_manual_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode='offline' if args.offline else 'online',
        )
        
        try:
            run_single_sweep_experiment(
                dataset=args.dataset,
                method=args.method,
                target_metric=args.target_metric,
                offline=args.offline,
            )
        finally:
            wandb.finish()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
