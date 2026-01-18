#!/usr/bin/env python3
"""WandB Sweep Runner for DPG Counterfactual Generation.

This script runs a single experiment as part of a WandB hyperparameter sweep,
optimizing the genetic algorithm parameters for counterfactual quality.

Usage:
  # Run as part of a sweep (called by wandb agent)
  python scripts/run_sweep.py --dataset iris --method dpg
  
  # Run with specific target metric
  python scripts/run_sweep.py --dataset iris --target-metric plausibility_sum
  python scripts/run_sweep.py --dataset iris --target-metric perc_valid_cf
  
  # Initialize a new sweep
  python scripts/run_sweep.py --init-sweep --dataset iris --target-metric perc_valid_cf
  
  # Run sweep agent (after initialization)
  python scripts/run_sweep.py --run-agent --sweep-id <sweep_id> --count 10

Available Target Metrics (9 recommended):
  Metric Name               | Goal     | Description
  --------------------------|----------|---------------------------------------------
  plausibility_sum          | minimize | Distance to nearest training sample (DEFAULT)
  perc_valid_cf             | maximize | Percentage of valid counterfactuals
  distance_l2               | minimize | Euclidean distance from original
  distance_mad              | minimize | MAD-normalized distance
  avg_nbr_changes_per_cf    | minimize | Feature sparsity
  diversity_l2              | maximize | Pairwise diversity among CFs
  perc_valid_actionable_cf  | maximize | Valid AND actionable CFs percentage
  accuracy_knn_sklearn      | maximize | KNN fidelity score
  delta                     | maximize | Mean prediction probability change
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

# Define target metrics with their optimization goals
TARGET_METRICS = {
    # Plausibility (default)
    'plausibility_sum': {'goal': 'minimize', 'description': 'Distance to nearest training sample'},
    'plausibility_nbr_cf': {'goal': 'minimize', 'description': 'Normalized plausibility'},
    
    # Validity
    'perc_valid_cf': {'goal': 'maximize', 'description': 'Percentage of valid CFs'},
    'nbr_valid_cf': {'goal': 'maximize', 'description': 'Number of valid CFs'},
    
    # Distance
    'distance_l2': {'goal': 'minimize', 'description': 'Euclidean distance'},
    'distance_mad': {'goal': 'minimize', 'description': 'MAD-normalized distance'},
    
    # Sparsity
    'avg_nbr_changes_per_cf': {'goal': 'minimize', 'description': 'Average feature changes'},
    'avg_nbr_changes': {'goal': 'minimize', 'description': 'Normalized sparsity'},
    
    # Diversity
    'diversity_l2': {'goal': 'maximize', 'description': 'Pairwise L2 diversity'},
    'diversity_mad': {'goal': 'maximize', 'description': 'Pairwise MAD diversity'},
    
    # Actionability
    'perc_valid_actionable_cf': {'goal': 'maximize', 'description': 'Valid & actionable percentage'},
    'perc_actionable_cf': {'goal': 'maximize', 'description': 'Actionable percentage'},
    
    # Fidelity
    'accuracy_knn_sklearn': {'goal': 'maximize', 'description': 'KNN accuracy'},
    'accuracy_knn_dist': {'goal': 'maximize', 'description': 'Distance-based KNN accuracy'},
    
    # Delta
    'delta': {'goal': 'maximize', 'description': 'Mean prediction change'},
    'delta_max': {'goal': 'maximize', 'description': 'Max prediction change'},
}

# 9 recommended metrics for sweep optimization
RECOMMENDED_METRICS = [
    'plausibility_sum',
    'perc_valid_cf', 
    'distance_l2',
    'distance_mad',
    'avg_nbr_changes_per_cf',
    'diversity_l2',
    'perc_valid_actionable_cf',
    'accuracy_knn_sklearn',
    'delta',
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
    
    return {
        'program': 'scripts/run_sweep.py',
        'method': 'random',
        'name': f'dpg_sweep_{dataset}_{target_metric}',
        'metric': {
            'name': target_metric,
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
            'python',
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
    target_metric: str = 'plausibility_sum',
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
    configs_dir = REPO_ROOT / 'configs'
    config = load_config(configs_dir, dataset, method)
    
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
        if results and 'aggregated_metrics' in results:
            agg_metrics = results['aggregated_metrics']
            target_value = agg_metrics.get(target_metric, None)
            
            if target_value is not None:
                # Log the target metric explicitly for sweep optimization
                wandb.log({target_metric: target_value})
                wandb.summary[target_metric] = target_value
                print(f"\n✓ Target metric '{target_metric}': {target_value}")
            else:
                print(f"\n⚠ Target metric '{target_metric}' not found in results")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def init_sweep(
    dataset: str,
    target_metric: str = 'plausibility_sum',
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
    target_metric: str = 'plausibility_sum',
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
        run_single_sweep_experiment(
            dataset=dataset,
            method='dpg',
            target_metric=target_metric,
        )
    
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
    parser.add_argument('--target-metric', type=str, default='plausibility_sum',
                       help='Metric to optimize (default: plausibility_sum)')
    
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
            marker = "★" if metric == 'plausibility_sum' else " "
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
