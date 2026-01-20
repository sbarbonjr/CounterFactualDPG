"""Enhanced experiment runner with WandB integration.

This script provides a parameterized way to run counterfactual generation experiments
with automatic logging to Weights & Biases for experiment tracking and comparison.

Usage:
  # Run with unified config (recommended)
  python scripts/run_experiment.py --dataset german_credit --method dpg
  python scripts/run_experiment.py --dataset iris --method dice
  
  # Run with explicit config path (legacy or unified)
  python scripts/run_experiment.py --config configs/german_credit/config.yaml --method dpg
  
  # Override specific params
  python scripts/run_experiment.py --config configs/iris/dpg/config.yaml \
    --set counterfactual.population_size=50 \
    --set experiment_params.seed=123
    
  # Resume a previous run
  python scripts/run_experiment.py --resume <wandb_run_id>
  
  # Offline mode (no wandb sync)
  python scripts/run_experiment.py --config configs/iris/dpg/config.yaml --offline
"""

from __future__ import annotations

import pathlib
import sys

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os
import pickle
import argparse
import traceback
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    print("Warning: dice-ml not available. Install with: pip install dice-ml")

try:
    from cf_eval.metrics import (
        nbr_valid_cf,
        perc_valid_cf,
        continuous_distance,
        avg_nbr_changes_per_cf,
        nbr_changes_per_cf,
    )
    CF_EVAL_AVAILABLE = True
except ImportError:
    CF_EVAL_AVAILABLE = False

from CounterFactualModel import CounterFactualModel
from ConstraintParser import ConstraintParser
from CounterFactualExplainer import CounterFactualExplainer

# Import comprehensive metrics
try:
    from CounterFactualMetrics import evaluate_cf_list as evaluate_cf_list_comprehensive
    COMPREHENSIVE_METRICS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_METRICS_AVAILABLE = False
    print("Warning: CounterFactualMetrics not available. Install scipy and sklearn for comprehensive metrics.")
import CounterFactualVisualizer as CounterFactualVisualizer
from CounterFactualVisualizer import (
    plot_sample_and_counterfactual_heatmap,
    plot_sample_and_counterfactual_comparison,
    plot_pairwise_with_counterfactual_df,
    plot_pca_with_counterfactuals,
    plot_pca_loadings,
    plot_dpg_constraints_overview,
)

from utils.notebooks.experiment_storage import (
    get_sample_id,
    save_sample_metadata,
    save_visualizations_data,
    _get_sample_dir as get_sample_dir,
)
from utils.dataset_loader import load_dataset, determine_feature_types
from utils.config_manager import (
    DictConfig,
    deep_merge_dicts,
    load_config,
    apply_overrides,
    build_dict_non_actionable,
)
from utils.replication_runner import (
    _run_single_replication_dpg,
    _run_single_replication_dice,
    _run_single_replication,
)
from utils.wandb_helper import (
    init_wandb,
    configure_wandb_metrics,
)
from utils.experiment_status import (
    PersistentStatus,
    write_status,
    read_status,
    get_log_file_path,
    append_log,
    clear_log,
)

# Config utilities moved to utils/config_manager.py
# Replication workers moved to utils/replication_runner.py
# WandB utilities moved to utils/wandb_helper.py
# WandB utilities moved to utils/wandb_helper.py


# Replication worker functions moved to utils/replication_runner.py


def run_single_sample(
    sample_index: int,
    config: DictConfig,
    model,
    constraints: Dict,
    dataset_data: Dict,
    class_colors_list: List[str],
    wandb_run=None,
    normalized_constraints=None
) -> Dict[str, Any]:
    """Run counterfactual generation for a single sample with WandB logging.
    
    Args:
        sample_index: Index of the sample in the training split
        config: Experiment configuration
        model: Trained classifier model
        constraints: Feature constraints from DPG
        dataset_data: Dict containing features, labels, feature_names, train_features, train_labels
        class_colors_list: List of colors for visualization
        wandb_run: WandB run object for logging
        normalized_constraints: Normalized DPG constraints dict (optional)
        
    Returns:
        Dict with results including sample_id, success_rate, and file paths
    """
    
    FEATURES = dataset_data['features']
    LABELS = dataset_data['labels']
    FEATURE_NAMES = dataset_data['feature_names']
    TRAIN_FEATURES = dataset_data['train_features']
    TRAIN_LABELS = dataset_data['train_labels']
    
    # Get feature type indices for comprehensive metrics
    CONTINUOUS_INDICES = dataset_data.get('continuous_indices', list(range(len(FEATURE_NAMES))))
    CATEGORICAL_INDICES = dataset_data.get('categorical_indices', [])
    VARIABLE_INDICES = dataset_data.get('variable_indices', list(range(len(FEATURE_NAMES))))
    
    # Prepare training/test data arrays for comprehensive metrics
    X_TRAIN = TRAIN_FEATURES.values if hasattr(TRAIN_FEATURES, 'values') else TRAIN_FEATURES
    X_TEST = FEATURES  # Use full dataset as test for metrics computation
    
    # Compute ratio of continuous features
    nbr_features = len(FEATURE_NAMES)
    ratio_cont = len(CONTINUOUS_INDICES) / nbr_features if nbr_features > 0 else 1.0
    
    output_dir = getattr(config.output, 'local_dir', 'outputs') if hasattr(config, 'output') else 'outputs'
    
    # Get original sample from training split
    original_sample_values = TRAIN_FEATURES.iloc[sample_index].values if hasattr(TRAIN_FEATURES, 'iloc') else TRAIN_FEATURES[sample_index]
    ORIGINAL_SAMPLE = dict(zip(FEATURE_NAMES, map(float, original_sample_values)))
    SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])
    ORIGINAL_SAMPLE_PREDICTED_CLASS = int(model.predict(SAMPLE_DATAFRAME)[0])
    
    # Prepare CF parameters
    FEATURES_NAMES = list(ORIGINAL_SAMPLE.keys())
    
    # Check if using per-feature rules (preferred) or legacy rule combinations
    has_actionability = hasattr(config.counterfactual, 'actionability') and config.counterfactual.actionability
    has_feature_rules = hasattr(config.counterfactual, 'feature_rules') and config.counterfactual.feature_rules
    use_feature_rules = has_actionability or has_feature_rules
    
    if use_feature_rules:
        # Use per-feature rules directly (no combination testing)
        RULES_COMBINATIONS = [None]  # Single run
        num_combinations_to_test = 1
        print(f"INFO: Using per-feature actionability rules from config")
    else:
        # Legacy: test combinations of rules across all features
        RULES = config.counterfactual.rules
        RULES_COMBINATIONS = list(__import__('itertools').product(RULES, repeat=len(FEATURES_NAMES)))
        num_combinations_to_test = config.experiment_params.num_combinations_to_test
        if num_combinations_to_test is None:
            num_combinations_to_test = int(len(RULES_COMBINATIONS) / 2)
        print(f"INFO: Testing {num_combinations_to_test} rule combinations (legacy mode)")
    
    # Choose target class
    n_classes = len(np.unique(LABELS))
    TARGET_CLASS = 0 if ORIGINAL_SAMPLE_PREDICTED_CLASS != 0 else (ORIGINAL_SAMPLE_PREDICTED_CLASS + 1) % n_classes
    
    # Save sample metadata
    SAMPLE_ID = get_sample_id(sample_index)
    configname = getattr(config.experiment, 'name', None)
    save_sample_metadata(SAMPLE_ID, ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, TARGET_CLASS, sample_index, configname=configname, output_dir=output_dir) 
    
    # Define sample directory early for use throughout the function
    sample_dir = get_sample_dir(SAMPLE_ID, output_dir=output_dir, configname=configname)
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"INFO: Processing Sample ID: {SAMPLE_ID} (dataset index: {sample_index})")
    print(f"INFO: Original Predicted Class: {ORIGINAL_SAMPLE_PREDICTED_CLASS}, Target Class: {TARGET_CLASS}, combinations to test: {num_combinations_to_test}/{len(RULES_COMBINATIONS)}")
    
    # Log sample info to WandB as summary (these are single values per sample, not time series)
    if wandb_run:
        wandb.run.summary[f"sample_{SAMPLE_ID}/sample_index"] = sample_index
        wandb.run.summary[f"sample_{SAMPLE_ID}/original_class"] = ORIGINAL_SAMPLE_PREDICTED_CLASS
        wandb.run.summary[f"sample_{SAMPLE_ID}/target_class"] = TARGET_CLASS
        wandb.run.summary[f"sample_{SAMPLE_ID}/num_combinations_tested"] = num_combinations_to_test
    
    counterfactuals_df_combinations = []
    visualizations = []
    
    valid_counterfactuals = 0
    total_replications = 0
    
    # Determine parallelization settings
    use_parallel = getattr(config.experiment_params, 'parallel_replications', True)
    max_workers = getattr(config.experiment_params, 'max_workers', None)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Helper to get a descriptive label for logging
    def get_combination_label(combination, dict_non_actionable):
        """Get a descriptive label for logging purposes."""
        if combination is not None:
            return str(combination)
        # For per-feature rules, create a concise summary
        non_none_rules = {f: r for f, r in dict_non_actionable.items() if r != "none"}
        return f"per_feature_rules:{len(non_none_rules)}_constraints"
    
    # Loop through combinations
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:num_combinations_to_test]):
        # Build dict_non_actionable based on config (per-feature rules or combination)
        if use_feature_rules:
            # Use per-feature rules from config
            dict_non_actionable = build_dict_non_actionable(config, FEATURES_NAMES, VARIABLE_INDICES)
        else:
            # Legacy: build from rule combination
            dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
            # Force non-actionable features (not in VARIABLE_INDICES) to "no_change"
            for idx, feature_name in enumerate(FEATURES_NAMES):
                if idx not in VARIABLE_INDICES:
                    dict_non_actionable[feature_name] = "no_change"
        
        # Log actionability rules on first iteration
        if combination_num == 0:
            frozen_features = [f for f, rule in dict_non_actionable.items() if rule == "no_change"]
            directional_features = {f: rule for f, rule in dict_non_actionable.items() 
                                   if rule in ["non_increasing", "non_decreasing"]}
            if frozen_features:
                print(f"INFO: Freezing {len(frozen_features)} non-actionable features: {frozen_features}")
            if directional_features:
                print(f"INFO: Directional constraints: {directional_features}")
        
        counterfactuals_df_replications = []
        # Store dict_non_actionable directly for visualization (works for both per-feature rules and legacy mode)
        combination_viz = {'label': dict_non_actionable, 'pairwise': None, 'pca': None, 'replication': []}
        
        skip_combination = False
        
        # Prepare training DataFrame with target for DiCE
        # DiCE needs a DataFrame with features + outcome column
        train_df_for_dice = TRAIN_FEATURES.copy() if hasattr(TRAIN_FEATURES, 'copy') else pd.DataFrame(TRAIN_FEATURES, columns=FEATURE_NAMES)
        train_df_for_dice['_target_'] = TRAIN_LABELS
        
        # Prepare arguments for parallel execution
        replication_args = [
            (
                replication,
                ORIGINAL_SAMPLE,
                TARGET_CLASS,
                FEATURES_NAMES,
                dict_non_actionable,
                config.to_dict(),
                model,
                constraints,
                train_df_for_dice,  # Training data for DiCE
                CONTINUOUS_INDICES,  # Continuous feature indices
                CATEGORICAL_INDICES,  # Categorical feature indices
            )
            for replication in range(config.experiment_params.num_replications)
        ]
        
        # Run replications in parallel or sequential based on config
        replication_results = []
        if use_parallel and config.experiment_params.num_replications > 1:
            print(f"INFO: Running {config.experiment_params.num_replications} replications in parallel (max_workers={max_workers})")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_single_replication, args): args[0] 
                          for args in replication_args}
                
                for future in as_completed(futures):
                    replication_num = futures[future]
                    total_replications += 1
                    try:
                        result = future.result()
                        if result is not None:
                            replication_results.append(result)
                    except Exception as exc:
                        print(f"WARNING: Replication {replication_num} failed with exception: {exc}")
                        if wandb_run:
                            wandb.log({
                                "replication/sample_id": SAMPLE_ID,
                                "replication/combination": get_combination_label(combination, dict_non_actionable),
                                "replication/replication_num": replication_num,
                                "replication/success": False,
                            })
        else:
            # Sequential execution for backward compatibility or when parallel is disabled
            for args in replication_args:
                replication_num = args[0]
                total_replications += 1
                result = _run_single_replication(args)
                if result is not None:
                    replication_results.append(result)
                elif wandb_run:
                    wandb.log({
                        "replication/sample_id": SAMPLE_ID,
                        "replication/combination": get_combination_label(combination, dict_non_actionable),
                        "replication/replication_num": replication_num,
                        "replication/success": False,
                    })
        
        # Process results from all replications
        if not replication_results:
            skip_combination = True
        
        # Determine method for this run
        cf_method = getattr(config.counterfactual, 'method', 'dpg').lower()
        
        for result in replication_results:
            valid_counterfactuals += 1
            counterfactual = result['counterfactual']
            all_counterfactuals = result.get('all_counterfactuals', [counterfactual])
            evolution_history = result['evolution_history']
            best_fitness_list = result['best_fitness_list']
            average_fitness_list = result['average_fitness_list']
            replication_num = result['replication_num']
            result_method = result.get('method', 'dpg')
            
            # Calculate final best fitness
            best_fitness = best_fitness_list[-1] if best_fitness_list else 0.0
            
            # Create cf_model based on method
            if result_method == 'dice':
                # For DiCE, create a minimal cf_model for compatibility with explainer
                # Note: The explainer accepts cf_model=None for DiCE cases
                cf_model = CounterFactualModel(
                    model,
                    constraints,
                    dict_non_actionable=dict_non_actionable,
                    verbose=False,
                )
                # Set empty fitness lists for DiCE (no evolution tracking)
                cf_model.best_fitness_list = []
                cf_model.average_fitness_list = []
                cf_model.evolution_history = evolution_history
            else:
                # Recreate cf_model with stored DPG parameters (including dual-boundary parameters)
                cf_model = CounterFactualModel(
                    model,
                    constraints,
                    dict_non_actionable=dict_non_actionable,
                    verbose=False,
                    diversity_weight=result.get('diversity_weight', 0.5),
                    repulsion_weight=result.get('repulsion_weight', 4.0),
                    boundary_weight=result.get('boundary_weight', 15.0),
                    distance_factor=result.get('distance_factor', 2.0),
                    sparsity_factor=result.get('sparsity_factor', 1.0),
                    constraints_factor=result.get('constraints_factor', 3.0),
                    # Dual-boundary parameters
                    original_escape_weight=result.get('original_escape_weight', 2.0),
                    escape_pressure=result.get('escape_pressure', 0.5),
                    prioritize_non_overlapping=result.get('prioritize_non_overlapping', True),
                    # Fitness calculation parameters
                    max_bonus_cap=result.get('max_bonus_cap', 50.0),
                )
                # Restore fitness history
                cf_model.best_fitness_list = best_fitness_list
                cf_model.average_fitness_list = average_fitness_list
                cf_model.evolution_history = evolution_history
            
            # Store replication data with evolution history
            replication_viz = {
                'counterfactual': counterfactual,
                'all_counterfactuals': all_counterfactuals,
                'cf_model': cf_model,
                'evolution_history': evolution_history,
                'visualizations': [],
                'explanations': {},
                'replication_num': replication_num,
                'success': True,
                'best_fitness': best_fitness,
                'best_fitness_list': cf_model.best_fitness_list if hasattr(cf_model, 'best_fitness_list') else [],
                'method': result_method,
            }
            combination_viz['replication'].append(replication_viz)
            
            cf_data = counterfactual.copy()
            cf_data.update({'Rule_' + k: v for k, v in dict_non_actionable.items()})
            cf_data['Replication'] = replication_num + 1
            counterfactuals_df_replications.append(cf_data)
            
            # Get metrics from explainer with comprehensive evaluation
            explainer = CounterFactualExplainer(cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
            
            # Enable comprehensive metrics if available and configured
            compute_comprehensive = getattr(config.experiment_params, 'compute_comprehensive_metrics', True)
            
            metrics = explainer.get_all_metrics(
                X_train=X_TRAIN,
                X_test=X_TEST,
                variable_features=VARIABLE_INDICES,
                continuous_features=CONTINUOUS_INDICES,
                categorical_features=CATEGORICAL_INDICES,
                compute_comprehensive=compute_comprehensive and COMPREHENSIVE_METRICS_AVAILABLE
            )
            
            # Store metrics in replication for later aggregation
            replication_viz['metrics'] = metrics
            replication_viz['num_feature_changes'] = metrics.get('num_feature_changes', 0)
            
            # Compute cf_eval metrics if available (for backwards compatibility)
            cf_eval_metrics = {}
            if CF_EVAL_AVAILABLE:
                try:
                    x_original = np.array([ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES])
                    cf_array = np.array([[counterfactual[feat] for feat in FEATURES_NAMES]])
                    continuous_features = CONTINUOUS_INDICES
                    
                    cf_eval_metrics = {
                        'cf_eval/is_valid': int(nbr_valid_cf(cf_array, model, ORIGINAL_SAMPLE_PREDICTED_CLASS, y_desidered=TARGET_CLASS)),
                        'cf_eval/euclidean_distance': float(continuous_distance(x_original, cf_array, continuous_features, metric='euclidean')),
                        'cf_eval/manhattan_distance': float(continuous_distance(x_original, cf_array, continuous_features, metric='manhattan')),
                        'cf_eval/num_changes': float(avg_nbr_changes_per_cf(x_original, cf_array, continuous_features)),
                    }
                except Exception as exc:
                    print(f"WARNING: cf_eval metrics computation failed: {exc}")
            
            # Log replication metrics to WandB
            if wandb_run:
                best_fitness = cf_model.best_fitness_list[-1] if cf_model.best_fitness_list else None
                
                log_data = {
                    "replication/sample_id": SAMPLE_ID,
                    "replication/combination": get_combination_label(combination, dict_non_actionable),
                    "replication/replication_num": replication_num,
                    "replication/success": True,
                    "replication/final_fitness": best_fitness,
                    "replication/generations_to_converge": len(cf_model.best_fitness_list),
                    "replication/num_feature_changes": metrics['num_feature_changes'],
                    "replication/constraints_respected": metrics['constraints_respected'],
                    "replication/method": result_method,  # Track which method was used
                }
                
                # Add all metrics from explainer (per-counterfactual level)
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        log_data[f"metrics/per_counterfactual/{key}"] = value
                
                # Add cf_eval metrics
                log_data.update(cf_eval_metrics)
                
                wandb.log(log_data)
                
                # Log fitness evolution with generation as x-axis
                # Create separate series per replication for better comparison
                if cf_model.best_fitness_list and cf_model.average_fitness_list:
                    replication_key = f"s{SAMPLE_ID}_c{combination_num}_r{replication_num}"
                    for gen, (best, avg) in enumerate(zip(cf_model.best_fitness_list, cf_model.average_fitness_list)):
                        wandb.log({
                            "generation": gen,
                            f"fitness/best_{replication_key}": best,
                            f"fitness/avg_{replication_key}": avg,
                        })
                        
                    # Also log metadata about this fitness series to summary
                    wandb.run.summary[f"fitness_metadata/{replication_key}/sample_id"] = SAMPLE_ID
                    wandb.run.summary[f"fitness_metadata/{replication_key}/combination"] = get_combination_label(combination, dict_non_actionable)
                    wandb.run.summary[f"fitness_metadata/{replication_key}/replication"] = replication_num
                    wandb.run.summary[f"fitness_metadata/{replication_key}/final_fitness"] = best_fitness
                    wandb.run.summary[f"fitness_metadata/{replication_key}/generations"] = len(cf_model.best_fitness_list)
                
                # Save fitness data locally as CSV
                if getattr(config.output, 'save_visualization_images', False):
                    os.makedirs(sample_dir, exist_ok=True)
                    if cf_model.best_fitness_list and cf_model.average_fitness_list:
                        fitness_data = []
                        for gen, (best, avg) in enumerate(zip(cf_model.best_fitness_list, cf_model.average_fitness_list)):
                            fitness_data.append({
                                'generation': gen,
                                'best_fitness': best,
                                'average_fitness': avg
                            })
                        fitness_df = pd.DataFrame(fitness_data)
                        fitness_csv_path = os.path.join(sample_dir, f'fitness_combo_{combination_num}_rep_{replication_num}.csv')
                        fitness_df.to_csv(fitness_csv_path, index=False)
        
        if counterfactuals_df_replications:
            counterfactuals_df_replications = pd.DataFrame(counterfactuals_df_replications)
            counterfactuals_df_combinations.extend(counterfactuals_df_replications.to_dict('records'))
        
        # Compute combination-level comprehensive metrics
        combination_comprehensive_metrics = {}
        if combination_viz['replication'] and COMPREHENSIVE_METRICS_AVAILABLE:
            try:
                x_original = np.array([ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES])
                cf_list = [np.array([rep['counterfactual'][feat] for feat in FEATURES_NAMES]) 
                          for rep in combination_viz['replication']]
                cf_array = np.array(cf_list)
                
                # Compute comprehensive metrics for this combination
                combination_comprehensive_metrics = evaluate_cf_list_comprehensive(
                    cf_list=cf_array,
                    x=x_original,
                    model=model,
                    y_val=ORIGINAL_SAMPLE_PREDICTED_CLASS,
                    max_nbr_cf=config.experiment_params.num_replications,
                    variable_features=VARIABLE_INDICES,
                    continuous_features_all=CONTINUOUS_INDICES,
                    categorical_features_all=CATEGORICAL_INDICES,
                    X_train=X_TRAIN,
                    X_test=X_TEST,
                    ratio_cont=ratio_cont,
                    nbr_features=nbr_features
                )
                
                # Store for later persistence
                combination_viz['comprehensive_metrics'] = combination_comprehensive_metrics
                
                # Log to WandB
                if wandb_run:
                    combo_log = {
                        "combo/sample_id": SAMPLE_ID,
                        "combo/combination": get_combination_label(combination, dict_non_actionable),
                    }
                    for key, value in combination_comprehensive_metrics.items():
                        if isinstance(value, (int, float, bool)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                            combo_log[f"metrics/combination/{key}"] = value
                    wandb.log(combo_log)
                
            except Exception as exc:
                print(f"WARNING: Combination-level comprehensive metrics failed: {exc}")
        
        # Compute combination-level cf_eval metrics (for backwards compatibility)
        if combination_viz['replication'] and CF_EVAL_AVAILABLE and wandb_run:
            try:
                x_original = np.array([ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES])
                cf_list = [np.array([rep['counterfactual'][feat] for feat in FEATURES_NAMES]) 
                          for rep in combination_viz['replication']]
                cf_array = np.array(cf_list)
                continuous_features = list(range(len(FEATURES_NAMES)))
                
                num_valid = int(nbr_valid_cf(cf_array, model, ORIGINAL_SAMPLE_PREDICTED_CLASS, y_desidered=TARGET_CLASS))
                pct_valid = float(perc_valid_cf(cf_array, model, ORIGINAL_SAMPLE_PREDICTED_CLASS, y_desidered=TARGET_CLASS))
                avg_distance = float(continuous_distance(x_original, cf_array, continuous_features, metric='euclidean'))
                min_distance = float(continuous_distance(x_original, cf_array, continuous_features, metric='euclidean', agg='min'))
                max_distance = float(continuous_distance(x_original, cf_array, continuous_features, metric='euclidean', agg='max'))
                avg_changes = float(avg_nbr_changes_per_cf(x_original, cf_array, continuous_features))
                
                wandb.log({
                    "combination/sample_id": SAMPLE_ID,
                    "combination/combination": get_combination_label(combination, dict_non_actionable),
                    "combination/num_cfs": len(cf_list),
                    "combination/valid_cfs": num_valid,
                    "combination/validity_pct": pct_valid * 100,
                    "combination/avg_euclidean_distance": avg_distance,
                    "combination/min_euclidean_distance": min_distance,
                    "combination/max_euclidean_distance": max_distance,
                    "combination/avg_num_changes": avg_changes,
                })
            except Exception as exc:
                print(f"WARNING: Combination-level cf_eval metrics failed: {exc}")
        
        if combination_viz['replication']:
            visualizations.append(combination_viz)
    
    # Calculate sample-level metrics
    success_rate = valid_counterfactuals / total_replications if total_replications > 0 else 0.0
    
    if wandb_run:
        # Log sample-level summary statistics (single values per sample)
        wandb.run.summary[f"sample_{SAMPLE_ID}/num_valid_counterfactuals"] = valid_counterfactuals
        wandb.run.summary[f"sample_{SAMPLE_ID}/total_replications"] = total_replications
        wandb.run.summary[f"sample_{SAMPLE_ID}/success_rate"] = success_rate
        
        # Create a table of all replications for this sample (structured view)
        try:
            replication_table_data = []
            for combination_viz in visualizations:
                for rep_viz in combination_viz['replication']:
                    replication_table_data.append([
                        SAMPLE_ID,
                        combination_viz['label'],
                        rep_viz['replication_num'],
                        rep_viz.get('success', False),
                        rep_viz.get('best_fitness', 'N/A'),
                        len(rep_viz.get('best_fitness_list', [])),
                        rep_viz.get('num_feature_changes', 'N/A'),
                    ])
            
            if replication_table_data:
                replication_table = wandb.Table(
                    columns=["Sample ID", "Combination", "Replication", "Success", "Final Fitness", "Generations", "Feature Changes"],
                    data=replication_table_data
                )
                wandb.log({f"tables/sample_{SAMPLE_ID}_replications": replication_table})
        except Exception as exc:
            print(f"WARNING: Failed to create replication table: {exc}")
    
    # Save raw data
    raw_data = {
        'sample_id': SAMPLE_ID,
        'original_sample': ORIGINAL_SAMPLE,
        'target_class': TARGET_CLASS,
        'features_names': FEATURES_NAMES,
        'visualizations_structure': []
    }
    
    for combination_viz in visualizations:
        combo_copy = {'label': combination_viz['label'], 'replication': []}
        for replication_viz in combination_viz['replication']:
            best_fitness_list = getattr(replication_viz['cf_model'], 'best_fitness_list', [])
            average_fitness_list = getattr(replication_viz['cf_model'], 'average_fitness_list', [])
            
            rep_copy = {
                'counterfactual': replication_viz['counterfactual'],
                'best_fitness_list': best_fitness_list,
                'average_fitness_list': average_fitness_list
            }
            combo_copy['replication'].append(rep_copy)
        raw_data['visualizations_structure'].append(combo_copy)
    
    # Save to disk
    raw_filepath = os.path.join(sample_dir, 'raw_counterfactuals.pkl')
    with open(raw_filepath, 'wb') as f:
        pickle.dump(raw_data, f) 
    
    # Save normalized DPG constraints in sample folder
    if normalized_constraints:
        import json
        dpg_json_path = os.path.join(sample_dir, 'dpg_constraints_normalized.json')
        try:
            with open(dpg_json_path, 'w') as jf:
                json.dump(normalized_constraints, jf, indent=2, sort_keys=True)
            
            # Log to wandb as an artifact
            if wandb_run:
                artifact = wandb.Artifact(f'dpg_constraints_{SAMPLE_ID}', type='dpg_constraints')
                artifact.add_file(dpg_json_path)
                wandb.log_artifact(artifact)
                
        except Exception as exc:
            print(f"WARNING: Failed to save DPG constraints to sample folder: {exc}")
    
    # Generate visualizations if enabled
    if getattr(config.output, 'save_visualizations', True) if hasattr(config, 'output') else True:
        for combination_idx, combination_viz in enumerate(visualizations):
            # dict_non_actionable is now stored directly in combination_viz['label']
            dict_non_actionable = combination_viz['label']
            
            # Per-replication visualizations
            for replication_idx, replication_viz in enumerate(combination_viz['replication']):
                counterfactual = replication_viz['counterfactual']
                cf_model = replication_viz['cf_model']
                
                try:
                    # Create all replication-level visualizations
                    cf_pred_class = int(model.predict(pd.DataFrame([counterfactual]))[0])
                    
                    heatmap_fig = plot_sample_and_counterfactual_heatmap(
                        ORIGINAL_SAMPLE, 
                        ORIGINAL_SAMPLE_PREDICTED_CLASS, 
                        counterfactual, 
                        cf_pred_class,
                        dict_non_actionable
                    )
                    
                    comparison_fig = plot_sample_and_counterfactual_comparison(
                        model,
                        ORIGINAL_SAMPLE,
                        SAMPLE_DATAFRAME,
                        counterfactual,
                        constraints,
                        class_colors_list
                    )
                    
                    fitness_fig = cf_model.plot_fitness() if hasattr(cf_model, 'plot_fitness') else None
                    
                    # Store visualizations
                    replication_viz['visualizations'] = [heatmap_fig, comparison_fig, fitness_fig]
                    
                    # Save replication-level visualizations locally
                    if getattr(config.output, 'save_visualization_images', False):
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        if heatmap_fig:
                            heatmap_path = os.path.join(sample_dir, f'heatmap_combo_{combination_idx}_rep_{replication_idx}.png')
                            heatmap_fig.savefig(heatmap_path, bbox_inches='tight', dpi=150)
                        
                        if comparison_fig:
                            comparison_path = os.path.join(sample_dir, f'comparison_combo_{combination_idx}_rep_{replication_idx}.png')
                            comparison_fig.savefig(comparison_path, bbox_inches='tight', dpi=150)
                        
                        if fitness_fig:
                            fitness_path = os.path.join(sample_dir, f'fitness_combo_{combination_idx}_rep_{replication_idx}.png')
                            fitness_fig.savefig(fitness_path, bbox_inches='tight', dpi=150)
                    
                    # Log to WandB
                    if wandb_run:
                        log_dict = {
                            "viz/sample_id": SAMPLE_ID,
                            "viz/combination": str(combination_viz['label']),
                            "viz/replication": replication_idx,
                        }
                        
                        if heatmap_fig:
                            log_dict["visualizations/heatmap"] = wandb.Image(heatmap_fig)
                        if comparison_fig:
                            log_dict["visualizations/comparison"] = wandb.Image(comparison_fig)
                        if fitness_fig:
                            log_dict["visualizations/fitness_curve"] = wandb.Image(fitness_fig)
                        
                        wandb.log(log_dict)
                    
                    # Generate and log explainer metrics
                    explainer = CounterFactualExplainer(cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
                    
                    explanations = {
                        'Feature Modifications': explainer.explain_feature_modifications(),
                        'Constraints Respect': explainer.check_constraints_respect(),
                        'Stopping Criteria': explainer.explain_stopping_criteria(),
                        'Final Results': explainer.summarize_final_results(),
                    }
                    
                    replication_viz['explanations'] = explanations
                    
                    # Save explanations locally as text file
                    if getattr(config.output, 'save_visualization_images', False):
                        os.makedirs(sample_dir, exist_ok=True)
                        explanation_text = f"""Sample {SAMPLE_ID} - Combination {combination_idx} - Replication {replication_idx}

Feature Modifications
{explanations['Feature Modifications']}

Constraints Respect
{explanations['Constraints Respect']}

Stopping Criteria
{explanations['Stopping Criteria']}

Final Results
{explanations['Final Results']}
"""
                        explanation_path = os.path.join(sample_dir, f'explanation_combo_{combination_idx}_rep_{replication_idx}.txt')
                        with open(explanation_path, 'w') as f:
                            f.write(explanation_text)
                    
                    # Log explanations to WandB as text
                    if wandb_run:
                        explanation_text = f"""## Sample {SAMPLE_ID} - Combination {combination_idx} - Replication {replication_idx}

### Feature Modifications
{explanations['Feature Modifications']}

### Constraints Respect
{explanations['Constraints Respect']}

### Stopping Criteria
{explanations['Stopping Criteria']}

### Final Results
{explanations['Final Results']}
"""
                        wandb.log({
                            "explanations/text": wandb.Html(f"<pre>{explanation_text}</pre>"),
                            "expl/sample_id": SAMPLE_ID,
                            "expl/combination": str(combination_viz['label']),
                            "expl/replication": replication_idx,
                        })
                    
                except Exception as exc:
                    print(f"WARNING: Visualization generation failed for replication {replication_idx}: {exc}")
                    replication_viz['visualizations'] = []
                    replication_viz['explanations'] = {}
            
            # Combination-level visualizations (after all replications)
            try:
                if combination_viz['replication']:
                    counterfactuals_list = [rep['counterfactual'] for rep in combination_viz['replication']]
                    cf_features_df = pd.DataFrame(counterfactuals_list)
                    
                    # Collect all evolution histories for visualization
                    evolution_histories = [rep.get('evolution_history', []) for rep in combination_viz['replication']]
                    
                    # Create combination-level visualizations
                    pairwise_fig = plot_pairwise_with_counterfactual_df(
                        model,
                        FEATURES,
                        LABELS,
                        ORIGINAL_SAMPLE,
                        cf_features_df
                    )
                    
                    pca_fig = plot_pca_with_counterfactuals(
                        model,
                        pd.DataFrame(FEATURES, columns=FEATURE_NAMES),
                        LABELS,
                        ORIGINAL_SAMPLE,
                        cf_features_df,
                        evolution_histories=evolution_histories  # Pass evolution data
                    )
                    
                    combination_viz['pairwise'] = pairwise_fig
                    combination_viz['pca'] = pca_fig
                    
                    # Optionally save images and CSVs locally
                    try:
                        if getattr(config.output, 'save_visualization_images', False):
                            # Ensure sample_dir exists
                            os.makedirs(sample_dir, exist_ok=True)

                            if pairwise_fig:
                                pairwise_path = os.path.join(sample_dir, f'pairwise_combo_{combination_idx}.png')
                                pairwise_fig.savefig(pairwise_path, bbox_inches='tight')

                            if pca_fig:
                                pca_path = os.path.join(sample_dir, f'pca_combo_{combination_idx}.png')
                                pca_fig.savefig(pca_path, bbox_inches='tight')

                                # Also compute and save PCA numeric data (coords & loadings)
                                try:
                                    from sklearn.preprocessing import StandardScaler
                                    from sklearn.decomposition import PCA

                                    FEATURES_ARR = np.array(FEATURES)
                                    FEATURE_NAMES_LOCAL = FEATURE_NAMES
                                    df_features = pd.DataFrame(FEATURES_ARR, columns=FEATURE_NAMES_LOCAL).select_dtypes(include=[np.number])

                                    scaler = StandardScaler()
                                    df_scaled = scaler.fit_transform(df_features)
                                    pca_local = PCA(n_components=2)
                                    pca_local.fit(df_scaled)

                                    # Original sample coords
                                    sample_df_local = pd.DataFrame([ORIGINAL_SAMPLE])[FEATURE_NAMES_LOCAL].select_dtypes(include=[np.number])
                                    sample_scaled = scaler.transform(sample_df_local)
                                    sample_coords = pca_local.transform(sample_scaled)

                                    # Counterfactual coords
                                    cf_list_local = [rep['counterfactual'] for rep in combination_viz['replication']]
                                    cf_df_local = pd.DataFrame(cf_list_local)[FEATURE_NAMES_LOCAL].select_dtypes(include=[np.number])
                                    cf_scaled = scaler.transform(cf_df_local)
                                    cf_coords = pca_local.transform(cf_scaled)

                                    # Save coords CSV
                                    coords_rows = []
                                    coords_rows.append({'type':'original','pc1':float(sample_coords[0,0]),'pc2':float(sample_coords[0,1])})
                                    for i, row in enumerate(cf_coords):
                                        coords_rows.append({'type':f'counterfactual_{i}','pc1':float(row[0]),'pc2':float(row[1])})

                                    coords_df = pd.DataFrame(coords_rows)
                                    coords_df.to_csv(os.path.join(sample_dir, f'pca_coords_combo_{combination_idx}.csv'), index=False)

                                    # Save all generations evolution data
                                    gen_rows = []
                                    gen_rows.append({'replication': 'original', 'generation': 0, 'pc1': float(sample_coords[0,0]), 'pc2': float(sample_coords[0,1])})
                                    
                                    for rep_idx, rep in enumerate(combination_viz['replication']):
                                        evolution_history = rep.get('evolution_history', [])
                                        if evolution_history:
                                            history_df = pd.DataFrame(evolution_history)
                                            history_numeric = history_df[FEATURE_NAMES_LOCAL].select_dtypes(include=[np.number])
                                            history_scaled = scaler.transform(history_numeric)
                                            history_pca = pca_local.transform(history_scaled)
                                            
                                            for gen_idx, coords in enumerate(history_pca):
                                                gen_rows.append({
                                                    'replication': rep_idx,
                                                    'generation': gen_idx + 1,
                                                    'pc1': float(coords[0]),
                                                    'pc2': float(coords[1])
                                                })
                                    
                                    gen_df = pd.DataFrame(gen_rows)
                                    gen_df.to_csv(os.path.join(sample_dir, f'pca_generations_combo_{combination_idx}.csv'), index=False)

                                    # Save feature values for original, all generations, and final counterfactuals
                                    feature_rows = []
                                    # Add original sample
                                    orig_row = {'replication': 'original', 'generation': 0, 'predicted_class': ORIGINAL_SAMPLE_PREDICTED_CLASS}
                                    orig_row.update({f: ORIGINAL_SAMPLE[f] for f in FEATURE_NAMES_LOCAL})
                                    feature_rows.append(orig_row)
                                    
                                    for rep_idx, rep in enumerate(combination_viz['replication']):
                                        evolution_history = rep.get('evolution_history', [])
                                        if evolution_history:
                                            for gen_idx, gen_sample in enumerate(evolution_history):
                                                # Predict class for this generation
                                                gen_sample_df = pd.DataFrame([gen_sample])[FEATURE_NAMES_LOCAL]
                                                gen_pred_class = int(model.predict(gen_sample_df)[0])
                                                
                                                gen_row = {'replication': rep_idx, 'generation': gen_idx + 1, 'predicted_class': gen_pred_class}
                                                gen_row.update({f: gen_sample.get(f, np.nan) for f in FEATURE_NAMES_LOCAL})
                                                feature_rows.append(gen_row)
                                    
                                    feature_df = pd.DataFrame(feature_rows)
                                    feature_df.to_csv(os.path.join(sample_dir, f'feature_values_generations_combo_{combination_idx}.csv'), index=False)

                                    # Create seaborn pairplot for feature evolution visualization
                                    try:
                                        import seaborn as sns
                                        
                                        # Determine which features to include in pairplot
                                        # Use top features by variance or change, limit to 4-6 for readability
                                        numeric_features = [f for f in FEATURE_NAMES_LOCAL if f in feature_df.columns]
                                        
                                        # Calculate variance for each feature across generations
                                        feature_variance = {}
                                        for feat in numeric_features:
                                            if feat in feature_df.columns:
                                                feature_variance[feat] = feature_df[feat].var()
                                        
                                        # Sort by variance and select top features
                                        sorted_by_variance = sorted(feature_variance.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                                        max_pairplot_features = 5  # Limit for readability
                                        pairplot_features = [feat for feat, _ in sorted_by_variance[:max_pairplot_features]]
                                        
                                        if len(pairplot_features) >= 2:
                                            print(f"INFO: Creating seaborn pairplot for top {len(pairplot_features)} features by variance...")
                                            
                                            # Build combined dataframe with dataset samples + evolution points
                                            # 1. Dataset samples (background)
                                            dataset_df = pd.DataFrame(FEATURES, columns=FEATURE_NAMES_LOCAL)
                                            dataset_df['point_type'] = 'Dataset'
                                            dataset_df['predicted_class'] = LABELS.tolist() if hasattr(LABELS, 'tolist') else list(LABELS)
                                            
                                            # 2. Evolution data from feature_df
                                            pairplot_df = feature_df.copy()
                                            
                                            # Create type labels for evolution data
                                            max_gen = feature_df[feature_df['replication'] != 'original']['generation'].max() if len(feature_df[feature_df['replication'] != 'original']) > 0 else 0
                                            
                                            def get_point_type(row):
                                                if row['replication'] == 'original':
                                                    return 'Original'
                                                elif row['generation'] == max_gen:
                                                    return 'Counterfactual'
                                                else:
                                                    return 'Evolution'
                                            
                                            pairplot_df['point_type'] = pairplot_df.apply(get_point_type, axis=1)
                                            
                                            # Combine dataset + evolution data
                                            combined_df = pd.concat([
                                                dataset_df[pairplot_features + ['point_type']],
                                                pairplot_df[pairplot_features + ['point_type']]
                                            ], ignore_index=True)
                                            
                                            # Define color palette matching class colors
                                            orig_class_color = class_colors_list[ORIGINAL_SAMPLE_PREDICTED_CLASS % len(class_colors_list)]
                                            target_class_color = class_colors_list[TARGET_CLASS % len(class_colors_list)]
                                            
                                            # Define full palette and markers mapping
                                            full_palette = {
                                                'Dataset': 'lightgray',
                                                'Original': orig_class_color,
                                                'Evolution': 'gray',
                                                'Counterfactual': target_class_color
                                            }
                                            full_markers = {
                                                'Dataset': 'o',
                                                'Original': 'o',
                                                'Evolution': '.',
                                                'Counterfactual': 's'
                                            }
                                            full_hue_order = ['Dataset', 'Evolution', 'Original', 'Counterfactual']
                                            
                                            # Set style
                                            sns.set_style("whitegrid")
                                            
                                            # Create pairplot with hue_order to control layering
                                            pairplot_data = combined_df.dropna()
                                            
                                            # Determine actual unique point types present in data
                                            actual_point_types = pairplot_data['point_type'].unique().tolist()
                                            
                                            # Filter hue_order, palette, and markers to only include actual values
                                            hue_order = [h for h in full_hue_order if h in actual_point_types]
                                            palette = {k: full_palette[k] for k in hue_order}
                                            markers = [full_markers[k] for k in hue_order]
                                            
                                            g = sns.pairplot(
                                                pairplot_data,
                                                hue='point_type',
                                                hue_order=hue_order,
                                                palette=palette,
                                                markers=markers,
                                                diag_kind='kde',
                                                plot_kws={'alpha': 0.6},
                                                diag_kws={'alpha': 0.5, 'linewidth': 2},
                                                corner=False,
                                                height=2.5,
                                                aspect=1
                                            )
                                            
                                            # Customize marker sizes - make Original and Counterfactual larger
                                            for ax in g.axes.flatten():
                                                if ax is not None:
                                                    for collection in ax.collections:
                                                        # Adjust sizes based on label
                                                        pass  # seaborn handles this internally
                                            
                                            # Customize the plot
                                            g.fig.suptitle(
                                                f'Feature Evolution Pairplot - Sample {SAMPLE_ID}\n'
                                                f'Original (Class {ORIGINAL_SAMPLE_PREDICTED_CLASS}) → Counterfactual (Class {TARGET_CLASS})',
                                                y=1.02, fontsize=12, weight='bold'
                                            )
                                            
                                            # Adjust legend
                                            if g._legend:
                                                g._legend.set_title('Point Type')
                                            
                                            # Save the pairplot
                                            pairplot_path = os.path.join(sample_dir, f'pairplot_combo_{combination_idx}.png')
                                            g.savefig(pairplot_path, bbox_inches='tight', dpi=150)
                                            plt.close(g.fig)
                                            print(f"INFO: Successfully saved seaborn pairplot to {pairplot_path}")
                                        else:
                                            print(f"INFO: Skipping pairplot - not enough features (need at least 2)")
                                    except Exception as exc:
                                        print(f"ERROR: Failed to create seaborn pairplot: {exc}")
                                        traceback.print_exc()

                                    # Create PCA pairplot (one PC per feature, max 5)
                                    try:
                                        import seaborn as sns
                                        from sklearn.decomposition import PCA as PCA_pairplot
                                        from sklearn.preprocessing import StandardScaler as Scaler_pairplot
                                        
                                        # Determine number of PCs (one per feature, max 5)
                                        n_pcs = min(5, len(FEATURE_NAMES_LOCAL))
                                        
                                        if n_pcs >= 2:
                                            print(f"INFO: Creating PCA pairplot with {n_pcs} principal components...")
                                            
                                            # Fit PCA on dataset
                                            dataset_numeric = pd.DataFrame(FEATURES, columns=FEATURE_NAMES_LOCAL).select_dtypes(include=[np.number])
                                            scaler_pp = Scaler_pairplot()
                                            dataset_scaled = scaler_pp.fit_transform(dataset_numeric)
                                            
                                            pca_pp = PCA_pairplot(n_components=n_pcs)
                                            dataset_pcs = pca_pp.fit_transform(dataset_scaled)
                                            
                                            # Create PC column names
                                            pc_names = [f'PC{i+1}' for i in range(n_pcs)]
                                            
                                            # Dataset PCs
                                            dataset_pc_df = pd.DataFrame(dataset_pcs, columns=pc_names)
                                            dataset_pc_df['point_type'] = 'Dataset'
                                            
                                            # Original sample PCs
                                            orig_numeric = pd.DataFrame([ORIGINAL_SAMPLE])[FEATURE_NAMES_LOCAL].select_dtypes(include=[np.number])
                                            orig_scaled = scaler_pp.transform(orig_numeric)
                                            orig_pcs = pca_pp.transform(orig_scaled)
                                            orig_pc_df = pd.DataFrame(orig_pcs, columns=pc_names)
                                            orig_pc_df['point_type'] = 'Original'
                                            
                                            # Evolution and counterfactual PCs
                                            evolution_pc_rows = []
                                            cf_pc_rows = []
                                            
                                            for rep_idx, rep in enumerate(combination_viz['replication']):
                                                evolution_history = rep.get('evolution_history', [])
                                                if evolution_history:
                                                    for gen_idx, gen_sample in enumerate(evolution_history):
                                                        gen_numeric = pd.DataFrame([gen_sample])[FEATURE_NAMES_LOCAL].select_dtypes(include=[np.number])
                                                        gen_scaled = scaler_pp.transform(gen_numeric)
                                                        gen_pcs = pca_pp.transform(gen_scaled)
                                                        
                                                        is_final = (gen_idx == len(evolution_history) - 1)
                                                        pc_row = {pc_names[i]: gen_pcs[0, i] for i in range(n_pcs)}
                                                        pc_row['point_type'] = 'Counterfactual' if is_final else 'Evolution'
                                                        
                                                        if is_final:
                                                            cf_pc_rows.append(pc_row)
                                                        else:
                                                            evolution_pc_rows.append(pc_row)
                                            
                                            evolution_pc_df = pd.DataFrame(evolution_pc_rows) if evolution_pc_rows else pd.DataFrame(columns=pc_names + ['point_type'])
                                            cf_pc_df = pd.DataFrame(cf_pc_rows) if cf_pc_rows else pd.DataFrame(columns=pc_names + ['point_type'])
                                            
                                            # Combine all data
                                            combined_pc_df = pd.concat([dataset_pc_df, evolution_pc_df, orig_pc_df, cf_pc_df], ignore_index=True)
                                            
                                            # Define colors
                                            orig_class_color = class_colors_list[ORIGINAL_SAMPLE_PREDICTED_CLASS % len(class_colors_list)]
                                            target_class_color = class_colors_list[TARGET_CLASS % len(class_colors_list)]
                                            
                                            # Define full palette and markers mapping
                                            full_palette_pc = {
                                                'Dataset': 'lightgray',
                                                'Evolution': 'gray',
                                                'Original': orig_class_color,
                                                'Counterfactual': target_class_color
                                            }
                                            full_markers_pc = {
                                                'Dataset': 'o',
                                                'Evolution': '.',
                                                'Original': 'o',
                                                'Counterfactual': 's'
                                            }
                                            full_hue_order_pc = ['Dataset', 'Evolution', 'Original', 'Counterfactual']
                                            
                                            sns.set_style("whitegrid")
                                            
                                            # Determine actual unique point types present in data
                                            actual_point_types_pc = combined_pc_df['point_type'].unique().tolist()
                                            
                                            # Filter hue_order, palette, and markers to only include actual values
                                            hue_order_pc = [h for h in full_hue_order_pc if h in actual_point_types_pc]
                                            palette_pc = {k: full_palette_pc[k] for k in hue_order_pc}
                                            markers_pc = [full_markers_pc[k] for k in hue_order_pc]
                                            
                                            g_pc = sns.pairplot(
                                                combined_pc_df,
                                                hue='point_type',
                                                hue_order=hue_order_pc,
                                                palette=palette_pc,
                                                markers=markers_pc,
                                                diag_kind='kde',
                                                plot_kws={'alpha': 0.6},
                                                diag_kws={'alpha': 0.5, 'linewidth': 2},
                                                corner=False,
                                                height=2.5,
                                                aspect=1
                                            )
                                            
                                            # Add explained variance to title
                                            explained_var = pca_pp.explained_variance_ratio_
                                            var_str = ', '.join([f'PC{i+1}:{v:.1%}' for i, v in enumerate(explained_var)])
                                            
                                            g_pc.fig.suptitle(
                                                f'PCA Pairplot - Sample {SAMPLE_ID}\n'
                                                f'Original (Class {ORIGINAL_SAMPLE_PREDICTED_CLASS}) → Counterfactual (Class {TARGET_CLASS})\n'
                                                f'Explained Variance: {var_str}',
                                                y=1.04, fontsize=11, weight='bold'
                                            )
                                            
                                            if g_pc._legend:
                                                g_pc._legend.set_title('Point Type')
                                            
                                            # Save the PCA pairplot
                                            pca_pairplot_path = os.path.join(sample_dir, f'pca_pairplot_combo_{combination_idx}.png')
                                            g_pc.savefig(pca_pairplot_path, bbox_inches='tight', dpi=150)
                                            plt.close(g_pc.fig)
                                            print(f"INFO: Successfully saved PCA pairplot to {pca_pairplot_path}")
                                        else:
                                            print(f"INFO: Skipping PCA pairplot - not enough features for multiple PCs")
                                    except Exception as exc:
                                        print(f"ERROR: Failed to create PCA pairplot: {exc}")
                                        traceback.print_exc()

                                    # Calculate feature changes and filter for visualization
                                    # 1. Get final counterfactual from last generation of first replication
                                    final_cf = None
                                    if combination_viz['replication']:
                                        evolution_history = combination_viz['replication'][0].get('evolution_history', [])
                                        if evolution_history:
                                            final_cf = evolution_history[-1]
                                    
                                    # 2. Filter actionable features and calculate changes
                                    actionable_features = [f for f in FEATURE_NAMES_LOCAL 
                                                         if dict_non_actionable.get(f, "none") != "no_change"]
                                    
                                    feature_changes = {}
                                    if final_cf:
                                        for feat in actionable_features:
                                            orig_val = ORIGINAL_SAMPLE.get(feat, 0)
                                            cf_val = final_cf.get(feat, 0)
                                            # Calculate absolute change
                                            feature_changes[feat] = abs(cf_val - orig_val)
                                    
                                    # 3. Sort features by change magnitude and filter non-zero changes
                                    sorted_features = sorted(feature_changes.items(), key=lambda x: x[1], reverse=True)
                                    # Filter out features with zero or negligible change (< 0.001)
                                    sorted_features_nonzero = [(feat, change) for feat, change in sorted_features if change > 0.001]
                                    
                                    print(f"\nINFO: Feature changes ranked (most to least):")
                                    for feat, change in sorted_features_nonzero:
                                        print(f"  {feat}: {change:.4f}")
                                    
                                    # Select top 6 most changed features for pairwise matrix
                                    max_features_for_pairwise = 6
                                    features_to_plot = [feat for feat, _ in sorted_features_nonzero[:max_features_for_pairwise]]
                                    
                                    # For radar chart, use all features with non-zero changes
                                    features_for_radar = [feat for feat, _ in sorted_features_nonzero]
                                    
                                    # Create 4D visualization (pairwise scatter matrix) of feature evolution
                                    try:
                                        if len(features_to_plot) > 0:
                                            print(f"INFO: Creating pairwise feature evolution plot for top {len(features_to_plot)} changed features...")
                                            n_features = len(features_to_plot)
                                            fig_4d, axes = plt.subplots(n_features, n_features, figsize=(14, 14))
                                        
                                            # Use class colors from the main visualization
                                            orig_class_color = class_colors_list[ORIGINAL_SAMPLE_PREDICTED_CLASS % len(class_colors_list)]
                                            target_class_color = class_colors_list[TARGET_CLASS % len(class_colors_list)]
                                            
                                            # Extract DPG constraints for original and target classes
                                            orig_class_key = f"Class {ORIGINAL_SAMPLE_PREDICTED_CLASS}"
                                            target_class_key = f"Class {TARGET_CLASS}"
                                            
                                            orig_constraints = {}
                                            target_constraints = {}
                                            
                                            if orig_class_key in constraints:
                                                for constraint in constraints[orig_class_key]:
                                                    feat = constraint.get('feature')
                                                    if feat:
                                                        orig_constraints[feat] = {
                                                            'min': constraint.get('min'),
                                                            'max': constraint.get('max')
                                                        }
                                            
                                            if target_class_key in constraints:
                                                for constraint in constraints[target_class_key]:
                                                    feat = constraint.get('feature')
                                                    if feat:
                                                        target_constraints[feat] = {
                                                            'min': constraint.get('min'),
                                                            'max': constraint.get('max')
                                                        }
                                            
                                            for i, feat_y in enumerate(features_to_plot):
                                                for j, feat_x in enumerate(features_to_plot):
                                                    ax = axes[i, j] if n_features > 1 else axes
                                                    
                                                    if i == j:
                                                        # Diagonal: show feature name
                                                        ax.text(0.5, 0.5, feat_x, ha='center', va='center', 
                                                               fontsize=10, weight='bold', transform=ax.transAxes)
                                                        ax.set_xlim(0, 1)
                                                        ax.set_ylim(0, 1)
                                                        ax.axis('off')
                                                    else:
                                                        # Off-diagonal: scatter plot
                                                        # Plot original sample with its class color
                                                        ax.scatter(ORIGINAL_SAMPLE[feat_x], ORIGINAL_SAMPLE[feat_y], 
                                                                 marker='o', s=200, c=orig_class_color, edgecolors='black', 
                                                                 linewidths=2, zorder=10, alpha=0.7)
                                                        ax.text(ORIGINAL_SAMPLE[feat_x], ORIGINAL_SAMPLE[feat_y], 'S',
                                                               ha='center', va='center', fontsize=8, color='white', 
                                                               weight='bold', zorder=11)
                                                        
                                                        # Plot evolution for each replication
                                                        for rep_idx, rep in enumerate(combination_viz['replication']):
                                                            evolution_history = rep.get('evolution_history', [])
                                                            if evolution_history:
                                                                # Plot path (use target class color)
                                                                x_vals = [gen_sample.get(feat_x, np.nan) for gen_sample in evolution_history]
                                                                y_vals = [gen_sample.get(feat_y, np.nan) for gen_sample in evolution_history]
                                                                ax.plot(x_vals, y_vals, color=target_class_color, alpha=0.3, linewidth=1, zorder=3)
                                                                
                                                                # Plot generation points colored by predicted class
                                                                for gen_idx, gen_sample in enumerate(evolution_history):
                                                                    x_val = gen_sample.get(feat_x, np.nan)
                                                                    y_val = gen_sample.get(feat_y, np.nan)
                                                                    
                                                                    # Predict class for this generation
                                                                    gen_sample_df = pd.DataFrame([gen_sample])[FEATURE_NAMES_LOCAL]
                                                                    gen_pred_class = int(model.predict(gen_sample_df)[0])
                                                                    gen_color = class_colors_list[gen_pred_class % len(class_colors_list)]
                                                                    
                                                                    # Determine if this is the last generation (final counterfactual)
                                                                    is_final = (gen_idx == len(evolution_history) - 1)
                                                                    marker_size = 120 if is_final else 80
                                                                    alpha_val = 1.0 if is_final else 0.3 + (0.5 * gen_idx / max(1, len(evolution_history) - 1))
                                                                    # Clamp alpha to [0, 1] to avoid floating-point precision errors
                                                                    alpha_val = np.clip(alpha_val, 0.0, 1.0)
                                                                    
                                                                    ax.scatter(x_val, y_val, marker='o', s=marker_size, 
                                                                             c='none', edgecolors=gen_color, linewidths=1.5 if is_final else 1, 
                                                                             alpha=alpha_val, zorder=5)
                                                                    
                                                                    # Add label
                                                                    label = 'C' if is_final else str(gen_idx + 1)
                                                                    ax.text(x_val, y_val, label, ha='center', va='center',
                                                                           fontsize=7 if is_final else 6, color=gen_color, 
                                                                           weight='bold', zorder=6, alpha=alpha_val)
                                                        
                                                        # Now add constraint boundaries with labels (after data is plotted)
                                                        # Plot constraint lines/regions
                                                        # Original class constraints (dashed) - positioned at edges
                                                        if feat_x in orig_constraints:
                                                            x_min = orig_constraints[feat_x].get('min')
                                                            x_max = orig_constraints[feat_x].get('max')
                                                            if x_min is not None:
                                                                ax.axvline(x=x_min, color=orig_class_color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                                                                # Use axis coordinates for consistent positioning
                                                                ax.text(x_min, 0.98, f'C{ORIGINAL_SAMPLE_PREDICTED_CLASS} min={x_min:.2f}', 
                                                                       rotation=90, va='top', ha='right', fontsize=4, 
                                                                       color=orig_class_color, alpha=0.9, transform=ax.get_xaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=orig_class_color, alpha=0.7, linewidth=0.5))
                                                            if x_max is not None:
                                                                ax.axvline(x=x_max, color=orig_class_color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                                                                ax.text(x_max, 0.98, f'C{ORIGINAL_SAMPLE_PREDICTED_CLASS} max={x_max:.2f}', 
                                                                       rotation=90, va='top', ha='left', fontsize=4, 
                                                                       color=orig_class_color, alpha=0.9, transform=ax.get_xaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=orig_class_color, alpha=0.7, linewidth=0.5))
                                                        
                                                        if feat_y in orig_constraints:
                                                            y_min = orig_constraints[feat_y].get('min')
                                                            y_max = orig_constraints[feat_y].get('max')
                                                            if y_min is not None:
                                                                ax.axhline(y=y_min, color=orig_class_color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                                                                ax.text(0.98, y_min, f'C{ORIGINAL_SAMPLE_PREDICTED_CLASS} min={y_min:.2f}', 
                                                                       rotation=0, va='bottom', ha='right', fontsize=4, 
                                                                       color=orig_class_color, alpha=0.9, transform=ax.get_yaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=orig_class_color, alpha=0.7, linewidth=0.5))
                                                            if y_max is not None:
                                                                ax.axhline(y=y_max, color=orig_class_color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                                                                ax.text(0.98, y_max, f'C{ORIGINAL_SAMPLE_PREDICTED_CLASS} max={y_max:.2f}', 
                                                                       rotation=0, va='top', ha='right', fontsize=4, 
                                                                       color=orig_class_color, alpha=0.9, transform=ax.get_yaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=orig_class_color, alpha=0.7, linewidth=0.5))
                                                        
                                                        # Target class constraints (solid) - positioned at opposite edges
                                                        if feat_x in target_constraints:
                                                            x_min = target_constraints[feat_x].get('min')
                                                            x_max = target_constraints[feat_x].get('max')
                                                            if x_min is not None:
                                                                ax.axvline(x=x_min, color=target_class_color, linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
                                                                ax.text(x_min, 0.02, f'C{TARGET_CLASS} min={x_min:.2f}', 
                                                                       rotation=90, va='bottom', ha='right', fontsize=4, 
                                                                       color=target_class_color, alpha=0.9, transform=ax.get_xaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=target_class_color, alpha=0.7, linewidth=0.5))
                                                            if x_max is not None:
                                                                ax.axvline(x=x_max, color=target_class_color, linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
                                                                ax.text(x_max, 0.02, f'C{TARGET_CLASS} max={x_max:.2f}', 
                                                                       rotation=90, va='bottom', ha='left', fontsize=4, 
                                                                       color=target_class_color, alpha=0.9, transform=ax.get_xaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=target_class_color, alpha=0.7, linewidth=0.5))
                                                        
                                                        if feat_y in target_constraints:
                                                            y_min = target_constraints[feat_y].get('min')
                                                            y_max = target_constraints[feat_y].get('max')
                                                            if y_min is not None:
                                                                ax.axhline(y=y_min, color=target_class_color, linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
                                                                ax.text(0.02, y_min, f'C{TARGET_CLASS} min={y_min:.2f}', 
                                                                       rotation=0, va='bottom', ha='left', fontsize=4, 
                                                                       color=target_class_color, alpha=0.9, transform=ax.get_yaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=target_class_color, alpha=0.7, linewidth=0.5))
                                                            if y_max is not None:
                                                                ax.axhline(y=y_max, color=target_class_color, linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
                                                                ax.text(0.02, y_max, f'C{TARGET_CLASS} max={y_max:.2f}', 
                                                                       rotation=0, va='top', ha='left', fontsize=4, 
                                                                       color=target_class_color, alpha=0.9, transform=ax.get_yaxis_transform(),
                                                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=target_class_color, alpha=0.7, linewidth=0.5))
                                                        
                                                        ax.set_xlabel(feat_x if i == n_features - 1 else '', fontsize=8)
                                                        ax.set_ylabel(feat_y if j == 0 else '', fontsize=8)
                                                        ax.tick_params(labelsize=7)
                                        
                                            try:
                                                plt.tight_layout()
                                            except:
                                                pass  # Ignore tight_layout warnings
                                            fig_4d.savefig(os.path.join(sample_dir, f'feature_evolution_4d_combo_{combination_idx}.png'), 
                                                          bbox_inches='tight', dpi=150)
                                            plt.close(fig_4d)
                                            print(f"INFO: Successfully saved pairwise feature evolution plot with {len(features_to_plot)} features")
                                        else:
                                            print(f"INFO: Skipping pairwise plot - no actionable features with changes")
                                    except Exception as exc:
                                        print(f"ERROR: Failed to save pairwise feature evolution plot: {exc}")
                                        traceback.print_exc()

                                    # Create radar chart for features with non-zero changes
                                    try:
                                        if final_cf and len(features_for_radar) > 0:
                                            print(f"INFO: Creating radar chart for {len(features_for_radar)} features with changes...")
                                            
                                            # Extract DPG constraints for radar chart
                                            orig_class_key = f"Class {ORIGINAL_SAMPLE_PREDICTED_CLASS}"
                                            target_class_key = f"Class {TARGET_CLASS}"
                                            
                                            orig_constraints = {}
                                            target_constraints = {}
                                            
                                            if orig_class_key in constraints:
                                                for constraint in constraints[orig_class_key]:
                                                    feat = constraint.get('feature')
                                                    if feat:
                                                        orig_constraints[feat] = {
                                                            'min': constraint.get('min'),
                                                            'max': constraint.get('max')
                                                        }
                                            
                                            if target_class_key in constraints:
                                                for constraint in constraints[target_class_key]:
                                                    feat = constraint.get('feature')
                                                    if feat:
                                                        target_constraints[feat] = {
                                                            'min': constraint.get('min'),
                                                            'max': constraint.get('max')
                                                        }
                                            
                                            # Prepare data for radar chart using filtered features
                                            categories = features_for_radar
                                            original_values = [ORIGINAL_SAMPLE.get(f, 0) for f in categories]
                                            cf_values = [final_cf.get(f, 0) for f in categories]
                                            
                                            # Collect constraint bounds for normalization
                                            constraint_values = []
                                            for f in categories:
                                                if f in orig_constraints:
                                                    if orig_constraints[f]['min'] is not None:
                                                        constraint_values.append(orig_constraints[f]['min'])
                                                    if orig_constraints[f]['max'] is not None:
                                                        constraint_values.append(orig_constraints[f]['max'])
                                                if f in target_constraints:
                                                    if target_constraints[f]['min'] is not None:
                                                        constraint_values.append(target_constraints[f]['min'])
                                                    if target_constraints[f]['max'] is not None:
                                                        constraint_values.append(target_constraints[f]['max'])
                                            
                                            # Normalize values to 0-1 range for better visualization
                                            all_values = original_values + cf_values + constraint_values
                                            min_val = min(all_values) if all_values else 0
                                            max_val = max(all_values) if all_values else 1
                                            value_range = max_val - min_val if max_val > min_val else 1.0
                                            
                                            # Normalize and clip to [0, 1] to avoid floating-point precision errors
                                            original_norm = [np.clip((v - min_val) / value_range, 0.0, 1.0) for v in original_values]
                                            cf_norm = [np.clip((v - min_val) / value_range, 0.0, 1.0) for v in cf_values]
                                            
                                            # Normalize constraint bounds
                                            orig_constraint_min = []
                                            orig_constraint_max = []
                                            target_constraint_min = []
                                            target_constraint_max = []
                                            
                                            for f in categories:
                                                # Original class constraints
                                                if f in orig_constraints:
                                                    if orig_constraints[f]['min'] is not None:
                                                        orig_constraint_min.append(np.clip((orig_constraints[f]['min'] - min_val) / value_range, 0.0, 1.0))
                                                    else:
                                                        orig_constraint_min.append(None)
                                                    if orig_constraints[f]['max'] is not None:
                                                        orig_constraint_max.append(np.clip((orig_constraints[f]['max'] - min_val) / value_range, 0.0, 1.0))
                                                    else:
                                                        orig_constraint_max.append(None)
                                                else:
                                                    orig_constraint_min.append(None)
                                                    orig_constraint_max.append(None)
                                                
                                                # Target class constraints
                                                if f in target_constraints:
                                                    if target_constraints[f]['min'] is not None:
                                                        target_constraint_min.append(np.clip((target_constraints[f]['min'] - min_val) / value_range, 0.0, 1.0))
                                                    else:
                                                        target_constraint_min.append(None)
                                                    if target_constraints[f]['max'] is not None:
                                                        target_constraint_max.append(np.clip((target_constraints[f]['max'] - min_val) / value_range, 0.0, 1.0))
                                                    else:
                                                        target_constraint_max.append(None)
                                                else:
                                                    target_constraint_min.append(None)
                                                    target_constraint_max.append(None)
                                            
                                            # Create radar chart
                                            num_vars = len(categories)
                                            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                                            
                                            # Close the plot by appending first value
                                            original_norm += original_norm[:1]
                                            cf_norm += cf_norm[:1]
                                            angles += angles[:1]
                                            
                                            fig_radar, ax_radar = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
                                            
                                            # Plot original and counterfactual
                                            orig_class_color = class_colors_list[ORIGINAL_SAMPLE_PREDICTED_CLASS % len(class_colors_list)]
                                            target_class_color = class_colors_list[TARGET_CLASS % len(class_colors_list)]
                                            
                                            # Plot constraint bounds as shaded regions
                                            for idx, (angle, orig_min, orig_max, target_min, target_max) in enumerate(zip(
                                                angles[:-1], orig_constraint_min, orig_constraint_max, 
                                                target_constraint_min, target_constraint_max)):
                                                
                                                # Original class constraint region
                                                if orig_min is not None and orig_max is not None:
                                                    ax_radar.plot([angle, angle], [orig_min, orig_max], 
                                                                color=orig_class_color, linewidth=4, alpha=0.4, 
                                                                linestyle='-', zorder=1)
                                                elif orig_min is not None:
                                                    ax_radar.plot([angle, angle], [orig_min, 1.0], 
                                                                color=orig_class_color, linewidth=4, alpha=0.3, 
                                                                linestyle=':', zorder=1)
                                                elif orig_max is not None:
                                                    ax_radar.plot([angle, angle], [0.0, orig_max], 
                                                                color=orig_class_color, linewidth=4, alpha=0.3, 
                                                                linestyle=':', zorder=1)
                                                
                                                # Target class constraint region
                                                if target_min is not None and target_max is not None:
                                                    ax_radar.plot([angle, angle], [target_min, target_max], 
                                                                color=target_class_color, linewidth=4, alpha=0.4, 
                                                                linestyle='-', zorder=1)
                                                elif target_min is not None:
                                                    ax_radar.plot([angle, angle], [target_min, 1.0], 
                                                                color=target_class_color, linewidth=4, alpha=0.3, 
                                                                linestyle=':', zorder=1)
                                                elif target_max is not None:
                                                    ax_radar.plot([angle, angle], [0.0, target_max], 
                                                                color=target_class_color, linewidth=4, alpha=0.3, 
                                                                linestyle=':', zorder=1)
                                            
                                            ax_radar.plot(angles, original_norm, 'o-', linewidth=2, 
                                                        label=f'Original (Class {ORIGINAL_SAMPLE_PREDICTED_CLASS})', 
                                                        color=orig_class_color, markersize=8, zorder=3)
                                            ax_radar.fill(angles, original_norm, alpha=0.15, color=orig_class_color, zorder=2)
                                            
                                            ax_radar.plot(angles, cf_norm, 's-', linewidth=2, 
                                                        label=f'Counterfactual (Class {TARGET_CLASS})', 
                                                        color=target_class_color, markersize=8, zorder=3)
                                            ax_radar.fill(angles, cf_norm, alpha=0.15, color=target_class_color, zorder=2)
                                            
                                            # Add numerical value annotations
                                            for idx, (angle, orig_val, cf_val, orig_norm_val, cf_norm_val, 
                                                     orig_min, orig_max, target_min, target_max) in enumerate(zip(
                                                angles[:-1], original_values, cf_values, original_norm[:-1], cf_norm[:-1],
                                                orig_constraint_min, orig_constraint_max, 
                                                target_constraint_min, target_constraint_max)):
                                                
                                                # Annotate original value
                                                ax_radar.text(angle, orig_norm_val + 0.05, f'{orig_val:.2f}', 
                                                            ha='center', va='bottom', fontsize=6, 
                                                            color=orig_class_color, weight='bold',
                                                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                                                     edgecolor=orig_class_color, alpha=0.7, linewidth=0.5),
                                                            zorder=4)
                                                
                                                # Annotate counterfactual value
                                                ax_radar.text(angle, cf_norm_val - 0.05, f'{cf_val:.2f}', 
                                                            ha='center', va='top', fontsize=6, 
                                                            color=target_class_color, weight='bold',
                                                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                                                     edgecolor=target_class_color, alpha=0.7, linewidth=0.5),
                                                            zorder=4)
                                                
                                                # Annotate original class constraints
                                                if orig_min is not None:
                                                    orig_min_real = orig_constraints[categories[idx]]['min']
                                                    ax_radar.text(angle, orig_min - 0.08, f'min:{orig_min_real:.2f}', 
                                                                ha='center', va='top', fontsize=5, 
                                                                color=orig_class_color, alpha=0.8, style='italic',
                                                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                                                         edgecolor=orig_class_color, alpha=0.5, linewidth=0.3),
                                                                zorder=2)
                                                
                                                if orig_max is not None:
                                                    orig_max_real = orig_constraints[categories[idx]]['max']
                                                    ax_radar.text(angle, orig_max + 0.08, f'max:{orig_max_real:.2f}', 
                                                                ha='center', va='bottom', fontsize=5, 
                                                                color=orig_class_color, alpha=0.8, style='italic',
                                                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                                                         edgecolor=orig_class_color, alpha=0.5, linewidth=0.3),
                                                                zorder=2)
                                                
                                                # Annotate target class constraints (offset slightly to avoid overlap)
                                                if target_min is not None:
                                                    target_min_real = target_constraints[categories[idx]]['min']
                                                    # Offset angle slightly for readability
                                                    offset_angle = angle + 0.05
                                                    ax_radar.text(offset_angle, target_min - 0.08, f'min:{target_min_real:.2f}', 
                                                                ha='left', va='top', fontsize=5, 
                                                                color=target_class_color, alpha=0.8, style='italic',
                                                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                                                         edgecolor=target_class_color, alpha=0.5, linewidth=0.3),
                                                                zorder=2)
                                                
                                                if target_max is not None:
                                                    target_max_real = target_constraints[categories[idx]]['max']
                                                    # Offset angle slightly for readability
                                                    offset_angle = angle + 0.05
                                                    ax_radar.text(offset_angle, target_max + 0.08, f'max:{target_max_real:.2f}', 
                                                                ha='left', va='bottom', fontsize=5, 
                                                                color=target_class_color, alpha=0.8, style='italic',
                                                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                                                         edgecolor=target_class_color, alpha=0.5, linewidth=0.3),
                                                                zorder=2)
                                            
                                            # Add custom legend entries for constraints
                                            from matplotlib.lines import Line2D
                                            custom_lines = [
                                                Line2D([0], [0], color=orig_class_color, linewidth=4, alpha=0.4, label=f'Class {ORIGINAL_SAMPLE_PREDICTED_CLASS} Constraints'),
                                                Line2D([0], [0], color=target_class_color, linewidth=4, alpha=0.4, label=f'Class {TARGET_CLASS} Constraints')
                                            ]
                                            
                                            # Set category labels
                                            ax_radar.set_xticks(angles[:-1])
                                            ax_radar.set_xticklabels(categories, size=8)
                                            
                                            # Set radial limits
                                            ax_radar.set_ylim(0, 1)
                                            ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
                                            ax_radar.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7)
                                            ax_radar.grid(True, linestyle='--', alpha=0.3)
                                            
                                            # Add legend and title
                                            handles, labels = ax_radar.get_legend_handles_labels()
                                            handles.extend(custom_lines)
                                            ax_radar.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
                                            ax_radar.set_title(f'Feature Changes: Original vs Counterfactual\\n(Normalized values with DPG Constraints)', 
                                                             size=14, weight='bold', pad=20)
                                            
                                            try:
                                                plt.tight_layout()
                                            except:
                                                pass  # Ignore tight_layout warnings for complex polar plots
                                            fig_radar.savefig(os.path.join(sample_dir, f'feature_changes_radar_combo_{combination_idx}.png'), 
                                                            bbox_inches='tight', dpi=150)
                                            plt.close(fig_radar)
                                            print(f"INFO: Successfully saved radar chart")
                                        else:
                                            print(f"INFO: Skipping radar chart - no counterfactual or actionable features")
                                    except Exception as exc:
                                        print(f"ERROR: Failed to save radar chart: {exc}")
                                        traceback.print_exc()

                                    # Save loadings
                                    loadings = pca_local.components_.T * (pca_local.explained_variance_**0.5)
                                    loadings_df = pd.DataFrame(loadings, index=FEATURE_NAMES_LOCAL, columns=['pc1_loading','pc2_loading'])
                                    loadings_df.to_csv(os.path.join(sample_dir, f'pca_loadings_combo_{combination_idx}.csv'))

                                except Exception as exc:
                                    print(f"ERROR: Failed to save PCA numeric data: {exc}")
                                    traceback.print_exc()
                    except Exception as exc:
                        print(f"ERROR: Failed saving visualization images: {exc}")
                        traceback.print_exc()

                    # Log to WandB
                    if wandb_run:
                        log_dict = {
                            "viz_combo/sample_id": SAMPLE_ID,
                            "viz_combo/combination": str(combination_viz['label']),
                        }
                        
                        if pairwise_fig:
                            log_dict["visualizations/pairwise"] = wandb.Image(pairwise_fig)
                        if pca_fig:
                            log_dict["visualizations/pca"] = wandb.Image(pca_fig)
                        
                        # Log 4D feature evolution plot
                        feature_4d_path = os.path.join(sample_dir, f'feature_evolution_4d_combo_{combination_idx}.png')
                        if os.path.exists(feature_4d_path):
                            log_dict["visualizations/feature_evolution_4d"] = wandb.Image(feature_4d_path)
                        
                        # Log radar chart
                        radar_path = os.path.join(sample_dir, f'feature_changes_radar_combo_{combination_idx}.png')
                        if os.path.exists(radar_path):
                            log_dict["visualizations/feature_changes_radar"] = wandb.Image(radar_path)
                        
                        # Log seaborn pairplot
                        pairplot_path = os.path.join(sample_dir, f'pairplot_combo_{combination_idx}.png')
                        if os.path.exists(pairplot_path):
                            log_dict["visualizations/pairplot"] = wandb.Image(pairplot_path)
                        
                        # Log PCA pairplot
                        pca_pairplot_path = os.path.join(sample_dir, f'pca_pairplot_combo_{combination_idx}.png')
                        if os.path.exists(pca_pairplot_path):
                            log_dict["visualizations/pca_pairplot"] = wandb.Image(pca_pairplot_path)
                        
                        # Log CSV files as wandb Tables
                        pca_coords_path = os.path.join(sample_dir, f'pca_coords_combo_{combination_idx}.csv')
                        if os.path.exists(pca_coords_path):
                            pca_coords_table = wandb.Table(dataframe=pd.read_csv(pca_coords_path))
                            log_dict["data/pca_coords"] = pca_coords_table
                        
                        pca_generations_path = os.path.join(sample_dir, f'pca_generations_combo_{combination_idx}.csv')
                        if os.path.exists(pca_generations_path):
                            pca_generations_table = wandb.Table(dataframe=pd.read_csv(pca_generations_path))
                            log_dict["data/pca_generations"] = pca_generations_table
                        
                        feature_values_path = os.path.join(sample_dir, f'feature_values_generations_combo_{combination_idx}.csv')
                        if os.path.exists(feature_values_path):
                            feature_values_table = wandb.Table(dataframe=pd.read_csv(feature_values_path))
                            log_dict["data/feature_values_generations"] = feature_values_table
                        
                        pca_loadings_path = os.path.join(sample_dir, f'pca_loadings_combo_{combination_idx}.csv')
                        if os.path.exists(pca_loadings_path):
                            pca_loadings_table = wandb.Table(dataframe=pd.read_csv(pca_loadings_path))
                            log_dict["data/pca_loadings"] = pca_loadings_table
                        
                        wandb.log(log_dict)
                        
            except Exception as exc:
                print(f"WARNING: Combination-level visualization generation failed: {exc}")
                combination_viz['pairwise'] = None
                combination_viz['pca'] = None
    
    # Save visualizations data
    viz_filepath = os.path.join(sample_dir, 'after_viz_generation.pkl')
    with open(viz_filepath, 'wb') as f:
        pickle.dump({
            'sample_id': SAMPLE_ID,
            'visualizations': visualizations,
            'original_sample': ORIGINAL_SAMPLE,
            'features_names': FEATURES_NAMES,
            'target_class': TARGET_CLASS
        }, f)
    
    # Save comprehensive metrics to CSV files
    try:
        # Collect all replication-level metrics
        replication_metrics_list = []
        for combination_idx, combination_viz in enumerate(visualizations):
            for replication_idx, replication_viz in enumerate(combination_viz['replication']):
                if 'metrics' in replication_viz:
                    metrics_row = {
                        'sample_id': SAMPLE_ID,
                        'combination_idx': combination_idx,
                        'combination': str(combination_viz['label']),
                        'replication_idx': replication_idx,
                    }
                    metrics_row.update(replication_viz['metrics'])
                    replication_metrics_list.append(metrics_row)
        
        if replication_metrics_list:
            replication_metrics_df = pd.DataFrame(replication_metrics_list)
            replication_metrics_csv = os.path.join(sample_dir, 'replication_metrics.csv')
            replication_metrics_df.to_csv(replication_metrics_csv, index=False)
            print(f"INFO: Saved replication metrics to {replication_metrics_csv}")
        
        # Collect all combination-level metrics
        combination_metrics_list = []
        for combination_idx, combination_viz in enumerate(visualizations):
            if 'comprehensive_metrics' in combination_viz:
                metrics_row = {
                    'sample_id': SAMPLE_ID,
                    'combination_idx': combination_idx,
                    'combination': str(combination_viz['label']),
                }
                metrics_row.update(combination_viz['comprehensive_metrics'])
                combination_metrics_list.append(metrics_row)
        
        if combination_metrics_list:
            combination_metrics_df = pd.DataFrame(combination_metrics_list)
            combination_metrics_csv = os.path.join(sample_dir, 'combination_metrics.csv')
            combination_metrics_df.to_csv(combination_metrics_csv, index=False)
            print(f"INFO: Saved combination metrics to {combination_metrics_csv}")
            
    except Exception as exc:
        print(f"WARNING: Failed to save metrics CSV files: {exc}")
        import traceback
        traceback.print_exc()
    
    # Use the storage helper to save structured data (as in experiment_generation.py)
    try:
        save_visualizations_data(SAMPLE_ID, visualizations, ORIGINAL_SAMPLE, constraints, FEATURES_NAMES, TARGET_CLASS, configname=configname, output_dir=output_dir)
    except Exception as exc:
        print(f"WARNING: save_visualizations_data failed: {exc}")
    
    # Log artifacts to WandB
    if wandb_run:
        artifact = wandb.Artifact(f"sample_{SAMPLE_ID}_results", type="results")
        artifact.add_file(raw_filepath)
        artifact.add_file(viz_filepath)
        wandb.log_artifact(artifact)
    
    print(f"INFO: Completed sample {SAMPLE_ID}: {valid_counterfactuals}/{total_replications} successful counterfactuals")
    
    return {
        'sample_id': SAMPLE_ID,
        'sample_dir': sample_dir,
        'raw_filepath': raw_filepath,
        'viz_filepath': viz_filepath,
        'success_rate': success_rate,
        'valid_counterfactuals': valid_counterfactuals,
        'total_replications': total_replications,
    }


def run_experiment(config: DictConfig, wandb_run=None):
    """Run full experiment with multiple samples."""
    
    # Set random seed
    np.random.seed(config.experiment_params.seed)
    
    # Load data using flexible loader
    dataset_info = load_dataset(config, repo_root=REPO_ROOT)
    
    FEATURES = dataset_info['features']
    LABELS = dataset_info['labels']
    FEATURE_NAMES = dataset_info['feature_names']
    FEATURES_DF = dataset_info['features_df']
    
    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        FEATURES_DF, LABELS, 
        test_size=config.data.test_size, 
        random_state=config.data.random_state
    )
    
    # Train model
    print("INFO: Training model...")
    if config.model.type == "RandomForestClassifier":
        model_params = {
            'n_estimators': config.model.n_estimators,
            'random_state': config.model.random_state
        }
        # Add optional parameters if they exist in config
        if hasattr(config.model, 'max_depth') and config.model.max_depth is not None:
            model_params['max_depth'] = config.model.max_depth
        
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
    
    # Train model with DataFrame (preserves feature names)
    model.fit(TRAIN_FEATURES, TRAIN_LABELS)
    
    # Log model performance
    train_score = model.score(TRAIN_FEATURES, TRAIN_LABELS)
    test_score = model.score(TEST_FEATURES, TEST_LABELS)
    print(f"INFO: Model trained - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
    
    # Determine counterfactual generation method
    cf_method = getattr(config.counterfactual, 'method', 'dpg').lower()
    print(f"INFO: Using counterfactual generation method: {cf_method.upper()}")
    
    if cf_method == 'dice' and not DICE_AVAILABLE:
        raise RuntimeError("DiCE method selected but dice-ml is not installed. Install with: pip install dice-ml")
    
    if wandb_run:
        wandb.log({
            "model/train_accuracy": train_score,
            "model/test_accuracy": test_score,
            "experiment/cf_method": cf_method,
        })
    
    # Extract constraints (pass numpy array for DPG compatibility)
    print("INFO: Extracting constraints...")
    
    # Get DPG config from counterfactual section if available
    # Restructure to match DPG's expected format: {'dpg': {'default': {...}, 'visualization': {...}}}
    raw_dpg_config = getattr(config.counterfactual, 'config', None)
    dpg_config = None
    if raw_dpg_config is not None:
        # Convert to dict if needed
        if hasattr(raw_dpg_config, 'to_dict'):
            raw_dpg_config = raw_dpg_config.to_dict()
        elif hasattr(raw_dpg_config, '_config'):
            raw_dpg_config = raw_dpg_config._config
        
        # Restructure: flat config -> nested DPG format
        dpg_config = {
            'dpg': {
                'default': {
                    'perc_var': raw_dpg_config.get('perc_var', 0.0000001),
                    'decimal_threshold': raw_dpg_config.get('decimal_threshold', 3),
                    'n_jobs': raw_dpg_config.get('n_jobs', -1),
                },
                'visualization': raw_dpg_config.get('visualization', {}),
            }
        }
    
    constraints = ConstraintParser.extract_constraints_from_dataset(
        model, TRAIN_FEATURES.values, TRAIN_LABELS, FEATURE_NAMES,
        dpg_config=dpg_config
    )

    # --- DPG: send extracted boundaries to WandB under a new 'dpg' section ---
    normalized_constraints = None  # Will store normalized constraints for sample folders
    if wandb_run:
        try:
            # If constraints are empty, log and skip heavy logging
            if not constraints:
                print("INFO: No DPG constraints extracted; skipping detailed WandB logging for DPG")
            else:
                import json

                # Normalize constraints into per-class, per-feature intervals with deterministic ordering
                normalized = ConstraintParser.normalize_constraints(constraints)

                # Store for saving in sample folders
                normalized_constraints = normalized

                # Put normalized constraints and dpg config into config so they appear under the Config tab
                try:
                    dpg_wandb_data = {
                        'constraints': normalized,
                        'config': dpg_config.get('dpg', {}) if dpg_config else {}
                    }
                    try:
                        wandb_run.config['dpg'] = dpg_wandb_data
                    except Exception:
                        wandb_run.config.update({'dpg': dpg_wandb_data})
                except Exception:
                    print("WARNING: Unable to add normalized DPG constraints to wandb config")

                # Add a compact summary into the run summary (best-effort)
                try:
                    class_sizes = {c: len(normalized[c]) for c in normalized}
                    summary_entry = {'num_classes': len(normalized), 'features_per_class': class_sizes}
                    if hasattr(wandb_run, 'summary') and isinstance(wandb_run.summary, dict):
                        wandb_run.summary['dpg'] = summary_entry
                    else:
                        wandb_run.summary.update({'dpg': summary_entry})
                except Exception:
                    print("WARNING: Unable to add DPG summary to wandb summary")

                # Log a tidy table with one row per (class, feature, min, max) for easy visual comparison
                try:
                    table_rows = []
                    for cname in sorted(normalized.keys()):
                        for feat, bounds in normalized[cname].items():
                            minv = bounds['min'] if bounds['min'] is not None else None
                            maxv = bounds['max'] if bounds['max'] is not None else None
                            table_rows.append([cname, feat, minv, maxv])

                    table = wandb.Table(columns=["class", "feature", "min", "max"], data=table_rows)
                    wandb.log({"dpg/constraints_table": table})
                except Exception as exc:
                    print(f"WARNING: Failed to log normalized DPG constraints table to WandB: {exc}")
        except Exception as exc:
            print(f"WARNING: Failed to log DPG constraints to WandB: {exc}")
    # -----------------------------------------------------------------------
    
    # Prepare data dict (renamed from iris_data for generality)
    dataset_data = {
        'features': FEATURES,
        'labels': LABELS,
        'feature_names': FEATURE_NAMES,
        'train_features': TRAIN_FEATURES,
        'train_labels': TRAIN_LABELS,
        'continuous_indices': dataset_info.get('continuous_indices', list(range(len(FEATURE_NAMES)))),
        'categorical_indices': dataset_info.get('categorical_indices', []),
        'variable_indices': dataset_info.get('variable_indices', list(range(len(FEATURE_NAMES)))),
    }
    
    # Determine number of classes for color assignment
    n_classes = len(np.unique(LABELS))
    class_colors_list = ["purple", "green", "orange", "red", "blue", "yellow", "pink", "cyan"][:n_classes]
    
    # Ensure normalized_constraints is computed even without wandb
    if normalized_constraints is None and constraints:
        normalized_constraints = ConstraintParser.normalize_constraints(constraints)
    
    # --- Generate DPG Constraints Overview Visualization ---
    # This visualization shows all constraint boundaries before any counterfactual generation
    if normalized_constraints and getattr(config.output, 'save_visualizations', True):
        try:
            # Create output directory for experiment-level visualizations
            output_dir = getattr(config.output, 'local_dir', 'outputs')
            experiment_name = getattr(config.experiment, 'name', 'experiment')
            experiment_dir = os.path.join(output_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Generate and save the constraints overview
            constraints_fig = plot_dpg_constraints_overview(
                normalized_constraints=normalized_constraints,
                feature_names=FEATURE_NAMES,
                class_colors_list=class_colors_list,
                output_path=os.path.join(experiment_dir, 'dpg_constraints_overview.png'),
                title="DPG Constraints Overview"
            )
            
            # Log to WandB if available
            if wandb_run and constraints_fig:
                wandb.log({"dpg/constraints_overview": wandb.Image(constraints_fig)})
            
            if constraints_fig:
                plt.close(constraints_fig)
                
        except Exception as exc:
            print(f"WARNING: Failed to generate DPG constraints overview: {exc}")
            import traceback as tb
            tb.print_exc()
    # -----------------------------------------------------------------------
    
    # Determine sample indices to process (always from training split)
    # sample_indices is optional - if not specified, randomly select samples using seed
    sample_indices_config = getattr(config.experiment_params, 'sample_indices', None)
    if sample_indices_config is not None:
        sample_indices = sample_indices_config
    else:
        # Select random samples from training split using seeded RNG for reproducibility
        rng = np.random.default_rng(config.experiment_params.seed)
        sample_indices = rng.choice(
            len(TRAIN_FEATURES),
            size=min(config.experiment_params.num_samples, len(TRAIN_FEATURES)),
            replace=False
        ).tolist()
    
    print(f"INFO: Processing {len(sample_indices)} samples: {sample_indices}")
    
    # Process each sample
    results = []
    for sample_idx in sample_indices:
        result = run_single_sample(
            sample_idx,
            config,
            model,
            constraints,
            dataset_data,
            class_colors_list,
            wandb_run,
            normalized_constraints
        )
        results.append(result)
    
    # Log experiment-level summary
    total_success_rate = np.mean([r['success_rate'] for r in results])
    total_valid = sum(r['valid_counterfactuals'] for r in results)
    total_replications = sum(r['total_replications'] for r in results)
    
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Samples processed: {len(results)}")
    print(f"Total valid counterfactuals: {total_valid}/{total_replications}")
    print(f"Overall success rate: {total_success_rate:.2%}")
    print(f"{'='*60}\n")
    
    if wandb_run:
        # Log experiment-level summary (single values for the entire run)
        wandb.run.summary["experiment/total_samples"] = len(results)
        wandb.run.summary["experiment/total_valid_counterfactuals"] = total_valid
        wandb.run.summary["experiment/total_replications"] = total_replications
        wandb.run.summary["experiment/overall_success_rate"] = total_success_rate
        
        # Create summary table
        summary_data = []
        for r in results:
            summary_data.append([
                r['sample_id'],
                r['valid_counterfactuals'],
                r['total_replications'],
                f"{r['success_rate']:.2%}"
            ])
        
        summary_table = wandb.Table(
            columns=["Sample ID", "Valid CFs", "Total Attempts", "Success Rate"],
            data=summary_data
        )
        wandb.log({"experiment/summary_table": summary_table})
    
    # Save experiment-level summary metrics to CSV
    try:
        output_dir = getattr(config.output, 'local_dir', 'outputs')
        experiment_name = getattr(config.experiment, 'name', 'experiment')
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Aggregate all metrics from all samples
        all_replication_metrics = []
        all_combination_metrics = []
        
        for r in results:
            sample_dir = r['sample_dir']
            
            # Load replication metrics
            replication_csv = os.path.join(sample_dir, 'replication_metrics.csv')
            if os.path.exists(replication_csv):
                rep_df = pd.read_csv(replication_csv)
                all_replication_metrics.append(rep_df)
            
            # Load combination metrics
            combination_csv = os.path.join(sample_dir, 'combination_metrics.csv')
            if os.path.exists(combination_csv):
                comb_df = pd.read_csv(combination_csv)
                all_combination_metrics.append(comb_df)
        
        # Save aggregated metrics
        if all_replication_metrics:
            aggregated_rep_df = pd.concat(all_replication_metrics, ignore_index=True)
            aggregated_rep_csv = os.path.join(experiment_dir, 'all_replication_metrics.csv')
            aggregated_rep_df.to_csv(aggregated_rep_csv, index=False)
            print(f"INFO: Saved aggregated replication metrics to {aggregated_rep_csv}")
        
        if all_combination_metrics:
            aggregated_comb_df = pd.concat(all_combination_metrics, ignore_index=True)
            aggregated_comb_csv = os.path.join(experiment_dir, 'all_combination_metrics.csv')
            aggregated_comb_df.to_csv(aggregated_comb_csv, index=False)
            print(f"INFO: Saved aggregated combination metrics to {aggregated_comb_csv}")
            
            # Compute and save summary statistics
            numeric_cols = aggregated_comb_df.select_dtypes(include=[np.number]).columns
            summary_stats = aggregated_comb_df[numeric_cols].describe()
            summary_stats_csv = os.path.join(experiment_dir, 'metrics_summary_statistics.csv')
            summary_stats.to_csv(summary_stats_csv)
            print(f"INFO: Saved summary statistics to {summary_stats_csv}")
        
        # Save experiment configuration
        config_copy_path = os.path.join(experiment_dir, 'experiment_config.yaml')
        with open(config_copy_path, 'w') as f:
            yaml.dump(config.to_dict(), f)
        print(f"INFO: Saved experiment config to {config_copy_path}")
        
    except Exception as exc:
        print(f"WARNING: Failed to save experiment-level metrics: {exc}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual experiments with WandB tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (e.g., configs/iris/dpg/config.yaml)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (e.g., iris, german_credit) - used with --method to auto-construct config path'
    )
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Method name (e.g., dpg, dice) - selects method from unified config or used with --dataset for legacy paths'
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        dest='overrides',
        help='Override config values (e.g., --set counterfactual.population_size=50)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from WandB run ID'
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run WandB in offline mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--comment',
        type=str,
        default=None,
        help='Optional comment to log with this experiment run'
    )
    
    args = parser.parse_args()
    
    # Construct config path from dataset if provided
    if args.dataset:
        if args.config:
            print("WARNING: --config specified along with --dataset. Using --config value.")
        else:
            # First try unified config path
            unified_config_path = f"configs/{args.dataset}/config.yaml"
            legacy_config_path = f"configs/{args.dataset}/{args.method or 'dpg'}/config.yaml" if args.method else None
            
            if os.path.exists(os.path.join(REPO_ROOT, unified_config_path)):
                args.config = unified_config_path
                print(f"INFO: Using unified config: {args.config}")
            elif legacy_config_path and os.path.exists(os.path.join(REPO_ROOT, legacy_config_path)):
                args.config = legacy_config_path
                print(f"INFO: Using legacy config path: {args.config}")
            else:
                # Default to unified config path (will error if not found)
                args.config = unified_config_path
                print(f"INFO: Auto-constructed config path: {args.config}")
    elif not args.config:
        print("ERROR: Either --config or --dataset must be specified")
        print("Usage examples:")
        print("  python scripts/run_experiment.py --dataset german_credit --method dpg")
        print("  python scripts/run_experiment.py --config configs/iris/config.yaml --method dice")
        return None
    
    # Load config with method selection
    print(f"INFO: Loading config from {args.config}")
    config = load_config(args.config, method=args.method)
    
    # Apply overrides
    if args.overrides:
        print(f"INFO: Applying {len(args.overrides)} config overrides")
        config = apply_overrides(config, args.overrides)
    
    # Apply command-line comment
    if args.comment:
        if not hasattr(config, 'experiment'):
            config.experiment = {}
        config.experiment.notes = args.comment
        if args.verbose:
            print(f"INFO: Logging comment: {args.comment}")
    
    # Get output directory for status tracking
    output_dir = pathlib.Path(getattr(config.output, 'local_dir', 'outputs') if hasattr(config, 'output') else 'outputs')
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    
    # Determine dataset and method for status tracking
    dataset_name = args.dataset or getattr(config.data, 'dataset', 'unknown')
    method_name = args.method or getattr(config.counterfactual, 'method', 'unknown')
    
    # Add method to data config for easy filtering in compare_techniques.py
    # This must be done before init_wandb so it's included in the initial config
    # Use __setitem__ to ensure _config dict is updated for to_dict()
    if hasattr(config, 'data'):
        config.data['method'] = method_name
    
    # Write initial "running" status
    start_time = time.time()
    write_status(
        dataset=dataset_name,
        method=method_name,
        status=PersistentStatus.RUNNING,
        output_dir=output_dir,
        pid=os.getpid(),
        start_time=start_time,
    )
    clear_log(dataset_name, method_name, output_dir)
    print(f"INFO: Status tracking enabled for {dataset_name}/{method_name}")
    
    # Initialize WandB
    wandb_run = None
    if WANDB_AVAILABLE:
        print("INFO: Initializing Weights & Biases...")
        wandb_run = init_wandb(config, resume_id=args.resume, offline=args.offline)
        
        # Configure metric definitions for proper visualization
        if wandb_run:
            configure_wandb_metrics()
            print("INFO: Configured WandB metric definitions for improved visualization")
    else:
        print("WARNING: WandB not available. Running without experiment tracking.")
    
    try:
        # Run experiment
        results = run_experiment(config, wandb_run)
        
        # Finish WandB run
        if wandb_run:
            wandb.finish()
        
        # Write "finished" status
        write_status(
            dataset=dataset_name,
            method=method_name,
            status=PersistentStatus.FINISHED,
            output_dir=output_dir,
            pid=os.getpid(),
            start_time=start_time,
            end_time=time.time(),
        )
        print(f"INFO: Experiment {dataset_name}/{method_name} completed successfully")
        
        return results
    
    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        traceback.print_exc()
        
        # Write "error" status
        write_status(
            dataset=dataset_name,
            method=method_name,
            status=PersistentStatus.ERROR,
            output_dir=output_dir,
            pid=os.getpid(),
            start_time=start_time,
            end_time=time.time(),
            error_message=str(e),
        )
        
        if wandb_run:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
