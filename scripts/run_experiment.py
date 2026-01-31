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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    from DPG.dpg import plot_dpg_constraints_overview

    DPG_PKG_AVAILABLE = True
except ImportError:
    DPG_PKG_AVAILABLE = False
    print("Warning: DPG package not available. Install with requirements in DPG/")

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
    print(
        "Warning: CounterFactualMetrics not available. Install scipy and sklearn for comprehensive metrics."
    )
import CounterFactualVisualizer as CounterFactualVisualizer
from CounterFactualVisualizer import (
    plot_sample_and_counterfactual_comparison,
    plot_pairwise_with_counterfactual_df,
    plot_pca_with_counterfactuals_clean,
    plot_sample_and_counterfactual_heatmap,
)

from utils.notebooks.experiment_storage import (
    get_sample_id,
    save_sample_metadata,
    save_visualizations_data,
    _get_sample_dir as get_sample_dir,
)
from utils.dataset_loader import load_dataset
from utils.config_manager import (
    DictConfig,
    load_config,
    apply_overrides,
    build_dict_non_actionable,
)
from utils.replication_runner import (
    run_counterfactual_generation,
)
from utils.wandb_helper import (
    init_wandb,
    configure_wandb_metrics,
)
from utils.experiment_status import (
    PersistentStatus,
    write_status,
    clear_log,
)
from scripts.visualization_helpers import (
    create_pairwise_feature_evolution_plot,
)
from scripts.experiment_helpers import (
    save_replication_and_combination_metrics,
    log_sample_artifacts_to_wandb,
    save_experiment_level_metrics,
    log_experiment_summary_to_wandb,
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
    normalized_constraints=None,
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

    FEATURES = dataset_data["features"]
    LABELS = dataset_data["labels"]
    FEATURE_NAMES = dataset_data["feature_names"]
    TRAIN_FEATURES = dataset_data["train_features"]
    TRAIN_LABELS = dataset_data["train_labels"]

    # Get feature type indices for comprehensive metrics
    CONTINUOUS_INDICES = dataset_data.get(
        "continuous_indices", list(range(len(FEATURE_NAMES)))
    )
    CATEGORICAL_INDICES = dataset_data.get("categorical_indices", [])
    VARIABLE_INDICES = dataset_data.get(
        "variable_indices", list(range(len(FEATURE_NAMES)))
    )

    # Prepare training/test data arrays for comprehensive metrics
    X_TRAIN = (
        TRAIN_FEATURES.values if hasattr(TRAIN_FEATURES, "values") else TRAIN_FEATURES
    )
    X_TEST = FEATURES  # Use full dataset as test for metrics computation

    # Compute ratio of continuous features
    nbr_features = len(FEATURE_NAMES)
    ratio_cont = len(CONTINUOUS_INDICES) / nbr_features if nbr_features > 0 else 1.0

    output_dir = (
        getattr(config.output, "local_dir", "outputs")
        if hasattr(config, "output")
        else "outputs"
    )

    # Get original sample from training split
    original_sample_values = (
        TRAIN_FEATURES.iloc[sample_index].values
        if hasattr(TRAIN_FEATURES, "iloc")
        else TRAIN_FEATURES[sample_index]
    )
    ORIGINAL_SAMPLE = dict(zip(FEATURE_NAMES, map(float, original_sample_values)))
    SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])
    ORIGINAL_SAMPLE_PREDICTED_CLASS = int(model.predict(SAMPLE_DATAFRAME)[0])

    # Prepare CF parameters
    FEATURES_NAMES = list(ORIGINAL_SAMPLE.keys())

    # Build dict_non_actionable from per-feature actionability rules in config
    dict_non_actionable = build_dict_non_actionable(
        config, FEATURES_NAMES, VARIABLE_INDICES
    )
    print(f"INFO: Using per-feature actionability rules from config")

    # Choose target class
    # NOTE: Cannot assume classes are 0, 1, 2... - use actual unique class labels
    unique_classes = np.unique(LABELS)
    n_classes = len(unique_classes)
    # Pick the first class that is different from the original predicted class
    target_candidates = [
        c for c in unique_classes if c != ORIGINAL_SAMPLE_PREDICTED_CLASS
    ]
    TARGET_CLASS = target_candidates[0] if target_candidates else unique_classes[0]

    # Save sample metadata
    SAMPLE_ID = get_sample_id(sample_index)
    configname = getattr(config.experiment, "name", None)
    save_sample_metadata(
        SAMPLE_ID,
        ORIGINAL_SAMPLE,
        ORIGINAL_SAMPLE_PREDICTED_CLASS,
        TARGET_CLASS,
        sample_index,
        configname=configname,
        output_dir=output_dir,
    )

    # Define sample directory early for use throughout the function
    sample_dir = get_sample_dir(SAMPLE_ID, output_dir=output_dir, configname=configname)
    os.makedirs(sample_dir, exist_ok=True)

    print(f"INFO: Processing Sample ID: {SAMPLE_ID} (dataset index: {sample_index})")
    print(
        f"INFO: Original Predicted Class: {ORIGINAL_SAMPLE_PREDICTED_CLASS}, Target Class: {TARGET_CLASS}"
    )

    # Log sample info to WandB as summary (these are single values per sample, not time series)
    if wandb_run:
        wandb.run.summary[f"sample_{SAMPLE_ID}/sample_index"] = sample_index
        wandb.run.summary[f"sample_{SAMPLE_ID}/original_class"] = (
            ORIGINAL_SAMPLE_PREDICTED_CLASS
        )
        wandb.run.summary[f"sample_{SAMPLE_ID}/target_class"] = TARGET_CLASS

    counterfactuals_df_combinations = []
    visualizations = []

    valid_counterfactuals = 0
    requested_counterfactuals = getattr(config.experiment_params, "requested_counterfactuals", 5)

    # Helper to get a descriptive label for logging
    def get_combination_label(dict_non_actionable):
        """Get a descriptive label for logging purposes."""
        # Create a concise summary of actionability rules
        non_none_rules = {f: r for f, r in dict_non_actionable.items() if r != "none"}
        return f"per_feature_rules:{len(non_none_rules)}_constraints"

    # Process single run with per-feature actionability rules from config

    # Compute variable_features for metrics based on dict_non_actionable
    # Features are actionable if their rule is NOT "no_change"
    variable_features_for_metrics = [
        idx
        for idx, feature_name in enumerate(FEATURES_NAMES)
        if dict_non_actionable.get(feature_name, "none") != "no_change"
    ]

    # Log actionability rules
    frozen_features = [
        f for f, rule in dict_non_actionable.items() if rule == "no_change"
    ]
    directional_features = {
        f: rule
        for f, rule in dict_non_actionable.items()
        if rule in ["non_increasing", "non_decreasing"]
    }
    if frozen_features:
        print(
            f"INFO: Freezing {len(frozen_features)} non-actionable features: {frozen_features}"
        )
    if directional_features:
        print(f"INFO: Directional constraints: {directional_features}")

    counterfactuals_df_list = []
    # Store dict_non_actionable directly for visualization
    combination_viz = {
        "label": dict_non_actionable,
        "pairwise": None,
        "pca": None,
        "counterfactuals": [],
    }

    # Prepare training DataFrame with target for DiCE
    # DiCE needs a DataFrame with features + outcome column
    train_df_for_dice = (
        TRAIN_FEATURES.copy()
        if hasattr(TRAIN_FEATURES, "copy")
        else pd.DataFrame(TRAIN_FEATURES, columns=FEATURE_NAMES)
    )
    train_df_for_dice["_target_"] = TRAIN_LABELS

    # Prepare arguments for counterfactual generation
    generation_args = (
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

    print(f"INFO: Generating {requested_counterfactuals} counterfactuals...")
    generation_start_time = time.time()
    result = run_counterfactual_generation(generation_args)
    generation_runtime = time.time() - generation_start_time
    print(f"INFO: Counterfactual generation completed in {generation_runtime:.2f}s")

    # Process results from counterfactual generation
    if result is not None:
        evolution_history = result["evolution_history"]
        best_fitness_list = result["best_fitness_list"]
        average_fitness_list = result["average_fitness_list"]
        std_fitness_list = result.get("std_fitness_list", [])
        result_method = result.get("method", "dpg")
        
        # Get per-CF evolution histories (each CF has its own path from original)
        per_cf_evolution_histories = result.get("per_cf_evolution_histories", None)
        
        # Get generation_found info for each CF (which generation each CF was discovered)
        cf_generation_found_list = result.get("cf_generation_found", None)
        
        # Get all counterfactuals from this generation
        all_counterfactuals = result.get("all_counterfactuals", [])
        
        print(f"DEBUG run_experiment: Processing result, method={result_method}, candidates={len(all_counterfactuals)}, best_fitness={best_fitness_list[0] if best_fitness_list else 'N/A'}")

        if all_counterfactuals:
            # Predict classifications for all counterfactuals at once (for reuse)
            cf_df_for_prediction = pd.DataFrame(all_counterfactuals)
            all_cf_predicted_classes = model.predict(cf_df_for_prediction)
            
            # Process each counterfactual
            for cf_idx, counterfactual in enumerate(all_counterfactuals):
                # Get predicted class for this counterfactual
                cf_predicted_class = int(all_cf_predicted_classes[cf_idx])
                
                # Only count as valid if it achieved the target class
                if cf_predicted_class == TARGET_CLASS:
                    valid_counterfactuals += 1

                # Calculate final best fitness
                best_fitness = best_fitness_list[-1] if best_fitness_list else 0.0
                
                # Get the evolution history specific to this CF
                # Falls back to shared history if per-CF histories not available
                if per_cf_evolution_histories and cf_idx < len(per_cf_evolution_histories):
                    cf_evolution_history = per_cf_evolution_histories[cf_idx]
                else:
                    cf_evolution_history = evolution_history

                # Create cf_model based on method
                if result_method == "dice":
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
                    cf_model.std_fitness_list = []
                    cf_model.evolution_history = cf_evolution_history
                else:
                    # Recreate cf_model with stored DPG parameters (including dual-boundary parameters)
                    # Only pass parameters that exist in result, otherwise use CounterFactualModel defaults
                    dpg_params = {
                        "diversity_weight": result.get("diversity_weight"),
                        "repulsion_weight": result.get("repulsion_weight"),
                        "boundary_weight": result.get("boundary_weight"),
                        "distance_factor": result.get("distance_factor"),
                        "sparsity_factor": result.get("sparsity_factor"),
                        "constraints_factor": result.get("constraints_factor"),
                        "original_escape_weight": result.get("original_escape_weight"),
                        "escape_pressure": result.get("escape_pressure"),
                        "prioritize_non_overlapping": result.get("prioritize_non_overlapping"),
                        "max_bonus_cap": result.get("max_bonus_cap"),
                    }
                    # Filter out None values to use CounterFactualModel defaults
                    dpg_params = {k: v for k, v in dpg_params.items() if v is not None}

                    cf_model = CounterFactualModel(
                        model,
                        constraints,
                        dict_non_actionable=dict_non_actionable,
                        verbose=False,
                        **dpg_params,
                    )
                    # Restore fitness history
                    cf_model.best_fitness_list = best_fitness_list
                    cf_model.average_fitness_list = average_fitness_list
                    cf_model.std_fitness_list = std_fitness_list
                    cf_model.evolution_history = cf_evolution_history

                # Get generation where this CF was found
                generation_found = None
                if cf_generation_found_list and cf_idx < len(cf_generation_found_list):
                    generation_found = cf_generation_found_list[cf_idx]
                
                # Store counterfactual data with evolution history
                cf_viz = {
                    "counterfactual": counterfactual,
                    "all_counterfactuals": all_counterfactuals,
                    "cf_model": cf_model,
                    "evolution_history": cf_evolution_history,
                    "generation_found": generation_found,  # Generation where CF was found
                    "visualizations": [],
                    "explanations": {},
                    "cf_index": cf_idx,
                    "success": True,
                    "best_fitness": best_fitness,
                    "best_fitness_list": cf_model.best_fitness_list
                    if hasattr(cf_model, "best_fitness_list")
                    else [],
                    "method": result_method,
                }
                combination_viz["counterfactuals"].append(cf_viz)

                cf_data = counterfactual.copy()
                cf_data.update({"Rule_" + k: v for k, v in dict_non_actionable.items()})
                cf_data["CF_Index"] = cf_idx + 1
                counterfactuals_df_list.append(cf_data)

                # Get metrics from explainer with comprehensive evaluation
                explainer = CounterFactualExplainer(
                    cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS
                )

                # Enable comprehensive metrics if available and configured
                compute_comprehensive = getattr(
                    config.experiment_params, "compute_comprehensive_metrics", True
                )

                metrics = explainer.get_all_metrics(
                    X_train=X_TRAIN,
                    X_test=X_TEST,
                    variable_features=variable_features_for_metrics,
                    continuous_features=CONTINUOUS_INDICES,
                    categorical_features=CATEGORICAL_INDICES,
                    compute_comprehensive=compute_comprehensive
                    and COMPREHENSIVE_METRICS_AVAILABLE,
                )

                # Store metrics for later aggregation
                cf_viz["metrics"] = metrics
                cf_viz["num_feature_changes"] = metrics.get(
                    "num_feature_changes", 0
                )

                # Compute cf_eval metrics if available (for backwards compatibility)
                cf_eval_metrics = {}
                if CF_EVAL_AVAILABLE:
                    try:
                        x_original = np.array(
                            [ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES]
                        )
                        cf_array = np.array(
                            [[counterfactual[feat] for feat in FEATURES_NAMES]]
                        )
                        continuous_features = CONTINUOUS_INDICES

                        cf_eval_metrics = {
                            "cf_eval/is_valid": int(
                                nbr_valid_cf(
                                    cf_array,
                                    model,
                                    ORIGINAL_SAMPLE_PREDICTED_CLASS,
                                    y_desidered=TARGET_CLASS,
                                )
                            ),
                            "cf_eval/euclidean_distance": float(
                                continuous_distance(
                                    x_original,
                                    cf_array,
                                    continuous_features,
                                    metric="euclidean",
                                )
                            ),
                            "cf_eval/manhattan_distance": float(
                                continuous_distance(
                                    x_original,
                                    cf_array,
                                    continuous_features,
                                    metric="manhattan",
                                )
                            ),
                            "cf_eval/num_changes": float(
                                avg_nbr_changes_per_cf(
                                    x_original, cf_array, continuous_features
                                )
                            ),
                        }
                    except Exception as exc:
                        print(f"WARNING: cf_eval metrics computation failed: {exc}")

                # Log counterfactual metrics to WandB
                if wandb_run:
                    best_fitness = (
                        cf_model.best_fitness_list[-1]
                        if cf_model.best_fitness_list
                        else None
                    )

                    log_data = {
                        "counterfactual/sample_id": SAMPLE_ID,
                        "counterfactual/combination": get_combination_label(
                            dict_non_actionable
                        ),
                        "counterfactual/cf_index": cf_idx,
                        "counterfactual/success": True,
                        "counterfactual/final_fitness": best_fitness,
                        "counterfactual/generations_to_converge": len(
                            cf_model.best_fitness_list
                        ),
                        "counterfactual/num_feature_changes": metrics["num_feature_changes"],
                        "counterfactual/constraints_respected": metrics[
                            "constraints_respected"
                        ],
                        "counterfactual/method": result_method,  # Track which method was used
                    }

                    # Add all metrics from explainer (per-counterfactual level)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, bool)):
                            log_data[f"metrics/per_counterfactual/{key}"] = value

                    # Add cf_eval metrics
                    log_data.update(cf_eval_metrics)

                    wandb.log(log_data)

                    # Log fitness evolution with generation as x-axis (only for first CF to avoid duplicates)
                    if cf_idx == 0 and cf_model.best_fitness_list and cf_model.average_fitness_list:
                        sample_key = f"s{SAMPLE_ID}"
                        for gen, (best, avg) in enumerate(
                            zip(cf_model.best_fitness_list, cf_model.average_fitness_list)
                        ):
                            wandb.log(
                                {
                                    "generation": gen,
                                    f"fitness/best_{sample_key}": best,
                                    f"fitness/avg_{sample_key}": avg,
                                }
                            )

                        # Also log metadata about this fitness series to summary
                        wandb.run.summary[
                            f"fitness_metadata/{sample_key}/sample_id"
                        ] = SAMPLE_ID
                        wandb.run.summary[
                            f"fitness_metadata/{sample_key}/combination"
                        ] = get_combination_label(dict_non_actionable)
                        wandb.run.summary[
                            f"fitness_metadata/{sample_key}/final_fitness"
                        ] = best_fitness
                        wandb.run.summary[
                            f"fitness_metadata/{sample_key}/generations"
                        ] = len(cf_model.best_fitness_list)

                # Save fitness data locally as CSV (only for first CF to avoid duplicates)
                if cf_idx == 0 and getattr(config.output, "save_visualization_images", False):
                    os.makedirs(sample_dir, exist_ok=True)
                    if cf_model.best_fitness_list and cf_model.average_fitness_list:
                        fitness_data = []
                        for gen, (best, avg) in enumerate(
                            zip(
                                cf_model.best_fitness_list,
                                cf_model.average_fitness_list,
                            )
                        ):
                            fitness_data.append(
                                {
                                    "generation": gen,
                                    "best_fitness": best,
                                    "average_fitness": avg,
                                }
                            )
                        fitness_df = pd.DataFrame(fitness_data)
                        fitness_csv_path = os.path.join(
                            sample_dir,
                            "fitness_evolution.csv",
                        )
                        fitness_df.to_csv(fitness_csv_path, index=False)

    if counterfactuals_df_list:
        counterfactuals_df_list = pd.DataFrame(
            counterfactuals_df_list
        )
        counterfactuals_df_combinations.extend(
            counterfactuals_df_list.to_dict("records")
        )

    # Compute sample-level comprehensive metrics
    combination_comprehensive_metrics = {}
    if combination_viz["counterfactuals"] and COMPREHENSIVE_METRICS_AVAILABLE:
        try:
            x_original = np.array(
                [ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES]
            )
            cf_list = [
                np.array([cf_data["counterfactual"][feat] for feat in FEATURES_NAMES])
                for cf_data in combination_viz["counterfactuals"]
            ]
            cf_array = np.array(cf_list)

            # Compute comprehensive metrics for all counterfactuals
            combination_comprehensive_metrics = evaluate_cf_list_comprehensive(
                cf_list=cf_array,
                x=x_original,
                model=model,
                y_val=ORIGINAL_SAMPLE_PREDICTED_CLASS,
                max_nbr_cf=requested_counterfactuals,
                variable_features=variable_features_for_metrics,
                continuous_features_all=CONTINUOUS_INDICES,
                categorical_features_all=CATEGORICAL_INDICES,
                X_train=X_TRAIN,
                X_test=X_TEST,
                ratio_cont=ratio_cont,
                nbr_features=nbr_features,
            )

            # Store for later persistence
            combination_viz["comprehensive_metrics"] = (
                combination_comprehensive_metrics
            )

            # Log to WandB
            if wandb_run:
                combo_log = {
                    "combo/sample_id": SAMPLE_ID,
                    "combo/combination": get_combination_label(
                        dict_non_actionable
                    ),
                }
                for key, value in combination_comprehensive_metrics.items():
                    if isinstance(value, (int, float, bool)) and not (
                        isinstance(value, float)
                        and (np.isnan(value) or np.isinf(value))
                    ):
                        combo_log[f"metrics/combination/{key}"] = value
                wandb.log(combo_log)

        except Exception as exc:
            print(f"WARNING: Sample-level comprehensive metrics failed: {exc}")

    # Compute sample-level cf_eval metrics (for backwards compatibility)
    if combination_viz["counterfactuals"] and CF_EVAL_AVAILABLE and wandb_run:
        try:
            x_original = np.array(
                [ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES]
            )
            cf_list = [
                np.array([cf_data["counterfactual"][feat] for feat in FEATURES_NAMES])
                for cf_data in combination_viz["counterfactuals"]
            ]
            cf_array = np.array(cf_list)
            continuous_features = list(range(len(FEATURES_NAMES)))

            num_valid = int(
                nbr_valid_cf(
                    cf_array,
                    model,
                    ORIGINAL_SAMPLE_PREDICTED_CLASS,
                    y_desidered=TARGET_CLASS,
                )
            )
            pct_valid = float(
                perc_valid_cf(
                    cf_array,
                    model,
                    ORIGINAL_SAMPLE_PREDICTED_CLASS,
                    y_desidered=TARGET_CLASS,
                )
            )
            avg_distance = float(
                continuous_distance(
                    x_original, cf_array, continuous_features, metric="euclidean"
                )
            )
            min_distance = float(
                continuous_distance(
                    x_original,
                    cf_array,
                    continuous_features,
                    metric="euclidean",
                    agg="min",
                )
            )
            max_distance = float(
                continuous_distance(
                    x_original,
                    cf_array,
                    continuous_features,
                    metric="euclidean",
                    agg="max",
                )
            )
            avg_changes = float(
                avg_nbr_changes_per_cf(x_original, cf_array, continuous_features)
            )

            wandb.log(
                {
                    "sample/sample_id": SAMPLE_ID,
                    "sample/combination": get_combination_label(
                        dict_non_actionable
                    ),
                    "sample/num_cfs": len(cf_list),
                    "sample/valid_cfs": num_valid,
                    "sample/validity_pct": pct_valid * 100,
                    "sample/avg_euclidean_distance": avg_distance,
                    "sample/min_euclidean_distance": min_distance,
                    "sample/max_euclidean_distance": max_distance,
                    "sample/avg_num_changes": avg_changes,
                }
            )
        except Exception as exc:
            print(f"WARNING: Sample-level cf_eval metrics failed: {exc}")

    if combination_viz["counterfactuals"]:
        visualizations.append(combination_viz)

    # Calculate sample-level metrics
    success_rate = (
        valid_counterfactuals / requested_counterfactuals if requested_counterfactuals > 0 else 0.0
    )

    if wandb_run:
        # Log sample-level summary statistics (single values per sample)
        wandb.run.summary[f"sample_{SAMPLE_ID}/num_valid_counterfactuals"] = (
            valid_counterfactuals
        )
        wandb.run.summary[f"sample_{SAMPLE_ID}/requested_counterfactuals"] = requested_counterfactuals
        wandb.run.summary[f"sample_{SAMPLE_ID}/success_rate"] = success_rate

        # Create a table of all counterfactuals for this sample (structured view)
        try:
            cf_table_data = []
            for combination_viz in visualizations:
                for cf_viz in combination_viz["counterfactuals"]:
                    cf_table_data.append(
                        [
                            SAMPLE_ID,
                            str(combination_viz["label"]),
                            cf_viz["cf_index"],
                            cf_viz.get("success", False),
                            cf_viz.get("best_fitness", "N/A"),
                            len(cf_viz.get("best_fitness_list", [])),
                            cf_viz.get("num_feature_changes", "N/A"),
                        ]
                    )

            if cf_table_data:
                cf_table = wandb.Table(
                    columns=[
                        "Sample ID",
                        "Combination",
                        "CF Index",
                        "Success",
                        "Final Fitness",
                        "Generations",
                        "Feature Changes",
                    ],
                    data=cf_table_data,
                )
                wandb.log(
                    {f"tables/sample_{SAMPLE_ID}_counterfactuals": cf_table}
                )
        except Exception as exc:
            print(f"WARNING: Failed to create counterfactual table: {exc}")

    # Save raw data
    raw_data = {
        "sample_id": SAMPLE_ID,
        "original_sample": ORIGINAL_SAMPLE,
        "target_class": TARGET_CLASS,
        "features_names": FEATURES_NAMES,
        "visualizations_structure": [],
    }

    for combination_viz in visualizations:
        combo_copy = {"label": combination_viz["label"], "counterfactuals": []}
        for cf_viz in combination_viz["counterfactuals"]:
            best_fitness_list = getattr(
                cf_viz["cf_model"], "best_fitness_list", []
            )
            average_fitness_list = getattr(
                cf_viz["cf_model"], "average_fitness_list", []
            )
            std_fitness_list = getattr(
                cf_viz["cf_model"], "std_fitness_list", []
            )

            cf_copy = {
                "counterfactual": cf_viz["counterfactual"],
                "best_fitness_list": best_fitness_list,
                "average_fitness_list": average_fitness_list,
                "std_fitness_list": std_fitness_list,
            }
            combo_copy["counterfactuals"].append(cf_copy)
        raw_data["visualizations_structure"].append(combo_copy)

    # Save to disk
    raw_filepath = os.path.join(sample_dir, "raw_counterfactuals.pkl")
    with open(raw_filepath, "wb") as f:
        pickle.dump(raw_data, f)

    # Save normalized DPG constraints in sample folder
    if normalized_constraints:
        import json

        dpg_json_path = os.path.join(sample_dir, "dpg_constraints_normalized.json")
        try:
            with open(dpg_json_path, "w") as jf:
                json.dump(normalized_constraints, jf, indent=2, sort_keys=True)

            # Log to wandb as an artifact
            if wandb_run:
                artifact = wandb.Artifact(
                    f"dpg_constraints_{SAMPLE_ID}", type="dpg_constraints"
                )
                artifact.add_file(dpg_json_path)
                wandb.log_artifact(artifact)

        except Exception as exc:
            print(f"WARNING: Failed to save DPG constraints to sample folder: {exc}")

    # Generate visualizations if enabled
    if (
        getattr(config.output, "save_visualizations", True)
        if hasattr(config, "output")
        else True
    ):
        for combination_viz in visualizations:
            # dict_non_actionable is now stored directly in combination_viz['label']
            dict_non_actionable = combination_viz["label"]

            # Per-counterfactual visualizations
            for cf_idx, cf_viz in enumerate(
                combination_viz["counterfactuals"]
            ):
                counterfactual = cf_viz["counterfactual"]
                cf_model = cf_viz["cf_model"]

                try:
                    # Create all counterfactual-level visualizations
                    cf_pred_class = int(
                        model.predict(pd.DataFrame([counterfactual]))[0]
                    )

                    comparison_fig = plot_sample_and_counterfactual_comparison(
                        model,
                        ORIGINAL_SAMPLE,
                        SAMPLE_DATAFRAME,
                        counterfactual,
                        constraints,
                        class_colors_list,
                    )

                    # Generate heatmap visualization for all counterfactuals
                    is_valid = (cf_pred_class == TARGET_CLASS)
                    heatmap_fig = plot_sample_and_counterfactual_heatmap(
                        ORIGINAL_SAMPLE,
                        ORIGINAL_SAMPLE_PREDICTED_CLASS,
                        counterfactual,
                        cf_pred_class,
                        dict_non_actionable,
                        is_valid=is_valid
                    )

                    if getattr(config.output, "save_visualization_images", False):
                        try:
                            from visualization_helpers import (
                                create_feature_evolution_pairplot,
                                create_pca_pairplot,
                            )

                            # Get evolution history for this specific CF
                            # Note: evolution_history is shared across all CFs (single GA run)
                            # But each CF has different final counterfactual values
                            evolution_history = cf_viz.get("evolution_history", [])

                            if evolution_history or counterfactual:
                                # Build feature_df for THIS CF only (original + evolution + this CF)
                                feature_rows = []

                                # Add original sample row (use "replication" column name for compatibility)
                                orig_row = {
                                    "replication": "original",
                                    "generation": 0,
                                    "predicted_class": ORIGINAL_SAMPLE_PREDICTED_CLASS,
                                }
                                orig_row.update({f: ORIGINAL_SAMPLE[f] for f in FEATURE_NAMES})
                                feature_rows.append(orig_row)

                                # Add evolution generation rows (shared across CFs)
                                for gen_idx, gen_sample in enumerate(evolution_history):
                                    gen_sample_df = pd.DataFrame([gen_sample])[FEATURE_NAMES]
                                    gen_pred_class = int(model.predict(gen_sample_df)[0])

                                    gen_row = {
                                        "replication": cf_idx,  # Use cf_idx but column name is "replication"
                                        "generation": gen_idx + 1,
                                        "predicted_class": gen_pred_class,
                                    }
                                    gen_row.update({f: gen_sample.get(f, np.nan) for f in FEATURE_NAMES})
                                    feature_rows.append(gen_row)

                                # Add THIS specific counterfactual as the final generation
                                # (may differ from evolution_history[-1] since we return top-X from final population)
                                max_gen = len(evolution_history) + 1
                                cf_row = {
                                    "replication": cf_idx,
                                    "generation": max_gen,  # Final generation for this CF
                                    "predicted_class": cf_pred_class,
                                }
                                cf_row.update({f: counterfactual.get(f, np.nan) for f in FEATURE_NAMES})
                                feature_rows.append(cf_row)


                                # Create single-CF combination for pca_pairplot
                                # Append this specific counterfactual as the final point
                                # (since evolution_history is shared, we add the actual CF at the end)
                                cf_evolution = evolution_history.copy() if evolution_history else []
                                cf_evolution.append(counterfactual)  # Add this CF's actual values as endpoint
                                

                        except Exception as exc:
                            print(f"WARNING: Failed to create per-CF evolution plots for CF {cf_idx}: {exc}")
                            import traceback
                            traceback.print_exc()

                    # Store visualizations (fitness_fig is now shared across all CFs)
                    cf_viz["visualizations"] = [
                        comparison_fig,
                        heatmap_fig
                    ]

                    # Save counterfactual-level visualizations locally
                    if getattr(config.output, "save_visualization_images", False):
                        os.makedirs(sample_dir, exist_ok=True)

                       
                        if comparison_fig:
                            comparison_path = os.path.join(
                                sample_dir,
                                f"comparison_cf_{cf_idx}.png",
                            )
                            comparison_fig.savefig(
                                comparison_path, bbox_inches="tight", dpi=150
                            )
                        
                        if heatmap_fig:
                            heatmap_path = os.path.join(
                                sample_dir,
                                f"heatmap_cf_{cf_idx}.png",
                            )
                            heatmap_fig.savefig(
                                heatmap_path, bbox_inches="tight", dpi=150
                            )
                            
                    # Generate per-generation comparison images if enabled
                    if getattr(config.output, "save_visualizations_per_generation", False):
                        evolution_history = cf_viz.get("evolution_history", [])
                        if evolution_history:
                            os.makedirs(sample_dir, exist_ok=True)
                            for gen_idx, gen_candidate in enumerate(evolution_history):
                                try:
                                    # Filter out non-feature keys like _fitness
                                    gen_candidate_clean = {k: v for k, v in gen_candidate.items() if k in FEATURE_NAMES}
                                    
                                    # Create comparison image for this generation's best candidate
                                    gen_comparison_fig = plot_sample_and_counterfactual_comparison(
                                        model,
                                        ORIGINAL_SAMPLE,
                                        SAMPLE_DATAFRAME,
                                        gen_candidate_clean,
                                        constraints,
                                        class_colors_list,
                                        generation=gen_idx,  # Pass generation number for title
                                    )
                                    
                                    if gen_comparison_fig:
                                        gen_comparison_path = os.path.join(
                                            sample_dir,
                                            f"comparison_gen_{gen_idx}_cf_{cf_idx}.png",
                                        )
                                        gen_comparison_fig.savefig(
                                            gen_comparison_path, bbox_inches="tight", dpi=150
                                        )
                                        plt.close(gen_comparison_fig)
                                        
                                except Exception as exc:
                                    print(f"WARNING: Failed to create comparison for generation {gen_idx}, CF {cf_idx}: {exc}")

                    # Log per-CF visualizations to WandB (fitness_curve already logged before loop)
                    if wandb_run:
                        log_dict = {
                            "viz/sample_id": SAMPLE_ID,
                            "viz/combination": str(combination_viz["label"]),
                            "viz/cf_index": cf_idx,
                        }
                        if comparison_fig:
                            log_dict["visualizations/comparison"] = wandb.Image(
                                comparison_fig
                            )
                        
                        if heatmap_fig:
                            log_dict["visualizations/heatmap"] = wandb.Image(
                                heatmap_fig
                            )
                            
                        # Log per-generation comparison images if enabled
                        if getattr(config.output, "save_visualizations_per_generation", False):
                            evolution_history = cf_viz.get("evolution_history", [])
                            if evolution_history:
                                for gen_idx in range(len(evolution_history)):
                                    gen_comparison_path = os.path.join(
                                        sample_dir,
                                        f"comparison_gen_{gen_idx}_cf_{cf_idx}.png",
                                    )
                                    if os.path.exists(gen_comparison_path):
                                        wandb.log({
                                            "viz_gen/sample_id": SAMPLE_ID,
                                            "viz_gen/cf_index": cf_idx,
                                            "viz_gen/generation": gen_idx,
                                            "visualizations/comparison_generation": wandb.Image(gen_comparison_path),
                                        })

                        wandb.log(log_dict)

                    # Generate and log explainer metrics
                    explainer = CounterFactualExplainer(
                        cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS
                    )

                    explanations = {
                        "Feature Modifications": explainer.explain_feature_modifications(),
                        "Constraints Respect": explainer.check_constraints_respect(),
                        "Stopping Criteria": explainer.explain_stopping_criteria(),
                        "Final Results": explainer.summarize_final_results(),
                    }

                    cf_viz["explanations"] = explanations

                    # Save explanations locally as text file
                    if getattr(config.output, "save_visualization_images", False):
                        os.makedirs(sample_dir, exist_ok=True)
                        explanation_text = f"""Sample {SAMPLE_ID} - Counterfactual {cf_idx}

Feature Modifications
{explanations["Feature Modifications"]}

Constraints Respect
{explanations["Constraints Respect"]}

Stopping Criteria
{explanations["Stopping Criteria"]}

Final Results
{explanations["Final Results"]}
"""
                        explanation_path = os.path.join(
                            sample_dir,
                            f"explanation_cf_{cf_idx}.txt",
                        )
                        with open(explanation_path, "w") as f:
                            f.write(explanation_text)

                    # Log explanations to WandB as text
                    if wandb_run:
                        explanation_text = f"""## Sample {SAMPLE_ID} - Counterfactual {cf_idx}

### Feature Modifications
{explanations["Feature Modifications"]}

### Constraints Respect
{explanations["Constraints Respect"]}

### Stopping Criteria
{explanations["Stopping Criteria"]}

### Final Results
{explanations["Final Results"]}
"""
                        wandb.log(
                            {
                                "explanations/text": wandb.Html(
                                    f"<pre>{explanation_text}</pre>"
                                ),
                                "expl/sample_id": SAMPLE_ID,
                                "expl/combination": str(combination_viz["label"]),
                                "expl/cf_index": cf_idx,
                            }
                        )

                except Exception as exc:
                    print(
                        f"WARNING: Visualization generation failed for counterfactual {cf_idx}: {exc}"
                    )
                    cf_viz["visualizations"] = []
                    cf_viz["explanations"] = {}

            try:
                if combination_viz["counterfactuals"]:
                    counterfactuals_list = [
                        cf_data["counterfactual"] for cf_data in combination_viz["counterfactuals"]
                    ]
                    cf_features_df = pd.DataFrame(counterfactuals_list)
                    
                    # Predict all counterfactual classes once (for reuse in both plot functions)
                    cf_predicted_classes = model.predict(cf_features_df)

                    # Collect all evolution histories for visualization
                    evolution_histories = [
                        cf_data.get("evolution_history", [])
                        for cf_data in combination_viz["counterfactuals"]
                    ]
                    
                    # Collect generation_found for each CF
                    cf_generations_found = [
                        cf_data.get("generation_found", None)
                        for cf_data in combination_viz["counterfactuals"]
                    ]
                    
                    # Generate clean PCA visualization (without generation history)
                    pca_clean_fig = plot_pca_with_counterfactuals_clean(
                        model,
                        pd.DataFrame(FEATURES, columns=FEATURE_NAMES),
                        LABELS,
                        ORIGINAL_SAMPLE,
                        cf_features_df,
                        cf_predicted_classes=cf_predicted_classes,
                    )
                    combination_viz["pca_clean"] = pca_clean_fig

                    # Optionally save PCA numeric data and other combination-level CSVs locally
                    try:
                        if getattr(config.output, "save_visualization_images", False):
                            # Ensure sample_dir exists
                            os.makedirs(sample_dir, exist_ok=True)

                            # Save the clean PCA figure (without generation history)
                            if pca_clean_fig:
                                pca_clean_path = os.path.join(sample_dir, "pca_clean.png")
                                pca_clean_fig.savefig(pca_clean_path, bbox_inches="tight", dpi=150)

                            # Also compute and save PCA numeric data (coords & loadings)
                            try:
                                from sklearn.preprocessing import StandardScaler
                                from sklearn.decomposition import PCA

                                FEATURES_ARR = np.array(FEATURES)
                                FEATURE_NAMES_LOCAL = FEATURE_NAMES
                                df_features = pd.DataFrame(
                                    FEATURES_ARR, columns=FEATURE_NAMES_LOCAL
                                ).select_dtypes(include=[np.number])

                                scaler = StandardScaler()
                                df_scaled = scaler.fit_transform(df_features)
                                pca_local = PCA(n_components=2)
                                pca_local.fit(df_scaled)

                                # Original sample coords
                                sample_df_local = pd.DataFrame([ORIGINAL_SAMPLE])[
                                    FEATURE_NAMES_LOCAL
                                ].select_dtypes(include=[np.number])
                                sample_scaled = scaler.transform(sample_df_local)
                                sample_coords = pca_local.transform(sample_scaled)

                                # Counterfactual coords
                                cf_list_local = [
                                    cf_data["counterfactual"]
                                    for cf_data in combination_viz["counterfactuals"]
                                ]
                                cf_df_local = pd.DataFrame(cf_list_local)[
                                    FEATURE_NAMES_LOCAL
                                ].select_dtypes(include=[np.number])
                                cf_scaled = scaler.transform(cf_df_local)
                                cf_coords = pca_local.transform(cf_scaled)

                                # Save coords CSV
                                coords_rows = []
                                coords_rows.append(
                                    {
                                        "type": "original",
                                        "pc1": float(sample_coords[0, 0]),
                                        "pc2": float(sample_coords[0, 1]),
                                    }
                                )
                                for i, row in enumerate(cf_coords):
                                    coords_rows.append(
                                        {
                                            "type": f"counterfactual_{i}",
                                            "pc1": float(row[0]),
                                            "pc2": float(row[1]),
                                        }
                                    )

                                coords_df = pd.DataFrame(coords_rows)
                                coords_df.to_csv(
                                    os.path.join(
                                        sample_dir,
                                        "pca_coords.csv",
                                    ),
                                    index=False,
                                )

                                # Save all generations evolution data
                                gen_rows = []
                                gen_rows.append(
                                    {
                                        "cf_index": "original",
                                        "generation": 0,
                                        "pc1": float(sample_coords[0, 0]),
                                        "pc2": float(sample_coords[0, 1]),
                                    }
                                )

                                for cf_idx, cf_data in enumerate(
                                    combination_viz["counterfactuals"]
                                ):
                                    evolution_history = cf_data.get(
                                        "evolution_history", []
                                    )
                                    if evolution_history:
                                        history_df = pd.DataFrame(evolution_history)
                                        history_numeric = history_df[
                                            FEATURE_NAMES_LOCAL
                                        ].select_dtypes(include=[np.number])
                                        history_scaled = scaler.transform(
                                            history_numeric
                                        )
                                        history_pca = pca_local.transform(
                                            history_scaled
                                        )

                                        for gen_idx, coords in enumerate(
                                            history_pca
                                        ):
                                            gen_rows.append(
                                                {
                                                    "cf_index": cf_idx,
                                                    "generation": gen_idx + 1,
                                                    "pc1": float(coords[0]),
                                                    "pc2": float(coords[1]),
                                                }
                                            )

                                gen_df = pd.DataFrame(gen_rows)
                                gen_df.to_csv(
                                    os.path.join(
                                        sample_dir,
                                        "pca_generations.csv",
                                    ),
                                    index=False,
                                )

                                # Save feature values for original, all generations, and final counterfactuals
                                feature_rows = []
                                # Add original sample
                                orig_row = {
                                    "cf_index": "original",
                                    "generation": 0,
                                    "predicted_class": ORIGINAL_SAMPLE_PREDICTED_CLASS,
                                }
                                orig_row.update(
                                    {
                                        f: ORIGINAL_SAMPLE[f]
                                        for f in FEATURE_NAMES_LOCAL
                                    }
                                )
                                feature_rows.append(orig_row)

                                for cf_idx, cf_data in enumerate(
                                    combination_viz["counterfactuals"]
                                ):
                                    evolution_history = cf_data.get(
                                        "evolution_history", []
                                    )
                                    if evolution_history:
                                        for gen_idx, gen_sample in enumerate(
                                            evolution_history
                                        ):
                                            # Predict class for this generation
                                            gen_sample_df = pd.DataFrame(
                                                [gen_sample]
                                            )[FEATURE_NAMES_LOCAL]
                                            gen_pred_class = int(
                                                model.predict(gen_sample_df)[0]
                                            )

                                            gen_row = {
                                                "cf_index": cf_idx,
                                                "generation": gen_idx + 1,
                                                "predicted_class": gen_pred_class,
                                            }
                                            gen_row.update(
                                                {
                                                    f: gen_sample.get(f, np.nan)
                                                    for f in FEATURE_NAMES_LOCAL
                                                }
                                            )
                                            feature_rows.append(gen_row)

                                feature_df = pd.DataFrame(feature_rows)
                                feature_df.to_csv(
                                    os.path.join(
                                        sample_dir,
                                        "feature_values_generations.csv",
                                    ),
                                    index=False,
                                )

                                # Note: pairplot and pca_pairplot are now generated per-CF
                                # (moved to the per-CF loop)

                                # Calculate feature changes for pairwise evolution plot
                                # Get final counterfactual from last generation of first counterfactual
                                final_cf = None
                                if combination_viz["counterfactuals"]:
                                    evolution_history = combination_viz[
                                        "counterfactuals"
                                    ][0].get("evolution_history", [])
                                    if evolution_history:
                                        final_cf = evolution_history[-1]

                                # Filter actionable features and calculate changes
                                actionable_features = [
                                    f
                                    for f in FEATURE_NAMES_LOCAL
                                    if dict_non_actionable.get(f, "none")
                                    != "no_change"
                                ]

                                feature_changes = {}
                                if final_cf:
                                    for feat in actionable_features:
                                        orig_val = ORIGINAL_SAMPLE.get(feat, 0)
                                        cf_val = final_cf.get(feat, 0)
                                        # Calculate absolute change
                                        feature_changes[feat] = abs(
                                            cf_val - orig_val
                                        )

                                # Sort features by change magnitude and filter non-zero changes
                                sorted_features = sorted(
                                    feature_changes.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                                # Filter out features with zero or negligible change (< 0.001)
                                sorted_features_nonzero = [
                                    (feat, change)
                                    for feat, change in sorted_features
                                    if change > 0.001
                                ]

                                print(
                                    f"\nINFO: Feature changes ranked (most to least):"
                                )
                                for feat, change in sorted_features_nonzero:
                                    print(f"  {feat}: {change:.4f}")

                                # Save loadings
                                loadings = pca_local.components_.T * (
                                    pca_local.explained_variance_**0.5
                                )
                                loadings_df = pd.DataFrame(
                                    loadings,
                                    index=FEATURE_NAMES_LOCAL,
                                    columns=["pc1_loading", "pc2_loading"],
                                )
                                loadings_df.to_csv(
                                    os.path.join(
                                        sample_dir,
                                        "pca_loadings.csv",
                                    )
                                )

                            except Exception as exc:
                                print(
                                    f"ERROR: Failed to save PCA numeric data: {exc}"
                                )
                                traceback.print_exc()
                    except Exception as exc:
                        print(f"ERROR: Failed saving visualization images: {exc}")
                        traceback.print_exc()

                    # Log to WandB
                    if wandb_run:
                        log_dict = {
                            "viz_combo/sample_id": SAMPLE_ID,
                            "viz_combo/combination": str(combination_viz["label"]),
                        }

                        # Log clean PCA visualization (without generation history)
                        if combination_viz.get("pca_clean"):
                            log_dict["visualizations/pca_clean"] = wandb.Image(combination_viz["pca_clean"])

                        # Log CSV files as wandb Tables
                        pca_coords_path = os.path.join(
                            sample_dir, "pca_coords.csv"
                        )
                        if os.path.exists(pca_coords_path):
                            pca_coords_table = wandb.Table(
                                dataframe=pd.read_csv(pca_coords_path)
                            )
                            log_dict["data/pca_coords"] = pca_coords_table

                        feature_values_path = os.path.join(
                            sample_dir,
                            "feature_values_generations.csv",
                        )
                        if os.path.exists(feature_values_path):
                            feature_values_table = wandb.Table(
                                dataframe=pd.read_csv(feature_values_path)
                            )
                            log_dict["data/feature_values_generations"] = (
                                feature_values_table
                            )

                        pca_loadings_path = os.path.join(
                            sample_dir, "pca_loadings.csv"
                        )
                        if os.path.exists(pca_loadings_path):
                            pca_loadings_table = wandb.Table(
                                dataframe=pd.read_csv(pca_loadings_path)
                            )
                            log_dict["data/pca_loadings"] = pca_loadings_table

                        wandb.log(log_dict)

            except Exception as exc:
                print(
                    f"WARNING: Combination-level data processing failed: {exc}"
                )
                import traceback
                traceback.print_exc()

    # Save visualizations data
    viz_filepath = os.path.join(sample_dir, "after_viz_generation.pkl")
    with open(viz_filepath, "wb") as f:
        pickle.dump(
            {
                "sample_id": SAMPLE_ID,
                "visualizations": visualizations,
                "original_sample": ORIGINAL_SAMPLE,
                "features_names": FEATURES_NAMES,
                "target_class": TARGET_CLASS,
                "restrictions": dict_non_actionable,
            },
            f,
        )

    # Save comprehensive metrics to CSV files using helper
    save_replication_and_combination_metrics(
        sample_id=SAMPLE_ID,
        sample_dir=sample_dir,
        visualizations=visualizations
    )

    # Use the storage helper to save structured data (as in experiment_generation.py)
    try:
        save_visualizations_data(
            SAMPLE_ID,
            visualizations,
            ORIGINAL_SAMPLE,
            constraints,
            FEATURES_NAMES,
            TARGET_CLASS,
            configname=configname,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"WARNING: save_visualizations_data failed: {exc}")

    # Log artifacts to WandB
    log_sample_artifacts_to_wandb(wandb_run, SAMPLE_ID, raw_filepath, viz_filepath)

    # Log sample and final counterfactuals to WandB for export_comparison_results.py
    # These are stored in config/summary for easy retrieval by other scripts
    if wandb_run:
        try:
            import json
            
            # Store sample as a list of floats (for JSON compatibility)
            sample_values = [float(ORIGINAL_SAMPLE[f]) for f in FEATURES_NAMES]
            wandb.run.config["sample"] = sample_values
            wandb.run.config["sample_class"] = int(ORIGINAL_SAMPLE_PREDICTED_CLASS)
            wandb.run.config["target_class"] = int(TARGET_CLASS)
            wandb.run.config["feature_names"] = FEATURES_NAMES
            wandb.run.config["restrictions"] = dict_non_actionable
            
            # Collect all final counterfactuals as list of lists (for JSON compatibility)
            final_cfs = []
            for combination_viz in visualizations:
                for cf_data in combination_viz.get("counterfactuals", []):
                    cf = cf_data.get("counterfactual", {})
                    if cf:
                        cf_values = [float(cf.get(f, 0.0)) for f in FEATURES_NAMES]
                        final_cfs.append(cf_values)
            
            # Store in summary for easy retrieval
            wandb.run.summary["sample"] = sample_values
            wandb.run.summary["original_class"] = int(ORIGINAL_SAMPLE_PREDICTED_CLASS)
            wandb.run.summary["final_counterfactuals"] = final_cfs
            
            print(f"INFO: Logged sample and {len(final_cfs)} counterfactuals to WandB config/summary")
            
        except Exception as exc:
            print(f"WARNING: Failed to log sample/counterfactuals to WandB: {exc}")

    print(
        f"INFO: Completed sample {SAMPLE_ID}: {valid_counterfactuals}/{requested_counterfactuals} successful counterfactuals"
    )

    return {
        "sample_id": SAMPLE_ID,
        "sample_dir": sample_dir,
        "raw_filepath": raw_filepath,
        "viz_filepath": viz_filepath,
        "success_rate": success_rate,
        "valid_counterfactuals": valid_counterfactuals,
        "requested_counterfactuals": requested_counterfactuals,
        "original_class": ORIGINAL_SAMPLE_PREDICTED_CLASS,
        "target_class": TARGET_CLASS,
        "generation_runtime": generation_runtime,
    }


def run_experiment(config: DictConfig, wandb_run=None):
    """Run full experiment with multiple samples."""

    # Set random seed
    np.random.seed(config.experiment_params.seed)

    # Load data using flexible loader
    dataset_info = load_dataset(config, repo_root=REPO_ROOT)

    FEATURES = dataset_info["features"]
    LABELS = dataset_info["labels"]
    FEATURE_NAMES = dataset_info["feature_names"]
    FEATURES_DF = dataset_info["features_df"]

    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        FEATURES_DF,
        LABELS,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    # Train model
    print("INFO: Training model...")
    if config.model.type == "RandomForestClassifier":
        # Extract all model parameters from config (exclude 'type' which is used for model selection)
        model_config = (
            config.model.to_dict()
            if hasattr(config.model, "to_dict")
            else dict(config.model)
        )
        model_params = {
            k: v for k, v in model_config.items() if k != "type" and v is not None
        }

        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    # Train model with DataFrame (preserves feature names)
    model.fit(TRAIN_FEATURES, TRAIN_LABELS)

    # Log model performance
    train_score = model.score(TRAIN_FEATURES, TRAIN_LABELS)
    test_score = model.score(TEST_FEATURES, TEST_LABELS)
    print(
        f"INFO: Model trained - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}"
    )

    # Determine counterfactual generation method
    cf_method = getattr(config.counterfactual, "method", "dpg").lower()
    print(f"INFO: Using counterfactual generation method: {cf_method.upper()}")

    if cf_method == "dice" and not DICE_AVAILABLE:
        raise RuntimeError(
            "DiCE method selected but dice-ml is not installed. Install with: pip install dice-ml"
        )

    if wandb_run:
        wandb.log(
            {
                "model/train_accuracy": train_score,
                "model/test_accuracy": test_score,
                "experiment/cf_method": cf_method,
            }
        )

    # Extract constraints (pass numpy array for DPG compatibility)
    print("INFO: Extracting constraints...")

    # Get DPG config from counterfactual section if available
    # Restructure to match DPG's expected format: {'dpg': {'default': {...}, 'visualization': {...}}}
    raw_dpg_config = getattr(config.counterfactual, "config", None)
    dpg_config = None
    if raw_dpg_config is not None:
        # Convert to dict if needed
        if hasattr(raw_dpg_config, "to_dict"):
            raw_dpg_config = raw_dpg_config.to_dict()
        elif hasattr(raw_dpg_config, "_config"):
            raw_dpg_config = raw_dpg_config._config

        # Restructure: flat config -> nested DPG format
        dpg_config = {
            "dpg": {
                "default": {
                    "perc_var": raw_dpg_config.get("perc_var", 0.0000001),
                    "decimal_threshold": raw_dpg_config.get("decimal_threshold", 3),
                    "n_jobs": raw_dpg_config.get("n_jobs", -1),
                },
                "visualization": raw_dpg_config.get("visualization", {}),
            }
        }

    # Time boundary extraction (required for DPG, but only used for viz constraints in DICE)
    boundary_extraction_start = time.perf_counter()
    dpg_result = ConstraintParser.extract_constraints_from_dataset(
        model, TRAIN_FEATURES.values, TRAIN_LABELS, FEATURE_NAMES, dpg_config=dpg_config
    )
    boundary_extraction_time = time.perf_counter() - boundary_extraction_start
    constraints = dpg_result["constraints"]
    communities = dpg_result.get("communities", [])
    print(f"INFO: Boundary extraction completed in {boundary_extraction_time:.3f}s")

    # --- DPG: send extracted boundaries to WandB under a new 'dpg' section ---
    normalized_constraints = (
        None  # Will store normalized constraints for sample folders
    )
    if wandb_run:
        try:
            # If constraints are empty, log and skip heavy logging
            if not constraints:
                print(
                    "INFO: No DPG constraints extracted; skipping detailed WandB logging for DPG"
                )
            else:
                import json

                # Normalize constraints into per-class, per-feature intervals with deterministic ordering
                normalized = ConstraintParser.normalize_constraints(constraints)

                # Store for saving in sample folders
                normalized_constraints = normalized

                # Put normalized constraints, communities, and dpg config into config so they appear under the Config tab
                try:
                    dpg_wandb_data = {
                        "constraints": normalized,
                        "config": dpg_config.get("dpg", {}) if dpg_config else {},
                        "communities": communities,
                    }
                    try:
                        wandb_run.config["dpg"] = dpg_wandb_data
                    except Exception:
                        wandb_run.config.update({"dpg": dpg_wandb_data})
                except Exception:
                    print(
                        "WARNING: Unable to add normalized DPG constraints to wandb config"
                    )

                # Add a compact summary into the run summary (best-effort)
                try:
                    class_sizes = {c: len(normalized[c]) for c in normalized}
                    summary_entry = {
                        "num_classes": len(normalized),
                        "features_per_class": class_sizes,
                    }
                    if hasattr(wandb_run, "summary") and isinstance(
                        wandb_run.summary, dict
                    ):
                        wandb_run.summary["dpg"] = summary_entry
                    else:
                        wandb_run.summary.update({"dpg": summary_entry})
                except Exception:
                    print("WARNING: Unable to add DPG summary to wandb summary")

                # Log a tidy table with one row per (class, feature, min, max) for easy visual comparison
                try:
                    table_rows = []
                    for cname in sorted(normalized.keys()):
                        for feat, bounds in normalized[cname].items():
                            minv = bounds["min"] if bounds["min"] is not None else None
                            maxv = bounds["max"] if bounds["max"] is not None else None
                            table_rows.append([cname, feat, minv, maxv])

                    table = wandb.Table(
                        columns=["class", "feature", "min", "max"], data=table_rows
                    )
                    wandb.log({"dpg/constraints_table": table})
                except Exception as exc:
                    print(
                        f"WARNING: Failed to log normalized DPG constraints table to WandB: {exc}"
                    )
        except Exception as exc:
            print(f"WARNING: Failed to log DPG constraints to WandB: {exc}")
    # -----------------------------------------------------------------------

    # Prepare data dict (renamed from iris_data for generality)
    dataset_data = {
        "features": FEATURES,
        "labels": LABELS,
        "feature_names": FEATURE_NAMES,
        "train_features": TRAIN_FEATURES,
        "train_labels": TRAIN_LABELS,
        "continuous_indices": dataset_info.get(
            "continuous_indices", list(range(len(FEATURE_NAMES)))
        ),
        "categorical_indices": dataset_info.get("categorical_indices", []),
        "variable_indices": dataset_info.get(
            "variable_indices", list(range(len(FEATURE_NAMES)))
        ),
    }

    # Determine number of classes for color assignment
    n_classes = len(np.unique(LABELS))
    class_colors_list = [
        "purple",
        "green",
        "orange",
        "red",
        "blue",
        "yellow",
        "pink",
        "cyan",
    ][:n_classes]

    # Ensure normalized_constraints is computed even without wandb
    if normalized_constraints is None and constraints:
        normalized_constraints = ConstraintParser.normalize_constraints(constraints)

    # --- Generate DPG Constraints Overview Visualization ---
    # This visualization shows all constraint boundaries before any counterfactual generation
    if normalized_constraints and getattr(config.output, "save_visualizations", True):
        try:
            # Create output directory for experiment-level visualizations
            output_dir = getattr(config.output, "local_dir", "outputs")
            experiment_name = getattr(config.experiment, "name", "experiment")
            experiment_dir = os.path.join(output_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Generate and save the constraints overview
            constraints_fig = plot_dpg_constraints_overview(
                normalized_constraints=normalized_constraints,
                feature_names=FEATURE_NAMES,
                class_colors_list=class_colors_list,
                output_path=os.path.join(
                    experiment_dir, "dpg_constraints_overview.png"
                ),
                title="DPG Constraints Overview",
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
    sample_indices_config = getattr(config.experiment_params, "sample_indices", None)
    if sample_indices_config is not None:
        sample_indices = sample_indices_config
    else:
        # Select random samples from training split using seeded RNG for reproducibility
        rng = np.random.default_rng(config.experiment_params.seed)
        sample_indices = rng.choice(
            len(TRAIN_FEATURES),
            size=min(config.experiment_params.num_samples, len(TRAIN_FEATURES)),
            replace=False,
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
            normalized_constraints,
        )
        results.append(result)

    # Compute aggregate runtime metrics
    # For DPG: includes boundary extraction + all sample generation times
    # For DICE: only sample generation times (boundary extraction excluded as it's only for viz)
    total_generation_runtime = sum(r["generation_runtime"] for r in results)
    method_name = config.counterfactual.method if hasattr(config.counterfactual, "method") else "unknown"
    if method_name.lower() == "dpg":
        total_generation_runtime += boundary_extraction_time
    
    # Log experiment-level summary
    total_success_rate = np.mean([r["success_rate"] for r in results])
    total_valid = sum(r["valid_counterfactuals"] for r in results)
    total_requested = sum(r["requested_counterfactuals"] for r in results)

    print(f"\n{'=' * 60}")
    print("Experiment Complete!")
    print(f"{'=' * 60}")
    print(f"Samples processed: {len(results)}")
    print(f"Total valid counterfactuals: {total_valid}/{total_requested}")
    print(f"Overall success rate: {total_success_rate:.2%}")
    print(f"Total generation runtime: {total_generation_runtime:.3f}s")
    print(f"\nSample Details:")
    for r in results:
        print(f"  - {r['sample_id']}: Original Class={r['original_class']}, Target Class={r['target_class']}")
    print(f"{'=' * 60}\n")

    if wandb_run:
        # Log experiment-level summary using helper
        log_experiment_summary_to_wandb(
            wandb_run=wandb_run,
            results=results,
            total_valid=total_valid,
            total_requested=total_requested,
            total_success_rate=total_success_rate,
            total_generation_runtime=total_generation_runtime
        )

    # Save experiment-level summary metrics to CSV
    output_dir = getattr(config.output, "local_dir", "outputs")
    save_experiment_level_metrics(results, config, output_dir, total_generation_runtime)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual experiments with WandB tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (e.g., configs/iris/dpg/config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., iris, german_credit) - used with --method to auto-construct config path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name (e.g., dpg, dice) - selects method from unified config or used with --dataset for legacy paths",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="Override config values (e.g., --set counterfactual.population_size=50)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from WandB run ID"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Run WandB in offline mode"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Optional comment to log with this experiment run",
    )

    args = parser.parse_args()

    # Construct config path from dataset if provided
    if args.dataset:
        if args.config:
            print(
                "WARNING: --config specified along with --dataset. Using --config value."
            )
        else:
            # First try unified config path
            unified_config_path = f"configs/{args.dataset}/config.yaml"
            legacy_config_path = (
                f"configs/{args.dataset}/{args.method or 'dpg'}/config.yaml"
                if args.method
                else None
            )

            if os.path.exists(os.path.join(REPO_ROOT, unified_config_path)):
                args.config = unified_config_path
                print(f"INFO: Using unified config: {args.config}")
            elif legacy_config_path and os.path.exists(
                os.path.join(REPO_ROOT, legacy_config_path)
            ):
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
        print(
            "  python scripts/run_experiment.py --config configs/iris/config.yaml --method dice"
        )
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
        if not hasattr(config, "experiment"):
            config.experiment = {}
        config.experiment.notes = args.comment
        if args.verbose:
            print(f"INFO: Logging comment: {args.comment}")

    # Get output directory for status tracking
    output_dir = pathlib.Path(
        getattr(config.output, "local_dir", "outputs")
        if hasattr(config, "output")
        else "outputs"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    # Determine dataset and method for status tracking
    dataset_name = args.dataset or getattr(config.data, "dataset", "unknown")
    method_name = args.method or getattr(config.counterfactual, "method", "unknown")

    # Add method to data config for easy filtering in compare_techniques.py
    # This must be done before init_wandb so it's included in the initial config
    # Use __setitem__ to ensure _config dict is updated for to_dict()
    if hasattr(config, "data"):
        config.data["method"] = method_name

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
            print(
                "INFO: Configured WandB metric definitions for improved visualization"
            )
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


if __name__ == "__main__":
    main()
