#!/usr/bin/env python
"""RandomForest Hyperparameter Tuning Script.

This script performs randomized search for tuning RandomForest hyperparameters
on a specified dataset from the configs directory.

Usage:
    python scripts/tune_random_forest.py --dataset iris
    python scripts/tune_random_forest.py --dataset german_credit --n-iter 50
    python scripts/tune_random_forest.py --dataset diabetes --cv 10 --scoring f1_weighted
    python scripts/tune_random_forest.py --dataset iris --scoring accuracy dpg_constraints --refit dpg_constraints
    python scripts/tune_random_forest.py --dataset iris --scoring accuracy dpg_constraints --weights 0.5 0.5
"""

from __future__ import annotations

import pathlib
import sys

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
import yaml
from datetime import datetime

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

from utils.dataset_loader import load_dataset
from utils.config_manager import load_config
from ConstraintParser import ConstraintParser
from constraint_scorer import compute_constraint_score

# Import DPG visualization if available
try:
    from DPG.dpg import plot_dpg_constraints_overview
    import matplotlib.pyplot as plt
    DPG_PKG_AVAILABLE = True
except ImportError:
    DPG_PKG_AVAILABLE = False
    print("Warning: DPG package not available for visualization. Install with requirements in DPG/")

# =============================================================================
# RANDOMFOREST HYPERPARAMETER SEARCH SPACE
# Common parameters for classification problems
# =============================================================================

# Number of trees in the forest
N_ESTIMATORS = [2,3,4, 5,7, 10,12, 15, 20]

# Maximum depth of each tree (None means unlimited)
MAX_DEPTH = [3, 4, 5, 6, 7, 8, 10]

# Minimum samples required to split an internal node
# MIN_SAMPLES_SPLIT = [2, 3, 4, 5, 7, 10, 12, 15]

# Minimum samples required at a leaf node
# MIN_SAMPLES_LEAF = [1, 2, 4, 6, 8]

# Number of features to consider for the best split
MAX_FEATURES = [ None, 0.3, 0.5]

# Whether to bootstrap samples when building trees
# BOOTSTRAP = [True, False]

# Criterion for measuring quality of a split
# CRITERION = ['entropy','gini']

# Class weight options for imbalanced datasets
# CLASS_WEIGHT = [None, 'balanced', 'balanced_subsample']

# Maximum number of leaf nodes (None for unlimited)
# MAX_LEAF_NODES = [None, 10, 50, 100, 200]

# Minimum impurity decrease for a split
MIN_IMPURITY_DECREASE = [0.0,0.08, 0.01, 0.02, 0.05,0.1]

# Complete parameter distribution for RandomizedSearchCV
PARAM_DISTRIBUTIONS = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': MAX_DEPTH,
    # 'min_samples_split': MIN_SAMPLES_SPLIT,
    # 'min_samples_leaf': MIN_SAMPLES_LEAF,
    'max_features': MAX_FEATURES,
    # 'bootstrap': BOOTSTRAP,
    # 'criterion': CRITERION,
    # 'class_weight': CLASS_WEIGHT,
    # 'max_leaf_nodes': MAX_LEAF_NODES,
    'min_impurity_decrease': MIN_IMPURITY_DECREASE,
}

# Default RandomizedSearchCV settings
DEFAULT_N_ITER = 10  # Number of parameter combinations to try
DEFAULT_CV = 10  # Number of cross-validation folds
DEFAULT_SCORING = ['accuracy']  # Default scoring metrics (list for multi-metric support)
DEFAULT_REFIT = 'accuracy'  # Default metric to use for selecting best model
DEFAULT_N_JOBS = -1  # Use all available cores
DEFAULT_RANDOM_STATE = 42

# Available scoring metrics for classification
AVAILABLE_SCORING = [
    'accuracy',
    'f1',
    'f1_weighted',
    'f1_macro',
    'f1_micro',
    'precision',
    'precision_weighted',
    'recall',
    'recall_weighted',
    'roc_auc',
    'roc_auc_ovr',
    'roc_auc_ovo',
    'dpg_constraints',
]


def make_dpg_constraint_scorer(feature_names: list, dpg_config: dict = None):
    """Create a custom scorer that optimizes for DPG constraint separation.
    
    This scorer extracts DPG constraints from the fitted model and computes
    a constraint separation score. Higher scores indicate better separation
    between classes, which is beneficial for counterfactual generation.
    
    Args:
        feature_names: List of feature names for the dataset
        dpg_config: Optional DPG configuration dictionary
    
    Returns:
        A scorer function compatible with sklearn's cross-validation
    """
    def dpg_constraint_scorer(estimator, X, y):
        """Score function that computes DPG constraint separation.
        
        Args:
            estimator: Fitted sklearn estimator (RandomForest)
            X: Feature array
            y: Label array
        
        Returns:
            float: Constraint separation score in [0, 1]
        """
        try:
            # Extract constraints from the fitted model
            dpg_result = ConstraintParser.extract_constraints_from_dataset(
                model=estimator,
                train_features=X,
                train_labels=y,
                feature_names=feature_names,
                dpg_config=dpg_config
            )
            
            constraints = dpg_result.get('constraints', {})
            
            if not constraints:
                # No constraints extracted - return low score
                return 0.0
            
            # Normalize constraints to the expected format
            normalized_constraints = ConstraintParser.normalize_constraints(constraints)
            
            if not normalized_constraints:
                return 0.0
            
            # Compute constraint separation score
            # Pass total features and classes for accurate coverage calculation
            n_total_features = len(feature_names)
            n_total_classes = len(np.unique(y))
            score_result = compute_constraint_score(
                normalized_constraints,
                n_total_features=n_total_features,
                n_total_classes=n_total_classes,
                verbose=False
            )
            return score_result['score']
            
        except Exception as e:
            # On any error, return 0 to indicate poor constraints
            print(f"WARNING: DPG constraint scoring failed: {e}")
            return 0.0
    
    return dpg_constraint_scorer


def make_weighted_scorer(scorers: dict, weights: list, scorer_names: list):
    """Create a weighted combination scorer from multiple scorers.
    
    Args:
        scorers: Dictionary mapping metric names to scorer functions
        weights: List of weights for each scorer (must sum to 1.0)
        scorer_names: List of scorer names in order
    
    Returns:
        A weighted scorer function
    """
    # Normalize weights to sum to 1.0
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    def weighted_scorer(estimator, X, y):
        """Compute weighted combination of multiple scores."""
        total_score = 0.0
        for name, weight in zip(scorer_names, normalized_weights):
            scorer = scorers[name]
            # Handle both callable scorers and string metric names
            if callable(scorer):
                score = scorer(estimator, X, y)
            else:
                # Import make_scorer to handle string metrics
                from sklearn.metrics import get_scorer
                sklearn_scorer = get_scorer(scorer)
                score = sklearn_scorer(estimator, X, y)
            total_score += weight * score
        return total_score
    
    return weighted_scorer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Tune RandomForest hyperparameters using RandomizedSearchCV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tune_random_forest.py --dataset iris
    python scripts/tune_random_forest.py --dataset german_credit --n-iter 50
    python scripts/tune_random_forest.py --dataset diabetes --cv 10 --scoring f1_weighted
    python scripts/tune_random_forest.py --dataset iris --scoring accuracy dpg_constraints --refit dpg_constraints
    python scripts/tune_random_forest.py --dataset iris --scoring accuracy dpg_constraints --weights 0.5 0.5
    python scripts/tune_random_forest.py --dataset iris --output results/tuning
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Name of the dataset (must exist in configs/ directory)'
    )
    
    parser.add_argument(
        '--n-iter', '-n',
        type=int,
        default=DEFAULT_N_ITER,
        help=f'Number of parameter settings sampled (default: {DEFAULT_N_ITER})'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=DEFAULT_CV,
        help=f'Number of cross-validation folds (default: {DEFAULT_CV})'
    )
    
    parser.add_argument(
        '--scoring', '-s',
        type=str,
        nargs='+',
        default=DEFAULT_SCORING,
        choices=AVAILABLE_SCORING,
        help=f'Scoring metric(s) for evaluation (can specify multiple, default: {DEFAULT_SCORING})'
    )
    
    parser.add_argument(
        '--refit',
        type=str,
        default=None,
        help='Metric to use for selecting best model when using multiple scorers (default: first scoring metric). Ignored if --weights is specified.'
    )
    
    parser.add_argument(
        '--weights',
        type=float,
        nargs='+',
        default=None,
        help='Weights for each scoring metric (must match number of --scoring metrics). Creates a weighted combination scorer. Example: --weights 0.5 0.5'
    )
    
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=DEFAULT_N_JOBS,
        help=f'Number of parallel jobs (-1 for all cores, default: {DEFAULT_N_JOBS})'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f'Random state for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use as test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: outputs/<dataset>_rf_tuning)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=int,
        default=2,
        help='Verbosity level for RandomizedSearchCV (default: 2)'
    )
    
    parser.add_argument(
        '--export-constraints',
        action='store_true',
        help='Export constraints (text, JSON, visualization) for each hyperparameter combination tested'
    )
    
    return parser.parse_args()


def get_dataset_config_path(dataset_name: str) -> str:
    """Get the config path for a dataset."""
    config_dir = REPO_ROOT / 'configs' / dataset_name
    config_path = config_dir / 'config.yaml'
    
    if not config_path.exists():
        available_datasets = [
            d.name for d in (REPO_ROOT / 'configs').iterdir()
            if d.is_dir() and (d / 'config.yaml').exists()
        ]
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {sorted(available_datasets)}"
        )
    
    return str(config_path)


def save_config(config_path: str, config) -> None:
    """Save configuration dictionary to a YAML file.
    
    Args:
        config_path: Path to the YAML file to write
        config: Configuration dictionary or DictConfig to save
    """
    # Convert DictConfig to regular dict if needed
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        # If omegaconf is not available, try to convert to dict directly
        if hasattr(config, '__dict__'):
            config = dict(config)
        elif not isinstance(config, dict):
            config = dict(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)


def update_config_with_model_params(config_path: str, best_params: dict) -> None:
    """Update the config file's model section with best hyperparameters.
    
    This function REPLACES the entire 'model' section of the config file with
    the new tuned parameters, removing any old parameters that are not present
    in best_params. All other sections (data, experiment, experiment_params,
    output, methods, etc.) are preserved exactly as they were loaded.
    
    Args:
        config_path: Path to the config.yaml file
        best_params: Dictionary of best parameters from RandomizedSearchCV
    """
    # Load the dataset-specific config directly (without merging with base config)
    # This ensures we only modify the dataset-specific file and preserve its structure
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Replace the entire model section with only the new parameters
    # This ensures old parameters (like criterion) are removed if not in best_params
    config['model'] = {'type': 'RandomForestClassifier'}
    for param, value in best_params.items():
        # Handle None values properly (written as null in YAML)
        config['model'][param] = value
    
    # Save the updated config back to the file (only model section is modified)
    save_config(config_path, config)


def save_constraints_text(constraints, output_path, constraint_score=None):
    """Save constraints to a text file in a human-readable format.
    
    Args:
        constraints: Dictionary mapping class labels to feature constraints
        output_path: Path to save the text file
        constraint_score: Optional constraint separation score to include
    """
    with open(output_path, "w") as f:
        f.write("DPG Constraints\n")
        f.write("=" * 80 + "\n\n")
        
        if constraint_score is not None:
            f.write(f"Constraint Separation Score: {constraint_score:.4f}\n")
            f.write(f"(Score range: 0=complete overlap, 1=perfect separation)\n\n")
        
        if not constraints:
            f.write("No constraints extracted.\n")
            return
        
        for class_label, features in constraints.items():
            f.write(f"Class: {class_label}\n")
            f.write("-" * 40 + "\n")
            
            if isinstance(features, list):
                for feature_constraint in features:
                    feature = feature_constraint.get("feature", "unknown")
                    min_val = feature_constraint.get("min", None)
                    max_val = feature_constraint.get("max", None)
                    
                    constraint_str = f"  {feature}: "
                    if min_val is not None and max_val is not None:
                        constraint_str += f"{min_val:.4f} < x <= {max_val:.4f}"
                    elif min_val is not None:
                        constraint_str += f"x > {min_val:.4f}"
                    elif max_val is not None:
                        constraint_str += f"x <= {max_val:.4f}"
                    else:
                        constraint_str += "unconstrained"
                    
                    f.write(constraint_str + "\n")
            
            f.write("\n")


def save_constraints_json(constraints, output_path):
    """Save constraints to a JSON file.
    
    Args:
        constraints: Dictionary mapping class labels to feature constraints
        output_path: Path to save the JSON file
    """
    with open(output_path, "w") as f:
        json.dump(constraints, f, indent=2, sort_keys=True)


def export_constraints_for_params(model, X_train, y_train, feature_names, 
                                   output_dir, run_id, dataset_name, 
                                   dpg_config=None, constraint_score=None):
    """Extract and export constraints for a fitted model.
    
    Args:
        model: Fitted sklearn model
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        output_dir: Directory to save outputs
        run_id: Unique identifier for this run
        dataset_name: Name of the dataset
        dpg_config: Optional DPG configuration
        constraint_score: Optional pre-computed constraint score
    """
    # Extract constraints
    dpg_result = ConstraintParser.extract_constraints_from_dataset(
        model=model,
        train_features=X_train,
        train_labels=y_train,
        feature_names=feature_names,
        dpg_config=dpg_config
    )
    
    constraints = dpg_result.get('constraints', {})
    if not constraints:
        print(f"    WARNING: No constraints extracted for run {run_id}")
        return
    
    # Normalize constraints
    normalized_constraints = ConstraintParser.normalize_constraints(constraints)
    
    # Compute constraint score if not provided
    if constraint_score is None and normalized_constraints:
        try:
            n_total_features = len(feature_names)
            n_total_classes = len(np.unique(y_train))
            score_result = compute_constraint_score(
                normalized_constraints,
                n_total_features=n_total_features,
                n_total_classes=n_total_classes,
                verbose=False
            )
            constraint_score = score_result['score']
        except Exception:
            pass
    
    # Save text file
    text_path = output_dir / f"{run_id}.txt"
    save_constraints_text(constraints, text_path, constraint_score=constraint_score)
    
    # Save JSON file
    json_path = output_dir / f"{run_id}.json"
    save_constraints_json(normalized_constraints, json_path)
    
    # Generate visualization if available
    if normalized_constraints and DPG_PKG_AVAILABLE:
        try:
            n_classes = len(np.unique(y_train))
            class_colors_list = [
                "purple", "green", "orange", "red", "blue", "yellow", "pink", "cyan"
            ][:n_classes]
            
            viz_path = output_dir / f"{run_id}.png"
            constraints_fig = plot_dpg_constraints_overview(
                normalized_constraints=normalized_constraints,
                feature_names=feature_names,
                class_colors_list=class_colors_list,
                output_path=str(viz_path),
                title=f"DPG Constraints - {dataset_name} - {run_id}",
                constraint_score=constraint_score
            )
            
            if constraints_fig:
                plt.close(constraints_fig)
        except Exception as exc:
            print(f"    WARNING: Failed to generate visualization for run {run_id}: {exc}")


def evaluate_model(model, X_test, y_test, multiclass: bool = False) -> dict:
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        # 'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        # 'f1_macro': f1_score(y_test, y_pred, average='macro'),
        # 'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        # 'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # ROC-AUC for binary or multiclass
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if multiclass:
                metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
    except Exception:
        pass  # ROC-AUC not applicable
    
    return metrics


def main():
    """Main function to run hyperparameter tuning."""
    args = parse_args()
    
    # Handle scoring arguments
    scoring_metrics = args.scoring if isinstance(args.scoring, list) else [args.scoring]
    
    # Handle weights if provided
    use_weighted_scoring = args.weights is not None
    if use_weighted_scoring:
        if len(args.weights) != len(scoring_metrics):
            raise ValueError(
                f"Number of weights ({len(args.weights)}) must match number of scoring metrics ({len(scoring_metrics)}). "
                f"Metrics: {scoring_metrics}, Weights: {args.weights}"
            )
        if any(w < 0 for w in args.weights):
            raise ValueError("All weights must be non-negative")
        if sum(args.weights) == 0:
            raise ValueError("At least one weight must be positive")
        refit_metric = 'weighted_score'
    else:
        refit_metric = args.refit if args.refit else scoring_metrics[0]
        # Validate refit metric
        if refit_metric not in scoring_metrics:
            raise ValueError(f"--refit metric '{refit_metric}' must be one of the --scoring metrics: {scoring_metrics}")
    
    print("=" * 70)
    print("RandomForest Hyperparameter Tuning")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Iterations: {args.n_iter}")
    print(f"CV Folds: {args.cv}")
    print(f"Scoring: {', '.join(scoring_metrics)}")
    if use_weighted_scoring:
        print(f"Weights: {args.weights}")
        normalized_weights = [w / sum(args.weights) for w in args.weights]
        print(f"Normalized: {[f'{w:.3f}' for w in normalized_weights]}")
        print(f"Optimization: Weighted combination")
    elif len(scoring_metrics) > 1:
        print(f"Refit metric: {refit_metric}")
    print(f"Random State: {args.random_state}")
    print("=" * 70)
    
    # Load dataset configuration
    config_path = get_dataset_config_path(args.dataset)
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_data = load_dataset(config, repo_root=str(REPO_ROOT))
    
    X = dataset_data['features']
    y = dataset_data['labels']
    feature_names = dataset_data['feature_names']
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Check if multiclass
    n_classes = len(np.unique(y))
    is_multiclass = n_classes > 2
    
    # Create base classifier
    base_clf = RandomForestClassifier(
        random_state=args.random_state,
        n_jobs=1  # RandomizedSearchCV handles parallelism
    )
    
    # Create stratified k-fold cross-validator
    cv = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state
    )
    
    # Run RandomizedSearchCV
    print(f"\nStarting RandomizedSearchCV with {args.n_iter} iterations...")
    print(f"Parameter space size: {np.prod([len(v) for v in PARAM_DISTRIBUTIONS.values()]):,} combinations")
    
    start_time = datetime.now()
    
    # Build scoring dictionary for multi-metric support
    scoring_dict = {}
    for metric in scoring_metrics:
        if metric == 'dpg_constraints':
            print(f"Configuring DPG constraint separation scorer for '{metric}'")
            scoring_dict[metric] = make_dpg_constraint_scorer(feature_names, dpg_config=None)
        else:
            scoring_dict[metric] = metric
    
    # Handle weighted scoring
    if use_weighted_scoring:
        # Add weighted combination scorer
        scoring_dict['weighted_score'] = make_weighted_scorer(
            scoring_dict, args.weights, scoring_metrics
        )
        # Keep individual metrics for reporting
        scoring = scoring_dict
        refit = 'weighted_score'
        print(f"Using weighted combination scorer for optimization")
    # Use single scorer if only one metric, otherwise use dict
    elif len(scoring_dict) == 1:
        scoring = list(scoring_dict.values())[0]
        refit = True
    else:
        scoring = scoring_dict
        refit = refit_metric
        print(f"Using multi-metric evaluation, optimizing for: {refit_metric}")
    
    random_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.random_state,
        return_train_score=True,
        error_score='raise'
    )
    
    random_search.fit(X_train, y_train)
    
    elapsed_time = datetime.now() - start_time
    print(f"\nSearch completed in {elapsed_time}")
    
    # Best parameters and scores
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    for param, value in sorted(random_search.best_params_.items()):
        print(f"  {param}: {value}")
    
    print(f"\nBest CV Score ({refit_metric}): {random_search.best_score_:.4f}")
    
    # Show all metric scores if multi-metric or weighted
    if len(scoring_metrics) > 1 or use_weighted_scoring:
        print("\nAll CV Scores for Best Model:")
        best_idx = random_search.best_index_
        for metric in scoring_metrics:
            score_key = f'mean_test_{metric}'
            if score_key in random_search.cv_results_:
                score = random_search.cv_results_[score_key][best_idx]
                std = random_search.cv_results_[f'std_test_{metric}'][best_idx]
                weight_info = ""
                if use_weighted_scoring:
                    weight_idx = scoring_metrics.index(metric)
                    normalized_weight = args.weights[weight_idx] / sum(args.weights)
                    weight_info = f" [weight: {normalized_weight:.3f}]"
                print(f"  {metric}: {score:.4f} (+/- {std:.4f}){weight_info}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE")
    print("=" * 70)
    
    best_model = random_search.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test, multiclass=is_multiclass)
    
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Compute final DPG constraint score on full training set if used
    final_constraint_score = None
    if 'dpg_constraints' in scoring_metrics:
        print("\n" + "=" * 70)
        print("FINAL DPG CONSTRAINT SCORE (on full training set)")
        print("=" * 70)
        try:
            dpg_scorer = make_dpg_constraint_scorer(feature_names, dpg_config=None)
            final_constraint_score = dpg_scorer(best_model, X_train, y_train)
            print(f"  Constraint Separation Score: {final_constraint_score:.4f}")
            print(f"  (Score range: 0=complete overlap, 1=perfect separation)")
            test_metrics['dpg_constraint_score'] = final_constraint_score
        except Exception as e:
            print(f"  WARNING: Failed to compute final constraint score: {e}")
    
    # Classification report
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 70)
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save results
    output_dir = args.output or f"outputs/{args.dataset}_rf_tuning"
    output_path = REPO_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect all CV scores if multi-metric
    all_cv_scores = {}
    if len(scoring_metrics) > 1 or use_weighted_scoring:
        best_idx = random_search.best_index_
        for metric in scoring_metrics:
            score_key = f'mean_test_{metric}'
            if score_key in random_search.cv_results_:
                all_cv_scores[metric] = {
                    'mean': float(random_search.cv_results_[score_key][best_idx]),
                    'std': float(random_search.cv_results_[f'std_test_{metric}'][best_idx])
                }
        # Add weighted score if used
        if use_weighted_scoring and 'mean_test_weighted_score' in random_search.cv_results_:
            all_cv_scores['weighted_score'] = {
                'mean': float(random_search.cv_results_['mean_test_weighted_score'][best_idx]),
                'std': float(random_search.cv_results_['std_test_weighted_score'][best_idx])
            }
    
    # Save best parameters
    results = {
        'dataset': args.dataset,
        'timestamp': timestamp,
        'search_params': {
            'n_iter': args.n_iter,
            'cv': args.cv,
            'scoring': scoring_metrics,
            'refit': refit_metric,
            'weights': args.weights if use_weighted_scoring else None,
            'random_state': args.random_state,
            'test_size': args.test_size,
        },
        'best_params': random_search.best_params_,
        'best_cv_score': float(random_search.best_score_),
        'all_cv_scores': all_cv_scores if all_cv_scores else None,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'feature_importances': {
            feature_names[i]: float(importances[i])
            for i in np.argsort(importances)[::-1]
        },
        'elapsed_time_seconds': elapsed_time.total_seconds(),
        'n_samples': len(X),
        'n_features': len(feature_names),
        'n_classes': n_classes,
    }
    
    results_file = output_path / f"tuning_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Save CV results as CSV
    import pandas as pd
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_file = output_path / f"cv_results_{timestamp}.csv"
    cv_results_df.to_csv(cv_results_file, index=False)
    print(f"CV results saved to: {cv_results_file}")
    
    # Update the dataset's config.yaml file with best parameters
    update_config_with_model_params(config_path, random_search.best_params_)
    print(f"Config file updated with best parameters: {config_path}")
    
    # Export constraints for all parameter combinations if requested
    if args.export_constraints:
        print("\n" + "=" * 70)
        print("EXPORTING CONSTRAINTS FOR ALL PARAMETER COMBINATIONS")
        print("=" * 70)
        
        # Create constraints output directory
        constraints_dir = REPO_ROOT / "outputs" / "constraints" / args.dataset
        constraints_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {constraints_dir}")
        
        # Get DPG config from dataset config if available
        raw_dpg_config = getattr(config.counterfactual, "config", None) if hasattr(config, 'counterfactual') else None
        dpg_config = None
        if raw_dpg_config is not None:
            if hasattr(raw_dpg_config, "to_dict"):
                raw_dpg_config = raw_dpg_config.to_dict()
            elif hasattr(raw_dpg_config, "_config"):
                raw_dpg_config = raw_dpg_config._config
            
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
        
        # Iterate through all tested parameter combinations
        cv_results = random_search.cv_results_
        n_combinations = len(cv_results['params'])
        
        print(f"\nExporting constraints for {n_combinations} parameter combinations...")
        
        for idx, params in enumerate(cv_results['params']):
            # Create a unique run_id from the combination index
            run_id = f"{timestamp}_comb_{idx:04d}"
            
            print(f"\n[{idx+1}/{n_combinations}] Processing combination {idx}:")
            print(f"  Parameters: {params}")
            
            # Create and fit model with these parameters
            model = RandomForestClassifier(
                **params,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            model.fit(X_train, y_train)
            
            # Get the CV score for this combination (if available)
            cv_score = None
            if len(scoring_metrics) == 1:
                cv_score = cv_results['mean_test_score'][idx]
            elif refit_metric in scoring_metrics:
                score_key = f'mean_test_{refit_metric}'
                if score_key in cv_results:
                    cv_score = cv_results[score_key][idx]
            
            print(f"  CV Score ({refit_metric}): {cv_score:.4f}" if cv_score is not None else "")
            
            # Export constraints
            try:
                export_constraints_for_params(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    feature_names=feature_names,
                    output_dir=constraints_dir,
                    run_id=run_id,
                    dataset_name=args.dataset,
                    dpg_config=dpg_config
                )
                print(f"  Exported: {run_id}.txt, {run_id}.json, {run_id}.png")
            except Exception as exc:
                print(f"  ERROR: Failed to export constraints: {exc}")
        
        print(f"\nConstraint export complete! All artifacts saved to: {constraints_dir}")
    
    # Print config snippet for easy copy-paste
    print("\n" + "=" * 70)
    print("CONFIG SNIPPET (copy to your config.yaml)")
    print("=" * 70)
    print("model:")
    print("  type: RandomForestClassifier")
    for param, value in sorted(random_search.best_params_.items()):
        if value is None:
            print(f"  {param}: null")
        elif isinstance(value, str):
            print(f"  {param}: {value}")
        else:
            print(f"  {param}: {value}")
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
