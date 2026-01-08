"""Enhanced experiment runner with WandB integration.

This script provides a parameterized way to run counterfactual generation experiments
with automatic logging to Weights & Biases for experiment tracking and comparison.

Usage:
  # Run with config file
  python scripts/run_experiment.py --config configs/experiment_config.yaml
  
  # Override specific params
  python scripts/run_experiment.py --config configs/experiment_config.yaml \
    --set counterfactual.population_size=50 \
    --set experiment_params.seed=123
    
  # Resume a previous run
  python scripts/run_experiment.py --resume <wandb_run_id>
  
  # Offline mode (no wandb sync)
  python scripts/run_experiment.py --config configs/experiment_config.yaml --offline
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
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
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
import CounterFactualVisualizer as CounterFactualVisualizer
from CounterFactualVisualizer import (
    plot_sample_and_counterfactual_heatmap,
    plot_sample_and_counterfactual_comparison,
    plot_pairwise_with_counterfactual_df,
    plot_pca_with_counterfactuals,
    plot_pca_loadings,
)

from utils.notebooks.experiment_storage import (
    get_sample_id,
    save_sample_metadata,
    save_visualizations_data,
    _get_sample_dir as get_sample_dir,
) 


def load_dataset(config: 'DictConfig'):
    """Load dataset based on config specification.
    
    Returns:
        dict with keys:
            - features: numpy array of feature values
            - labels: numpy array of target labels
            - feature_names: list of feature names
            - features_df: pandas DataFrame with feature names
            - label_encoders: dict of LabelEncoder objects (for categorical features)
    """
    dataset_name = config.data.dataset.lower()
    
    if dataset_name == "iris":
        print("INFO: Loading Iris dataset...")
        iris = load_iris()
        features = iris.data
        labels = iris.target
        feature_names = list(iris.feature_names)
        features_df = pd.DataFrame(features, columns=feature_names)
        label_encoders = {}  # No categorical features in iris
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'features_df': features_df,
            'label_encoders': label_encoders,
        }
    
    elif dataset_name == "german_credit":
        print("INFO: Loading German Credit dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(REPO_ROOT, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Encode categorical variables
        label_encoders = {}
        features_df_encoded = features_df.copy()
        
        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category':
                print(f"INFO: Encoding categorical feature: {col}")
                le = LabelEncoder()
                features_df_encoded[col] = le.fit_transform(features_df[col])
                label_encoders[col] = le
        
        features = features_df_encoded.values
        feature_names = list(features_df_encoded.columns)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Encoded {len(label_encoders)} categorical features")
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'features_df': features_df_encoded,
            'label_encoders': label_encoders,
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: iris, german_credit")


class DictConfig:
    """Simple dict-based config wrapper for dot notation access."""
    
    def __init__(self, config_dict):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert config back to plain dict."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        self._config[key] = value
        setattr(self, key, value)


def load_config(config_path: str) -> DictConfig:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)


def apply_overrides(config: DictConfig, overrides: List[str]) -> DictConfig:
    """Apply CLI overrides to config.
    
    Args:
        config: Base configuration
        overrides: List of "key.subkey=value" strings
    
    Returns:
        Updated config
    """
    import ast
    
    for override in overrides:
        if '=' not in override:
            print(f"WARNING: Invalid override format: {override}. Expected 'key=value'")
            continue
        
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # Try to parse value as Python literal
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Keep as string if can't parse
            value = value_str
        
        # Navigate to the right place in config
        current = config
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                print(f"WARNING: Config key not found: {key_path}")
                break
        else:
            # Set the final value
            final_key = keys[-1]
            if hasattr(current, final_key):
                setattr(current, final_key, value)
                current._config[final_key] = value
                print(f"INFO: Override applied: {key_path} = {value}")
            else:
                print(f"WARNING: Config key not found: {key_path}")
    
    return config


# Manual git collection removed — rely on WandB's built-in git integration


def init_wandb(config: DictConfig, resume_id: Optional[str] = None, offline: bool = False):
    """Initialize Weights & Biases run."""
    if not WANDB_AVAILABLE:
        print("WARNING: WandB not available, skipping initialization")
        return None
    
    mode = "offline" if offline else "online"
    
    # Allow optional entity (organization/team) to be specified in config
    entity = getattr(config.experiment, 'entity', None)
    
    if resume_id:
        run = wandb.init(
            entity=entity,
            project=config.experiment.project,
            id=resume_id,
            resume="must",
            mode=mode
        )
    else:
        run = wandb.init(
            entity=entity,
            project=config.experiment.project,
            name=config.experiment.name,
            config=config.to_dict(),
            tags=getattr(config.experiment, 'tags', None),
            notes=getattr(config.experiment, 'notes', None),
            mode=mode
        )

    # Use WandB's built-in git integration; manual collection removed

    return run


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
        sample_index: Index of the sample in the dataset
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
    
    output_dir = config.output.local_dir
    
    # Get original sample
    original_sample_values = FEATURES[sample_index]
    ORIGINAL_SAMPLE = dict(zip(FEATURE_NAMES, map(float, original_sample_values)))
    SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])
    ORIGINAL_SAMPLE_PREDICTED_CLASS = int(model.predict(SAMPLE_DATAFRAME)[0])
    
    # Prepare CF parameters
    FEATURES_NAMES = list(ORIGINAL_SAMPLE.keys())
    RULES = config.counterfactual.rules
    RULES_COMBINATIONS = list(__import__('itertools').product(RULES, repeat=len(FEATURES_NAMES)))
    
    num_combinations_to_test = config.experiment_params.num_combinations_to_test
    if num_combinations_to_test is None:
        num_combinations_to_test = int(len(RULES_COMBINATIONS) / 2)
    
    # Choose target class
    n_classes = len(np.unique(LABELS))
    TARGET_CLASS = 0 if ORIGINAL_SAMPLE_PREDICTED_CLASS != 0 else (ORIGINAL_SAMPLE_PREDICTED_CLASS + 1) % n_classes
    
    # Save sample metadata
    SAMPLE_ID = get_sample_id(sample_index)
    configname = getattr(config.experiment, 'name', None)
    save_sample_metadata(SAMPLE_ID, ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, TARGET_CLASS, sample_index, configname=configname, output_dir=output_dir) 
    
    print(f"INFO: Processing Sample ID: {SAMPLE_ID} (dataset index: {sample_index})")
    print(f"INFO: Original Predicted Class: {ORIGINAL_SAMPLE_PREDICTED_CLASS}, Target Class: {TARGET_CLASS}, combinations to test: {num_combinations_to_test}/{len(RULES_COMBINATIONS)}")
    
    # Log sample info to WandB
    if wandb_run:
        wandb.log({
            "sample/sample_id": SAMPLE_ID,
            "sample/sample_index": sample_index,
            "sample/original_class": ORIGINAL_SAMPLE_PREDICTED_CLASS,
            "sample/target_class": TARGET_CLASS,
        })
    
    counterfactuals_df_combinations = []
    visualizations = []
    
    valid_counterfactuals = 0
    total_replications = 0
    
    # Loop through combinations
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:num_combinations_to_test]):
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
        counterfactuals_df_replications = []
        combination_viz = {'label': combination, 'pairwise': None, 'pca': None, 'replication': []}
        
        skip_combination = False
        
        for replication in range(config.experiment_params.num_replications):
            total_replications += 1
            
            if skip_combination:
                break
            
            # Create CF model with config parameters
            cf_model = CounterFactualModel(
                model, 
                constraints,
                dict_non_actionable=dict_non_actionable,
                verbose=False,
                diversity_weight=config.counterfactual.diversity_weight,
                repulsion_weight=config.counterfactual.repulsion_weight,
                boundary_weight=config.counterfactual.boundary_weight,
                distance_factor=config.counterfactual.distance_factor,
                sparsity_factor=config.counterfactual.sparsity_factor,
                constraints_factor=config.counterfactual.constraints_factor,
            )
            
            counterfactual = cf_model.generate_counterfactual(
                ORIGINAL_SAMPLE, 
                TARGET_CLASS, 
                config.counterfactual.population_size,
                config.counterfactual.max_generations
            )
            
            if counterfactual is None:
                if replication == (config.experiment_params.num_replications - 1):
                    skip_combination = True
                
                # Log failed replication
                if wandb_run:
                    wandb.log({
                        "replication/sample_id": SAMPLE_ID,
                        "replication/combination": str(combination),
                        "replication/replication_num": replication,
                        "replication/success": False,
                    })
                continue
            
            valid_counterfactuals += 1
            
            # Store replication data with evolution history
            evolution_history = getattr(cf_model, 'evolution_history', [])
            replication_viz = {
                'counterfactual': counterfactual,
                'cf_model': cf_model,
                'evolution_history': evolution_history,  # Store evolution for PCA visualization
                'visualizations': [],
                'explanations': {}
            }
            combination_viz['replication'].append(replication_viz)
            
            cf_data = counterfactual.copy()
            cf_data.update({'Rule_' + k: v for k, v in dict_non_actionable.items()})
            cf_data['Replication'] = replication + 1
            counterfactuals_df_replications.append(cf_data)
            
            # Get metrics from explainer
            explainer = CounterFactualExplainer(cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
            metrics = explainer.get_all_metrics()
            
            # Compute cf_eval metrics if available
            cf_eval_metrics = {}
            if CF_EVAL_AVAILABLE:
                try:
                    x_original = np.array([ORIGINAL_SAMPLE[feat] for feat in FEATURES_NAMES])
                    cf_array = np.array([[counterfactual[feat] for feat in FEATURES_NAMES]])
                    continuous_features = list(range(len(FEATURES_NAMES)))
                    
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
                    "replication/combination": str(combination),
                    "replication/replication_num": replication,
                    "replication/success": True,
                    "replication/final_fitness": best_fitness,
                    "replication/generations_to_converge": len(cf_model.best_fitness_list),
                    "replication/num_feature_changes": metrics['num_feature_changes'],
                    "replication/constraints_respected": metrics['constraints_respected'],
                }
                
                # Add all metrics from explainer
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        log_data[f"metrics/{key}"] = value
                
                # Add cf_eval metrics
                log_data.update(cf_eval_metrics)
                
                wandb.log(log_data)
                
                # Log fitness curve
                if cf_model.best_fitness_list and cf_model.average_fitness_list:
                    for gen, (best, avg) in enumerate(zip(cf_model.best_fitness_list, cf_model.average_fitness_list)):
                        wandb.log({
                            "fitness/generation": gen,
                            "fitness/best": best,
                            "fitness/average": avg,
                            "fitness/sample_id": SAMPLE_ID,
                            "fitness/combination": str(combination),
                            "fitness/replication": replication,
                        })
        
        if counterfactuals_df_replications:
            counterfactuals_df_replications = pd.DataFrame(counterfactuals_df_replications)
            counterfactuals_df_combinations.extend(counterfactuals_df_replications.to_dict('records'))
        
        # Compute combination-level cf_eval metrics
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
                    "combination/combination": str(combination),
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
        wandb.log({
            "sample/num_valid_counterfactuals": valid_counterfactuals,
            "sample/total_replications": total_replications,
            "sample/success_rate": success_rate,
        })
    
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
    sample_dir = get_sample_dir(SAMPLE_ID, output_dir=output_dir, configname=configname)
    os.makedirs(sample_dir, exist_ok=True)
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
        except Exception as exc:
            print(f"WARNING: Failed to save DPG constraints to sample folder: {exc}")
    
    # Generate visualizations if enabled
    if config.output.save_visualizations:
        for combination_idx, combination_viz in enumerate(visualizations):
            dict_non_actionable = dict(zip(FEATURES_NAMES, combination_viz['label']))
            
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

                                    # Save loadings
                                    loadings = pca_local.components_.T * (pca_local.explained_variance_**0.5)
                                    loadings_df = pd.DataFrame(loadings, index=FEATURE_NAMES_LOCAL, columns=['pc1_loading','pc2_loading'])
                                    loadings_df.to_csv(os.path.join(sample_dir, f'pca_loadings_combo_{combination_idx}.csv'))

                                except Exception as exc:
                                    print(f"WARNING: Failed to save PCA numeric data: {exc}")
                    except Exception as exc:
                        print(f"WARNING: Failed saving visualization images: {exc}")

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
    dataset_info = load_dataset(config)
    
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
    
    if wandb_run:
        wandb.log({
            "model/train_accuracy": train_score,
            "model/test_accuracy": test_score,
        })
    
    # Extract constraints (pass numpy array for DPG compatibility)
    print("INFO: Extracting constraints...")
    constraints = ConstraintParser.extract_constraints_from_dataset(
        model, TRAIN_FEATURES.values, TRAIN_LABELS, FEATURE_NAMES
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
                normalized = {}
                for cname in sorted(constraints.keys()):
                    feature_map = {}
                    for entry in constraints[cname]:
                        f = entry.get('feature')
                        minv = entry.get('min')
                        maxv = entry.get('max')
                        if f not in feature_map:
                            feature_map[f] = {'min': minv, 'max': maxv}
                        else:
                            cur = feature_map[f]
                            # For min (lower bound), keep the most restrictive (largest) value if present
                            if minv is not None:
                                if cur['min'] is None or minv > cur['min']:
                                    cur['min'] = minv
                            # For max (upper bound), keep the most restrictive (smallest) value if present
                            if maxv is not None:
                                if cur['max'] is None or maxv < cur['max']:
                                    cur['max'] = maxv
                    # Order features alphabetically for deterministic display
                    normalized[cname] = {k: feature_map[k] for k in sorted(feature_map.keys())}

                # Store for saving in sample folders
                normalized_constraints = normalized

                # Put normalized constraints into config so they appear under the Config tab
                try:
                    try:
                        wandb_run.config['dpg'] = normalized
                    except Exception:
                        wandb_run.config.update({'dpg': normalized})
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
    }
    
    # Determine number of classes for color assignment
    n_classes = len(np.unique(LABELS))
    class_colors_list = ["purple", "green", "orange", "red", "blue", "yellow", "pink", "cyan"][:n_classes]
    
    # Determine sample indices to process
    if config.experiment_params.sample_indices is not None:
        sample_indices = config.experiment_params.sample_indices
    else:
        # Select random samples
        sample_indices = np.random.choice(
            len(FEATURES),
            size=min(config.experiment_params.num_samples, len(FEATURES)),
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
        wandb.log({
            "experiment/total_samples": len(results),
            "experiment/total_valid_counterfactuals": total_valid,
            "experiment/total_replications": total_replications,
            "experiment/overall_success_rate": total_success_rate,
        })
        
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
        default='configs/experiment_config.yaml',
        help='Path to config YAML file'
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
    
    args = parser.parse_args()
    

    
    # Load config
    print(f"INFO: Loading config from {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.overrides:
        print(f"INFO: Applying {len(args.overrides)} config overrides")
        config = apply_overrides(config, args.overrides)
    
    # Initialize WandB
    wandb_run = None
    if WANDB_AVAILABLE:
        print("INFO: Initializing Weights & Biases...")
        wandb_run = init_wandb(config, resume_id=args.resume, offline=args.offline)
    else:
        print("WARNING: WandB not available. Running without experiment tracking.")
    
    try:
        # Run experiment
        results = run_experiment(config, wandb_run)
        
        # Finish WandB run
        if wandb_run:
            wandb.finish()
        
        return results
    
    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        traceback.print_exc()
        if wandb_run:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
