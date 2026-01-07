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
import logging
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

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
)

logger = logging.getLogger(__name__)


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
            logger.warning(f"Invalid override format: {override}. Expected 'key=value'")
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
                logger.warning(f"Config key not found: {key_path}")
                break
        else:
            # Set the final value
            final_key = keys[-1]
            if hasattr(current, final_key):
                setattr(current, final_key, value)
                current._config[final_key] = value
                logger.info(f"Override applied: {key_path} = {value}")
            else:
                logger.warning(f"Config key not found: {key_path}")
    
    return config


def init_wandb(config: DictConfig, resume_id: Optional[str] = None, offline: bool = False):
    """Initialize Weights & Biases run."""
    if not WANDB_AVAILABLE:
        logger.warning("WandB not available, skipping initialization")
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
    
    return run


def run_single_sample(
    sample_index: int,
    config: DictConfig,
    model,
    constraints: Dict,
    iris_data: Dict,
    class_colors_list: List[str],
    wandb_run=None
) -> Dict[str, Any]:
    """Run counterfactual generation for a single sample with WandB logging."""
    
    IRIS_FEATURES = iris_data['features']
    IRIS_LABELS = iris_data['labels']
    IRIS_FEATURE_NAMES = iris_data['feature_names']
    TRAIN_FEATURES = iris_data['train_features']
    TRAIN_LABELS = iris_data['train_labels']
    
    output_dir = config.output.local_dir
    
    # Get original sample
    original_sample_values = IRIS_FEATURES[sample_index]
    ORIGINAL_SAMPLE = dict(zip(IRIS_FEATURE_NAMES, map(float, original_sample_values)))
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
    n_classes = len(np.unique(IRIS_LABELS))
    TARGET_CLASS = 0 if ORIGINAL_SAMPLE_PREDICTED_CLASS != 0 else (ORIGINAL_SAMPLE_PREDICTED_CLASS + 1) % n_classes
    
    # Save sample metadata
    SAMPLE_ID = get_sample_id(sample_index)
    save_sample_metadata(SAMPLE_ID, ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, TARGET_CLASS, sample_index, output_dir=output_dir)
    
    logger.info(f"Processing Sample ID: {SAMPLE_ID} (dataset index: {sample_index})")
    logger.info(f"Original Predicted Class: {ORIGINAL_SAMPLE_PREDICTED_CLASS}, Target Class: {TARGET_CLASS}")
    
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
            
            # Store replication data
            replication_viz = {
                'counterfactual': counterfactual,
                'cf_model': cf_model,
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
    sample_dir = os.path.join(output_dir, str(SAMPLE_ID))
    os.makedirs(sample_dir, exist_ok=True)
    raw_filepath = os.path.join(sample_dir, 'raw_counterfactuals.pkl')
    with open(raw_filepath, 'wb') as f:
        pickle.dump(raw_data, f)
    
    # Generate visualizations if enabled
    if config.output.save_visualizations:
        for combination_idx, combination_viz in enumerate(visualizations):
            dict_non_actionable = dict(zip(FEATURES_NAMES, combination_viz['label']))
            
            for replication_idx, replication_viz in enumerate(combination_viz['replication']):
                counterfactual = replication_viz['counterfactual']
                cf_model = replication_viz['cf_model']
                
                try:
                    # Create fitness plot
                    fitness_fig = cf_model.plot_fitness() if hasattr(cf_model, 'plot_fitness') else None
                    
                    if fitness_fig and wandb_run:
                        wandb.log({
                            "visualizations/fitness_curve": wandb.Image(fitness_fig),
                            "viz/sample_id": SAMPLE_ID,
                            "viz/combination": str(combination_viz['label']),
                            "viz/replication": replication_idx,
                        })
                    
                    replication_viz['visualizations'] = [fitness_fig] if fitness_fig else []
                except Exception as exc:
                    logger.warning(f"Visualization generation failed: {exc}")
    
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
    
    # Log artifacts to WandB
    if wandb_run:
        artifact = wandb.Artifact(f"sample_{SAMPLE_ID}_results", type="results")
        artifact.add_file(raw_filepath)
        artifact.add_file(viz_filepath)
        wandb.log_artifact(artifact)
    
    logger.info(f"Completed sample {SAMPLE_ID}: {valid_counterfactuals}/{total_replications} successful counterfactuals")
    
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
    
    # Load data
    logger.info("Loading dataset...")
    IRIS = load_iris()
    IRIS_FEATURES = IRIS.data
    IRIS_LABELS = IRIS.target
    
    # Create DataFrame with feature names for consistent handling
    IRIS_FEATURES_DF = pd.DataFrame(IRIS_FEATURES, columns=IRIS.feature_names)
    
    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        IRIS_FEATURES_DF, IRIS_LABELS, 
        test_size=config.data.test_size, 
        random_state=config.data.random_state
    )
    
    # Train model
    logger.info("Training model...")
    if config.model.type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=config.model.n_estimators,
            random_state=config.model.random_state
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
    
    # Train model with DataFrame (preserves feature names)
    model.fit(TRAIN_FEATURES, TRAIN_LABELS)
    
    # Extract constraints (pass numpy array for DPG compatibility)
    logger.info("Extracting constraints...")
    constraints = ConstraintParser.extract_constraints_from_dataset(
        model, TRAIN_FEATURES.values, TRAIN_LABELS, IRIS.feature_names
    )
    
    # Prepare iris data dict
    iris_data = {
        'features': IRIS_FEATURES,
        'labels': IRIS_LABELS,
        'feature_names': list(IRIS.feature_names),
        'train_features': TRAIN_FEATURES,
        'train_labels': TRAIN_LABELS,
    }
    
    class_colors_list = ["purple", "green", "orange"]
    
    # Determine sample indices to process
    if config.experiment_params.sample_indices is not None:
        sample_indices = config.experiment_params.sample_indices
    else:
        # Select random samples
        sample_indices = np.random.choice(
            len(IRIS_FEATURES),
            size=min(config.experiment_params.num_samples, len(IRIS_FEATURES)),
            replace=False
        ).tolist()
    
    logger.info(f"Processing {len(sample_indices)} samples: {sample_indices}")
    
    # Process each sample
    results = []
    for sample_idx in sample_indices:
        result = run_single_sample(
            sample_idx,
            config,
            model,
            constraints,
            iris_data,
            class_colors_list,
            wandb_run
        )
        results.append(result)
    
    # Log experiment-level summary
    total_success_rate = np.mean([r['success_rate'] for r in results])
    total_valid = sum(r['valid_counterfactuals'] for r in results)
    total_replications = sum(r['total_replications'] for r in results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Samples processed: {len(results)}")
    logger.info(f"Total valid counterfactuals: {total_valid}/{total_replications}")
    logger.info(f"Overall success rate: {total_success_rate:.2%}")
    logger.info(f"{'='*60}\n")
    
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
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.overrides:
        logger.info(f"Applying {len(args.overrides)} config overrides")
        config = apply_overrides(config, args.overrides)
    
    # Initialize WandB
    wandb_run = None
    if WANDB_AVAILABLE:
        logger.info("Initializing Weights & Biases...")
        wandb_run = init_wandb(config, resume_id=args.resume, offline=args.offline)
    else:
        logger.warning("WandB not available. Running without experiment tracking.")
    
    try:
        # Run experiment
        results = run_experiment(config, wandb_run)
        
        # Finish WandB run
        if wandb_run:
            wandb.finish()
        
        return results
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        if wandb_run:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
