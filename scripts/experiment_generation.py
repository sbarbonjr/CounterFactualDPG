"""Script version of the `experiment_generation` notebook.

This script reproduces the notebook behaviour but selects a sample at random (seeded)
so it can be executed and tested easily outside of a notebook environment.

It exposes a function `run_experiment(...)` that can be imported and invoked from tests
with small parameter values for fast execution.
"""

from __future__ import annotations

import pathlib, sys
# Ensure repo root is on sys.path so the script can be executed directly without setting PYTHONPATH
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os
import pickle
import warnings
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# Defaults (match notebook values unless overridden)
DEFAULT_OUTPUT_DIR = "experiment_results"
CLASS_COLORS_LIST = ["purple", "green", "orange"]
RULES = ["no_change", "non_increasing", "non_decreasing", "none"]

logger = logging.getLogger(__name__)


def _make_original_sample_dict(feature_names: List[str], feature_values: List[float]) -> Dict[str, float]:
    return dict(zip(feature_names, map(float, feature_values)))


def run_experiment(
    seed: int = 42,
    sample_index: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_combinations_to_test: Optional[int] = None,
    num_replications: int = 3,
    initial_population_size: int = 20,
    max_generations: int = 60,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the counterfactual generation pipeline for one sample.

    Parameters are provided so tests can run with smaller values.
    Returns dict with sample_id, and saved filepaths.
    """

    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore")

    rng = np.random.RandomState(seed)

    # Load data and split
    IRIS = load_iris()
    IRIS_FEATURES = IRIS.data
    IRIS_LABELS = IRIS.target

    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        IRIS_FEATURES, IRIS_LABELS, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(TRAIN_FEATURES, TRAIN_LABELS)

    # Extract constraints
    constraints = ConstraintParser.extract_constraints_from_dataset(model, TRAIN_FEATURES, TRAIN_LABELS, IRIS.feature_names)

    # Choose a sample index (random if not provided)
    if sample_index is None:
        sample_index = int(rng.choice(len(IRIS_FEATURES)))

    original_sample_values = IRIS_FEATURES[sample_index]
    ORIGINAL_SAMPLE = _make_original_sample_dict(list(IRIS.feature_names), original_sample_values)
    SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])
    ORIGINAL_SAMPLE_PREDICTED_CLASS = int(model.predict(SAMPLE_DATAFRAME)[0])

    # Prepare CF model and parameters
    FEATURES_NAMES = list(ORIGINAL_SAMPLE.keys())
    RULES_COMBINATIONS = list(__import__('itertools').product(RULES, repeat=len(FEATURES_NAMES)))

    if num_combinations_to_test is None:
        num_combinations_to_test = int(len(RULES_COMBINATIONS) / 2)

    TARGET_CLASS = 0

    # Save sample metadata
    SAMPLE_ID = get_sample_id(sample_index)
    save_sample_metadata(SAMPLE_ID, ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, TARGET_CLASS, sample_index)

    if verbose:
        logger.info(f"Sample ID: {SAMPLE_ID} (dataset index: {sample_index})")
        logger.info(f"Original Predicted Class: {ORIGINAL_SAMPLE_PREDICTED_CLASS}")
        logger.info(f"Total possible rule combinations: {len(RULES_COMBINATIONS)}")
        logger.info(f"Testing {num_combinations_to_test} combinations")

    counterfactuals_df_combinations: List[Dict[str, Any]] = []
    visualizations: List[Dict[str, Any]] = []

    # Loop through combinations
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:num_combinations_to_test]):
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
        counterfactuals_df_replications: List[Dict[str, Any]] = []
        combination_viz = {'label': combination, 'pairwise': None, 'pca': None, 'replication': []}

        skip_combination = False

        for replication in range(num_replications):
            if skip_combination:
                break

            cf_model = CounterFactualModel(model, constraints)
            cf_model.dict_non_actionable = dict_non_actionable

            counterfactual = cf_model.generate_counterfactual(ORIGINAL_SAMPLE, TARGET_CLASS, initial_population_size, max_generations)

            if counterfactual is None:
                if replication == (num_replications - 1):
                    skip_combination = True
                continue

            # store
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

        if counterfactuals_df_replications:
            counterfactuals_df_replications = pd.DataFrame(counterfactuals_df_replications)
            counterfactuals_df_combinations.extend(counterfactuals_df_replications.to_dict('records'))

        if combination_viz['replication']:
            visualizations.append(combination_viz)

    # Save raw data (lightweight version for pickling)
    raw_data = {'sample_id': SAMPLE_ID, 'original_sample': ORIGINAL_SAMPLE, 'target_class': TARGET_CLASS,
                'features_names': FEATURES_NAMES, 'visualizations_structure': []}

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

    # Ensure sample directory exists and write raw file inside it
    sample_dir = os.path.join(output_dir, str(SAMPLE_ID))
    os.makedirs(sample_dir, exist_ok=True)
    raw_filepath = os.path.join(sample_dir, 'raw_counterfactuals.pkl')
    with open(raw_filepath, 'wb') as f:
        pickle.dump(raw_data, f)

    # Generate visualizations (best-effort, some visualizers may expect notebook backends)
    for combination_idx, combination_viz in enumerate(visualizations):
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination_viz['label']))

        for replication_idx, replication_viz in enumerate(combination_viz['replication']):
            counterfactual = replication_viz['counterfactual']
            cf_model = replication_viz['cf_model']

            # Attempt to create the same visualizations as the notebook
            try:
                replication_visualizations = [
                    plot_sample_and_counterfactual_heatmap(ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, counterfactual, model.predict(pd.DataFrame([counterfactual])), dict_non_actionable),
                    plot_sample_and_counterfactual_comparison(model, ORIGINAL_SAMPLE, SAMPLE_DATAFRAME, counterfactual, constraints, CLASS_COLORS_LIST),
                    cf_model.plot_fitness() if hasattr(cf_model, 'plot_fitness') else None
                ]
            except Exception as exc:  # pragma: no cover - visualizers can be flaky in test env
                logger.warning("Visualization generation failed for a replication: %s", exc)
                replication_visualizations = []

            replication_viz['visualizations'] = replication_visualizations

        # combination-level visualizations
        try:
            counterfactuals_list = [rep['counterfactual'] for rep in combination_viz['replication']]
            cf_features_df = pd.DataFrame(counterfactuals_list)
            combination_viz['pairwise'] = plot_pairwise_with_counterfactual_df(model, IRIS_FEATURES, IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_df)
            combination_viz['pca'] = plot_pca_with_counterfactuals(model, pd.DataFrame(IRIS_FEATURES), IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_df)
        except Exception as exc:
            logger.warning("Combination-level visualization generation failed: %s", exc)
            combination_viz['pairwise'] = None
            combination_viz['pca'] = None

    # Metrics / explainers
    for combination_viz in visualizations:
        for replication_viz in combination_viz['replication']:
            counterfactual = replication_viz['counterfactual']
            cf_model = replication_viz['cf_model']
            explainer = CounterFactualExplainer(cf_model, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)

            replication_viz['explanations'] = {
                'Feature Modifications': explainer.explain_feature_modifications(),
                'Constraints Respect': explainer.check_constraints_respect(),
                'Stopping Criteria': explainer.explain_stopping_criteria(),
                'Final Results': explainer.summarize_final_results(),
            }

    # Save visualizations data into the sample directory
    viz_filepath = os.path.join(sample_dir, 'after_viz_generation.pkl')
    with open(viz_filepath, 'wb') as f:
        pickle.dump({'sample_id': SAMPLE_ID, 'visualizations': visualizations, 'original_sample': ORIGINAL_SAMPLE, 'features_names': FEATURES_NAMES, 'target_class': TARGET_CLASS}, f)

    # Use the storage helper to save a final structured file (as in the notebook)
    try:
        save_visualizations_data(SAMPLE_ID, visualizations, ORIGINAL_SAMPLE, constraints, FEATURES_NAMES, TARGET_CLASS, output_dir=output_dir)
    except Exception as exc:  # pragma: no cover - storage helper may write to a different location
        logger.warning("save_visualizations_data failed: %s", exc)

    if verbose:
        logger.info("Saved raw counterfactuals data to %s", raw_filepath)
        logger.info("Saved visualization data to %s", viz_filepath)

    return {'sample_id': SAMPLE_ID, 'sample_dir': sample_dir, 'raw_filepath': raw_filepath, 'viz_filepath': viz_filepath, 'output_dir': output_dir}


def main():
    parser = argparse.ArgumentParser(description="Run counterfactual experiment for a single random sample (seeded)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-index', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--num-combinations', type=int, default=None)
    parser.add_argument('--num-replications', type=int, default=3)
    parser.add_argument('--initial-population-size', type=int, default=20)
    parser.add_argument('--max-generations', type=int, default=60)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    result = run_experiment(
        seed=args.seed,
        sample_index=args.sample_index,
        output_dir=args.output_dir,
        num_combinations_to_test=args.num_combinations,
        num_replications=args.num_replications,
        initial_population_size=args.initial_population_size,
        max_generations=args.max_generations,
        verbose=args.verbose,
    )

    print(f"Experiment complete: sample_id={result['sample_id']}")
    print(f"Raw data: {result['raw_filepath']}")
    print(f"Viz data: {result['viz_filepath']}")


if __name__ == '__main__':
    main()
