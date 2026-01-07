"""Script version of `experiment_visualization.ipynb`.

Provides utilities to:
- list available samples (reads metadata files saved by the generation step)
- load visualizations data for a sample
- export summary table (CSV/HTML)
- optionally export plots (matplotlib figures) to PNG files
- compute and export metrics (best-effort; requires cf_eval package)

Designed to be runnable from CLI and importable from tests.
"""

from __future__ import annotations

import pathlib, sys
# Ensure repo root is on sys.path so the script can run directly
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os
import pickle
import argparse
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid blocking on figure rendering when running as a script
matplotlib.use('Agg')
import time

# Attempt to import optional metric helpers
try:
    from cf_eval.metrics import (
        nbr_valid_cf,
        perc_valid_cf,
        continuous_distance,
        avg_nbr_changes_per_cf,
        nbr_changes_per_cf,
    )
    CF_EVAL_AVAILABLE = True
except Exception:
    CF_EVAL_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "experiment_results"

# Use the central storage helpers implemented in utils.notebooks.experiment_storage
from utils.notebooks.experiment_storage import list_available_samples as storage_list_available_samples, load_visualizations_data as storage_load_visualizations_data


def list_available_samples(output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[int, Dict[str, Any]]:
    """Delegate to utils.notebooks.experiment_storage.list_available_samples"""
    return storage_list_available_samples(output_dir)


def load_visualizations_data(sample_id: int, output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Delegate to utils.notebooks.experiment_storage.load_visualizations_data"""
    return storage_load_visualizations_data(sample_id, output_dir)


def build_summary_table(visualizations_data: Dict[str, Any]) -> pd.DataFrame:
    """Build a pandas DataFrame similar to the notebook's summary table."""
    FEATURES_NAMES = visualizations_data['features_names']
    table_data = []

    for combination_idx, combination_viz in enumerate(visualizations_data['visualizations']):
        rules_tuple = combination_viz['label']
        rules_dict = dict(zip(FEATURES_NAMES, rules_tuple))

        for replication_idx, replication_viz in enumerate(combination_viz['replication']):
            row = {
                'Combination': combination_idx + 1,
                'Replication': replication_idx + 1,
            }

            for feature_name in FEATURES_NAMES:
                row[f'Rule_{feature_name}'] = rules_dict[feature_name]

            counterfactual = replication_viz.get('counterfactual', {})
            for feature_name in FEATURES_NAMES:
                row[f'CF_{feature_name}'] = counterfactual.get(feature_name, None)

            row['Num_Visualizations'] = len(replication_viz.get('visualizations', []))
            row['Num_Explanations'] = len(replication_viz.get('explanations', {}))

            explanations = replication_viz.get('explanations', {})
            for explanation_name, explanation_value in explanations.items():
                if explanation_name == 'Feature Modifications' and isinstance(explanation_value, list):
                    formatted_mods = []
                    for mod in explanation_value:
                        feature_name = str(mod['feature_name'])
                        old_val = float(mod['old_value'])
                        new_val = float(mod['new_value'])
                        delta = new_val - old_val
                        formatted_mods.append(f"{feature_name}, {old_val} => {new_val} ({delta:+.2f})")
                    row[explanation_name] = "\n".join(formatted_mods)
                else:
                    row[explanation_name] = explanation_value

            # Copy any extra fields
            for key in replication_viz.keys():
                if key not in ['counterfactual', 'cf_model', 'visualizations', 'explanations']:
                    row[key] = replication_viz[key]

            table_data.append(row)

    summary_df = pd.DataFrame(table_data)
    return summary_df


def export_plots_for_sample(visualizations_data: Dict[str, Any], sample_id: int, output_dir: str, export_dir: Optional[str] = None, save_plots: bool = False) -> List[str]:
    """Export plots for a sample.

    By default this function will *not* write PNG files to disk; instead it writes
    a small index (`plots_index.json`) under the sample folder that describes which
    plots are available and where they would be saved. Set `save_plots=True` to
    actually write PNG files (this can be slow for many combinations).

    Returns list of saved file paths (or the generated index file path when not saving PNGs).
    """
    saved_files: List[str] = []

    sample_dir = os.path.join(output_dir, str(sample_id))
    plots_dir = export_dir if export_dir is not None else os.path.join(sample_dir, 'plots')
    os.makedirs(sample_dir, exist_ok=True)

    # Index of plots (will be written to sample_dir/plots_index.json)
    index = {
        'sample_id': sample_id,
        'plots': []
    }

    def _try_save(fig, path):
        """Attempt to save a visualization object to disk.

        Handles matplotlib Figure/Axes, some plotly objects (best-effort), and falls back
        to pickling unknown objects. Measures elapsed time and logs slow saves.
        """
        try:
            start = time.monotonic()

            # Matplotlib Figure
            if hasattr(fig, 'savefig'):
                fig.savefig(path, bbox_inches='tight')
            # Matplotlib Axes -> get the parent Figure
            elif hasattr(fig, 'figure') and hasattr(fig.figure, 'savefig'):
                fig.figure.savefig(path, bbox_inches='tight')
            else:
                # Try plotly static export if available
                try:
                    import plotly.io as pio
                    # Some plotly objects are Figures/dicts
                    pio.write_image(fig, path)
                except Exception:
                    # Fallback: pickle the object so it is not lost
                    with open(path + '.pkl', 'wb') as f:
                        pickle.dump(fig, f)

            elapsed = time.monotonic() - start
            if elapsed > 10:
                logger.warning("Saving figure to %s took %.2fs", path, elapsed)
            else:
                logger.info("Saved figure to %s (%.2fs)", path, elapsed)
            return True
        except Exception as exc:
            logger.warning("Failed to save figure to %s: %s", path, exc)
            return False

    # Iterate combinations and build index (or save files if requested)
    for combo_idx, combination_viz in enumerate(visualizations_data.get('visualizations', [])):
        combo_rel = f"combination_{combo_idx+1}"
        combo_dir = os.path.join(plots_dir, combo_rel)
        if save_plots:
            os.makedirs(combo_dir, exist_ok=True)

        for name in ('pca', 'pairwise'):
            fig = combination_viz.get(name)
            if fig is not None:
                rel_path = os.path.join(combo_rel, f"{name}.png")
                abs_path = os.path.join(plots_dir, rel_path)
                if save_plots:
                    if _try_save(fig, abs_path):
                        saved_files.append(abs_path)
                index['plots'].append({'type': name, 'combination': combo_idx+1, 'path': rel_path})

        for rep_idx, replication_viz in enumerate(combination_viz.get('replication', [])):
            rep_rel = os.path.join(combo_rel, f"replication_{rep_idx+1}")
            rep_dir = os.path.join(plots_dir, rep_rel)
            if save_plots:
                os.makedirs(rep_dir, exist_ok=True)
            for viz_idx, viz in enumerate(replication_viz.get('visualizations', [])):
                rel_path = os.path.join(rep_rel, f"viz_{viz_idx+1}.png")
                abs_path = os.path.join(plots_dir, rel_path)
                logger.info("Processing combination %d replication %d viz %d -> %s", combo_idx+1, rep_idx+1, viz_idx+1, rel_path)
                if save_plots:
                    success = _try_save(viz, abs_path)
                    if success:
                        saved_files.append(abs_path)
                    else:
                        logger.warning("Failed to save combination %d replication %d viz %d", combo_idx+1, rep_idx+1, viz_idx+1)
                index['plots'].append({'type': 'replication_viz', 'combination': combo_idx+1, 'replication': rep_idx+1, 'viz_idx': viz_idx+1, 'path': rel_path})

    # Always write a small index manifest so plots can be regenerated on demand
    try:
        import json
        index_path = os.path.join(sample_dir, 'plots_index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        saved_files.append(index_path)
    except Exception as exc:
        logger.warning("Failed to write plots index to %s: %s", index_path, exc)

    return saved_files


def compute_metrics_for_sample(visualizations_data: Dict[str, Any], model, output_dir: str, sample_id: int) -> Optional[pd.DataFrame]:
    """Compute metrics per combination using cf_eval.metrics when available.

    Returns a DataFrame of metrics or None if not available.
    """
    if not CF_EVAL_AVAILABLE:
        logger.warning("cf_eval.metrics not available; skipping metrics computation.")
        return None

    FEATURES_NAMES = visualizations_data['features_names']
    original_sample = visualizations_data['original_sample']
    target_class = visualizations_data['target_class']

    x_original = np.array([original_sample[feat] for feat in FEATURES_NAMES])
    y_original = int(model.predict(x_original.reshape(1, -1))[0])

    all_metrics = []
    for combination_idx, combination_viz in enumerate(visualizations_data['visualizations']):
        cf_list = []
        for replication_viz in combination_viz['replication']:
            cf_dict = replication_viz['counterfactual']
            cf_array = np.array([cf_dict[feat] for feat in FEATURES_NAMES])
            cf_list.append(cf_array)

        if len(cf_list) == 0:
            continue

        cf_array = np.array(cf_list)

        num_valid = nbr_valid_cf(cf_array, model, y_original, y_desidered=target_class)
        pct_valid = perc_valid_cf(cf_array, model, y_original, y_desidered=target_class)

        continuous_features = list(range(len(FEATURES_NAMES)))
        avg_distance = continuous_distance(x_original, cf_array, continuous_features, metric='euclidean')
        min_distance = continuous_distance(x_original, cf_array, continuous_features, metric='euclidean', agg='min')
        max_distance = continuous_distance(x_original, cf_array, continuous_features, metric='euclidean', agg='max')

        avg_changes = avg_nbr_changes_per_cf(x_original, cf_array, continuous_features)
        changes_per_cf = nbr_changes_per_cf(x_original, cf_array, continuous_features)

        all_metrics.append({
            'Combination': combination_idx + 1,
            'Rules': str(combination_viz['label']),
            'Num_CFs': len(cf_list),
            'Valid_CFs': num_valid,
            'Validity_%': f"{pct_valid*100:.1f}%",
            'Avg_Distance': f"{avg_distance:.4f}",
            'Min_Distance': f"{min_distance:.4f}",
            'Max_Distance': f"{max_distance:.4f}",
            'Avg_Changes': f"{avg_changes:.2f}",
            'Changes_Detail': [f"{c:.1f}" for c in changes_per_cf],
        })

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


def run_visualization(sample_id: Optional[int] = None,
                      output_dir: str = DEFAULT_OUTPUT_DIR,
                      export_plots: bool = False,
                      save_plots: bool = False,
                      save_summary: bool = True,
                      save_metrics: bool = True,
                      export_dir: Optional[str] = None,
                      model=None,
                      verbose: bool = False) -> Dict[str, Any]:
    """Main entrypoint. Loads a sample and optionally exports summary/plots/metrics.

    Notes:
    - `export_plots` controls whether an index of available plots is generated.
    - `save_plots` controls whether PNGs are actually written to disk (default: False).

    If sample_id is None, the latest available sample will be chosen.
    """
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        logging.basicConfig(level=logging.INFO)

    available = list_available_samples(output_dir)
    if not available:
        raise RuntimeError(f"No available samples found in {output_dir}")

    if sample_id is None:
        sample_id = max(available.keys())

    if sample_id not in available:
        raise ValueError(f"Sample {sample_id} not found. Available: {list(available.keys())}")

    # Load visualizations
    viz_data = load_visualizations_data(sample_id, output_dir)

    results: Dict[str, Any] = {'sample_id': sample_id}

    # Summary
    if save_summary:
        summary_df = build_summary_table(viz_data)
        sample_dir = os.path.join(output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        csv_path = os.path.join(sample_dir, "summary.csv")
        html_path = os.path.join(sample_dir, "summary.html")
        summary_df.to_csv(csv_path, index=False)
        try:
            summary_df.to_html(html_path, escape=False)
        except Exception:
            pass
        results['summary_csv'] = csv_path
        results['summary_html'] = html_path

    # Plots
    if export_plots:
        saved = export_plots_for_sample(viz_data, sample_id, output_dir, export_dir=export_dir, save_plots=save_plots)
        results['saved_plots'] = saved

    # Metrics
    if save_metrics:
        try:
            metrics_df = compute_metrics_for_sample(viz_data, model, output_dir, sample_id)
            if metrics_df is not None:
                sample_dir = os.path.join(output_dir, str(sample_id))
                os.makedirs(sample_dir, exist_ok=True)
                metrics_csv = os.path.join(sample_dir, "metrics.csv")
                metrics_df.to_csv(metrics_csv, index=False)
                results['metrics_csv'] = metrics_csv
            else:
                results['metrics_csv'] = None
        except Exception as exc:
            logger.warning("Failed to compute metrics: %s", exc)
            results['metrics_csv'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Export visualizations summary and plots for an experiment sample")
    parser.add_argument('--sample-id', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--export-plots', action='store_true', help='Generate an index of available plots for the sample')
    parser.add_argument('--save-plots', action='store_true', help='Actually write PNG files to disk (slow). If not set, an index is written instead')
    parser.add_argument('--no-summary', dest='save_summary', action='store_false')
    parser.add_argument('--no-metrics', dest='save_metrics', action='store_false')
    parser.add_argument('--export-dir', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # If metrics are requested but cf_eval is missing, warn and continue
    if args.save_metrics and not CF_EVAL_AVAILABLE:
        logger.warning('cf_eval.metrics not available. Metrics will be skipped.')

    # Try to import a model if present in workspace; fall back to training a quick one
    model = None
    try:
        # If a pickled model exists (from earlier runs), prefer that
        model_filepath = os.path.join(args.output_dir, 'trained_model.pkl')
        if os.path.exists(model_filepath):
            with open(model_filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            # Quick fallback: train an iris RF like the notebooks did
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            iris = load_iris()
            X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=3, random_state=42)
            model.fit(X_train, y_train)
    except Exception as exc:
        logger.warning('Failed to prepare a model for metrics: %s', exc)
        model = None

    result = run_visualization(sample_id=args.sample_id, output_dir=args.output_dir, export_plots=args.export_plots,
                               save_plots=args.save_plots, save_summary=args.save_summary, save_metrics=args.save_metrics, export_dir=args.export_dir, model=model, verbose=args.verbose)

    print(f"Visualization export complete: {result}")


if __name__ == '__main__':
    main()
