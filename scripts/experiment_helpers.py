"""Helper functions for experiment data persistence and WandB logging.

This module contains extracted helper code for saving metrics, logging to WandB,
and persisting experiment results to make the main experiment runner more maintainable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def save_replication_and_combination_metrics(
    sample_id: str,
    sample_dir: str,
    visualizations: List[Dict[str, Any]],
) -> None:
    """Save replication-level and combination-level metrics to CSV files.
    
    Args:
        sample_id: Sample identifier
        sample_dir: Directory to save metrics CSVs
        visualizations: List of visualization data dictionaries with metrics
    """
    try:
        # Collect all replication-level metrics
        replication_metrics_list = []
        for combination_viz in visualizations:
            for replication_idx, replication_viz in enumerate(
                combination_viz.get("replication", [])
            ):
                if "metrics" in replication_viz:
                    metrics_row = {
                        "sample_id": sample_id,
                        "combination": str(combination_viz["label"]),
                        "replication_idx": replication_idx,
                    }
                    metrics_row.update(replication_viz["metrics"])
                    replication_metrics_list.append(metrics_row)

        if replication_metrics_list:
            replication_metrics_df = pd.DataFrame(replication_metrics_list)
            replication_metrics_csv = os.path.join(
                sample_dir, "replication_metrics.csv"
            )
            replication_metrics_df.to_csv(replication_metrics_csv, index=False)
            print(f"INFO: Saved replication metrics to {replication_metrics_csv}")

        # Collect all combination-level metrics
        combination_metrics_list = []
        for combination_viz in visualizations:
            if "comprehensive_metrics" in combination_viz:
                metrics_row = {
                    "sample_id": sample_id,
                    "combination": str(combination_viz["label"]),
                }
                metrics_row.update(combination_viz["comprehensive_metrics"])
                combination_metrics_list.append(metrics_row)

        if combination_metrics_list:
            combination_metrics_df = pd.DataFrame(combination_metrics_list)
            combination_metrics_csv = os.path.join(
                sample_dir, "combination_metrics.csv"
            )
            combination_metrics_df.to_csv(combination_metrics_csv, index=False)
            print(f"INFO: Saved combination metrics to {combination_metrics_csv}")

    except Exception as exc:
        print(f"WARNING: Failed to save metrics CSV files: {exc}")
        import traceback
        traceback.print_exc()


def log_sample_artifacts_to_wandb(
    wandb_run: Any,
    sample_id: str,
    raw_filepath: str,
    viz_filepath: str,
) -> None:
    """Log sample result artifacts to WandB.
    
    Args:
        wandb_run: WandB run object
        sample_id: Sample identifier
        raw_filepath: Path to raw results pickle file
        viz_filepath: Path to visualization data pickle file
    """
    if wandb_run:
        try:
            import wandb
            artifact = wandb.Artifact(f"sample_{sample_id}_results", type="results")
            artifact.add_file(raw_filepath)
            artifact.add_file(viz_filepath)
            wandb.log_artifact(artifact)
        except Exception as exc:
            print(f"WARNING: Failed to log sample artifacts to WandB: {exc}")


def save_experiment_level_metrics(
    results: List[Dict[str, Any]],
    config: Any,
    output_dir: str = "outputs",
    total_generation_runtime: float = None,
) -> None:
    """Save aggregated experiment-level metrics and summary statistics.
    
    Args:
        results: List of per-sample result dictionaries
        config: Experiment configuration object
        output_dir: Root output directory
        total_generation_runtime: Total runtime for counterfactual generation (seconds)
    """
    try:
        experiment_name = getattr(config.experiment, "name", "experiment")
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Aggregate all metrics from all samples
        all_replication_metrics = []
        all_combination_metrics = []

        for r in results:
            sample_dir = r["sample_dir"]

            # Load replication metrics
            replication_csv = os.path.join(sample_dir, "replication_metrics.csv")
            if os.path.exists(replication_csv):
                rep_df = pd.read_csv(replication_csv)
                all_replication_metrics.append(rep_df)

            # Load combination metrics
            combination_csv = os.path.join(sample_dir, "combination_metrics.csv")
            if os.path.exists(combination_csv):
                comb_df = pd.read_csv(combination_csv)
                all_combination_metrics.append(comb_df)

        # Save aggregated metrics
        if all_replication_metrics:
            aggregated_rep_df = pd.concat(all_replication_metrics, ignore_index=True)
            aggregated_rep_csv = os.path.join(
                experiment_dir, "all_replication_metrics.csv"
            )
            aggregated_rep_df.to_csv(aggregated_rep_csv, index=False)
            print(f"INFO: Saved aggregated replication metrics to {aggregated_rep_csv}")

        if all_combination_metrics:
            aggregated_comb_df = pd.concat(all_combination_metrics, ignore_index=True)
            aggregated_comb_csv = os.path.join(
                experiment_dir, "all_combination_metrics.csv"
            )
            aggregated_comb_df.to_csv(aggregated_comb_csv, index=False)
            print(
                f"INFO: Saved aggregated combination metrics to {aggregated_comb_csv}"
            )

            # Compute and save summary statistics
            numeric_cols = aggregated_comb_df.select_dtypes(include=[np.number]).columns
            summary_stats = aggregated_comb_df[numeric_cols].describe()
            summary_stats_csv = os.path.join(
                experiment_dir, "metrics_summary_statistics.csv"
            )
            summary_stats.to_csv(summary_stats_csv)
            print(f"INFO: Saved summary statistics to {summary_stats_csv}")

        # Save experiment-level summary with runtime
        summary_data = {
            "total_samples": len(results),
            "total_valid_counterfactuals": sum(r["valid_counterfactuals"] for r in results),
            "total_requested_counterfactuals": sum(r["requested_counterfactuals"] for r in results),
            "overall_success_rate": np.mean([r["success_rate"] for r in results]),
        }
        if total_generation_runtime is not None:
            summary_data["total_generation_runtime"] = total_generation_runtime
        
        summary_df = pd.DataFrame([summary_data])
        summary_csv = os.path.join(experiment_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"INFO: Saved experiment summary to {summary_csv}")
        
        # Save experiment configuration
        config_copy_path = os.path.join(experiment_dir, "experiment_config.yaml")
        with open(config_copy_path, "w") as f:
            yaml.dump(config.to_dict(), f)
        print(f"INFO: Saved experiment config to {config_copy_path}")

    except Exception as exc:
        print(f"WARNING: Failed to save experiment-level metrics: {exc}")
        import traceback
        traceback.print_exc()


def log_experiment_summary_to_wandb(
    wandb_run: Any,
    results: List[Dict[str, Any]],
    total_valid: int,
    total_requested: int,
    total_success_rate: float,
    total_generation_runtime: float = None,
) -> None:
    """Log experiment summary statistics and table to WandB.
    
    Args:
        wandb_run: WandB run object
        results: List of per-sample result dictionaries
        total_valid: Total number of valid counterfactuals
        total_requested: Total number of requested counterfactuals
        total_success_rate: Overall success rate (0-1)
        total_generation_runtime: Total runtime for counterfactual generation (seconds)
    """
    if wandb_run:
        try:
            import wandb
            
            # Log experiment-level summary (single values for the entire run)
            wandb.run.summary["experiment/total_samples"] = len(results)
            wandb.run.summary["experiment/total_valid_counterfactuals"] = total_valid
            wandb.run.summary["experiment/total_requested_counterfactuals"] = total_requested
            wandb.run.summary["experiment/overall_success_rate"] = total_success_rate
            
            # Log runtime metric (used by compare_techniques.py)
            if total_generation_runtime is not None:
                wandb.run.summary["metrics/combination/runtime"] = total_generation_runtime
                wandb.run.summary["runtime"] = total_generation_runtime
            
            summary_data = [
                [
                    r["sample_id"],
                    r["valid_counterfactuals"],
                    r["requested_counterfactuals"],
                    f"{r['success_rate']:.2%}",
                ]
                for r in results
            ]

            summary_table = wandb.Table(
                columns=["Sample ID", "Valid CFs", "Requested CFs", "Success Rate"],
                data=summary_data,
            )
            wandb.log({"experiment/summary_table": summary_table})
        except Exception as exc:
            print(f"WARNING: Failed to log experiment summary to WandB: {exc}")
