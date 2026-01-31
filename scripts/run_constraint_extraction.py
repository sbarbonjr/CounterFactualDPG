"""Script to extract and visualize DPG constraints for a dataset.

This script trains a model on a dataset and extracts Decision Predicate Graph (DPG)
constraints, saving both a text file with the constraints and a visualization.

Usage:
  python scripts/run_constraint_extraction.py --dataset german_credit
  python scripts/run_constraint_extraction.py --dataset iris --config-path configs/iris/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import DPG visualization
try:
    from DPG.dpg import plot_dpg_constraints_overview

    DPG_PKG_AVAILABLE = True
except ImportError:
    DPG_PKG_AVAILABLE = False
    print("Warning: DPG package not available. Install with requirements in DPG/")

from ConstraintParser import ConstraintParser
from constraint_scorer import compute_constraint_score
from utils.dataset_loader import load_dataset
from utils.config_manager import load_config


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save DPG constraints for a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., iris, german_credit)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to config file. If not provided, uses configs/{dataset}/config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/constraints",
        help="Base output directory for constraints (default: outputs/constraints)",
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config_path:
        config_path = pathlib.Path(args.config_path)
    else:
        config_path = REPO_ROOT / "configs" / args.dataset / "config.yaml"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"INFO: Loading config from {config_path}")
    config = load_config(str(config_path))
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Output directory: {output_dir}")
    
    # Set random seed
    seed = getattr(config.experiment_params, "seed", 42)
    np.random.seed(seed)
    
    # Load dataset
    print("INFO: Loading dataset...")
    dataset_info = load_dataset(config, repo_root=REPO_ROOT)
    
    FEATURES = dataset_info["features"]
    LABELS = dataset_info["labels"]
    FEATURE_NAMES = dataset_info["feature_names"]
    FEATURES_DF = dataset_info["features_df"]
    
    # Split data
    test_size = getattr(config.data, "test_size", 0.2)
    random_state = getattr(config.data, "random_state", 42)
    
    TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
        FEATURES_DF,
        LABELS,
        test_size=test_size,
        random_state=random_state,
    )
    
    print(f"INFO: Training samples: {len(TRAIN_FEATURES)}, Test samples: {len(TEST_FEATURES)}")
    
    # Train model
    print("INFO: Training model...")
    if config.model.type == "RandomForestClassifier":
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
    
    model.fit(TRAIN_FEATURES, TRAIN_LABELS)
    
    # Log model performance
    train_score = model.score(TRAIN_FEATURES, TRAIN_LABELS)
    test_score = model.score(TEST_FEATURES, TEST_LABELS)
    print(f"INFO: Model trained - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
    
    # Extract constraints
    print("INFO: Extracting DPG constraints...")
    
    # Get DPG config from counterfactual section if available
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
    
    dpg_result = ConstraintParser.extract_constraints_from_dataset(
        model, TRAIN_FEATURES.values, TRAIN_LABELS, FEATURE_NAMES, dpg_config=dpg_config
    )
    constraints = dpg_result["constraints"]
    communities = dpg_result.get("communities", [])
    
    if not constraints:
        print("WARNING: No constraints extracted from DPG")
    else:
        print(f"INFO: Extracted constraints for {len(constraints)} classes")
    
    # Normalize constraints for visualization and saving
    normalized_constraints = ConstraintParser.normalize_constraints(constraints)
    
    # Compute constraint separation score
    constraint_score = None
    n_total_features = len(FEATURE_NAMES)
    n_total_classes = len(np.unique(LABELS))
    if normalized_constraints:
        try:
            score_result = compute_constraint_score(
                normalized_constraints,
                n_total_features=n_total_features,
                n_total_classes=n_total_classes,
                verbose=False,
            )
            constraint_score = score_result["score"]
            print(f"INFO: Constraint quality score: {constraint_score:.4f}")
            print(f"      Coverage: {score_result['coverage_score']:.4f} (features={score_result['n_features']}/{n_total_features}, classes={score_result['n_classes']}/{n_total_classes})")
            print(f"      Separation: {score_result['separation_score']:.4f}")
        except Exception as exc:
            print(f"WARNING: Failed to compute constraint score: {exc}")
    
    # Save constraints as text file
    text_output_path = output_dir / "constraints.txt"
    print(f"INFO: Saving constraints to {text_output_path}")
    save_constraints_text(constraints, text_output_path, constraint_score=constraint_score)
    
    # Save constraints as JSON file
    json_output_path = output_dir / "constraints.json"
    print(f"INFO: Saving constraints to {json_output_path}")
    save_constraints_json(normalized_constraints, json_output_path)
    
    # Generate and save visualization
    if normalized_constraints and DPG_PKG_AVAILABLE:
        print("INFO: Generating DPG constraints overview visualization...")
        
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
        
        viz_output_path = output_dir / "constraints_overview.png"
        
        try:
            constraints_fig = plot_dpg_constraints_overview(
                normalized_constraints=normalized_constraints,
                feature_names=FEATURE_NAMES,
                class_colors_list=class_colors_list,
                output_path=str(viz_output_path),
                title=f"DPG Constraints Overview - {args.dataset}",
                constraint_score=constraint_score,
            )
            
            if constraints_fig:
                print(f"INFO: Saved constraints overview to {viz_output_path}")
                plt.close(constraints_fig)
        except Exception as exc:
            print(f"ERROR: Failed to generate constraints overview: {exc}")
            import traceback
            traceback.print_exc()
    elif not DPG_PKG_AVAILABLE:
        print("WARNING: DPG package not available, skipping visualization")
    
    print(f"\nINFO: Constraint extraction complete!")
    print(f"  - Text file: {text_output_path}")
    print(f"  - JSON file: {json_output_path}")
    if normalized_constraints and DPG_PKG_AVAILABLE:
        print(f"  - Visualization: {viz_output_path}")


if __name__ == "__main__":
    main()
