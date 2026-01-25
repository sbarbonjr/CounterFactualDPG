"""Helper functions for generating visualizations in counterfactual experiments.

This module contains extracted visualization code to make the main experiment
runner more maintainable and LLM-friendly.
"""

from __future__ import annotations

import os
import traceback
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_feature_evolution_pairplot(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names_local: List[str],
    feature_df: pd.DataFrame,
    sample_id: str,
    original_sample_predicted_class: int,
    target_class: int,
    class_colors_list: List[str],
    output_path: str,
) -> bool:
    """Create seaborn pairplot showing feature evolution across generations.
    
    Args:
        features: Full dataset features array
        labels: Full dataset labels array
        feature_names_local: List of feature names
        feature_df: DataFrame with columns: replication, generation, and feature columns
        sample_id: Sample identifier for the plot title
        original_sample_predicted_class: Original predicted class label
        target_class: Target class label
        class_colors_list: List of colors for each class
        output_path: Full path where to save the pairplot PNG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import seaborn as sns

        # Determine which features to include in pairplot
        # Use top features by variance or change, limit to 4-6 for readability
        numeric_features = [
            f for f in feature_names_local if f in feature_df.columns
        ]

        # Calculate variance for each feature across generations
        feature_variance = {}
        for feat in numeric_features:
            if feat in feature_df.columns:
                feature_variance[feat] = feature_df[feat].var()

        # Sort by variance and select top features
        sorted_by_variance = sorted(
            feature_variance.items(),
            key=lambda x: x[1] if x[1] is not None else 0,
            reverse=True,
        )
        max_pairplot_features = 5  # Limit for readability
        pairplot_features = [
            feat for feat, _ in sorted_by_variance[:max_pairplot_features]
        ]

        if len(pairplot_features) < 2:
            print(
                f"INFO: Skipping pairplot - not enough features (need at least 2)"
            )
            return False

        print(
            f"INFO: Creating seaborn pairplot for top {len(pairplot_features)} features by variance..."
        )

        # Build combined dataframe with dataset samples + evolution points
        # 1. Dataset samples (background)
        dataset_df = pd.DataFrame(features, columns=feature_names_local)
        dataset_df["point_type"] = "Dataset"
        dataset_df["predicted_class"] = (
            labels.tolist() if hasattr(labels, "tolist") else list(labels)
        )

        # 2. Evolution data from feature_df
        pairplot_df = feature_df.copy()

        # Create type labels for evolution data
        max_gen = (
            feature_df[feature_df["replication"] != "original"]["generation"].max()
            if len(feature_df[feature_df["replication"] != "original"]) > 0
            else 0
        )

        def get_point_type(row):
            if row["replication"] == "original":
                return "Original"
            elif row["generation"] == max_gen:
                return "Counterfactual"
            else:
                return "Evolution"

        pairplot_df["point_type"] = pairplot_df.apply(get_point_type, axis=1)

        # Combine dataset + evolution data
        combined_df = pd.concat(
            [
                dataset_df[pairplot_features + ["point_type"]],
                pairplot_df[pairplot_features + ["point_type"]],
            ],
            ignore_index=True,
        )

        # Define color palette matching class colors
        orig_class_color = class_colors_list[
            original_sample_predicted_class % len(class_colors_list)
        ]
        target_class_color = class_colors_list[
            target_class % len(class_colors_list)
        ]

        # Define full palette and markers mapping
        full_palette = {
            "Dataset": "lightgray",
            "Original": orig_class_color,
            "Evolution": "gray",
            "Counterfactual": target_class_color,
        }
        full_markers = {
            "Dataset": "o",
            "Original": "o",
            "Evolution": ".",
            "Counterfactual": "s",
        }
        full_hue_order = [
            "Dataset",
            "Evolution",
            "Original",
            "Counterfactual",
        ]

        # Set style
        sns.set_style("whitegrid")

        # Create pairplot with hue_order to control layering
        pairplot_data = combined_df.dropna()

        # Determine actual unique point types present in data
        actual_point_types = pairplot_data["point_type"].unique().tolist()

        # Filter hue_order, palette, and markers to only include actual values
        hue_order = [h for h in full_hue_order if h in actual_point_types]
        palette = {k: full_palette[k] for k in hue_order}
        markers = [full_markers[k] for k in hue_order]

        g = sns.pairplot(
            pairplot_data,
            hue="point_type",
            hue_order=hue_order,
            palette=palette,
            markers=markers,
            diag_kind="kde",
            plot_kws={"alpha": 0.6},
            diag_kws={"alpha": 0.5, "linewidth": 2},
            corner=False,
            height=2.5,
            aspect=1,
        )

        # Customize marker sizes - make Original and Counterfactual larger
        for ax in g.axes.flatten():
            if ax is not None:
                for collection in ax.collections:
                    # Adjust sizes based on label
                    pass  # seaborn handles this internally

        # Customize the plot
        g.fig.suptitle(
            f"Feature Evolution Pairplot - Sample {sample_id}\n"
            f"Original (Class {original_sample_predicted_class}) → Counterfactual (Class {target_class})",
            y=1.02,
            fontsize=12,
            weight="bold",
        )

        # Adjust legend
        if g._legend:
            g._legend.set_title("Point Type")

        # Save the pairplot
        g.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(g.fig)
        print(f"INFO: Successfully saved seaborn pairplot to {output_path}")
        return True

    except Exception as exc:
        print(f"ERROR: Failed to create seaborn pairplot: {exc}")
        traceback.print_exc()
        return False
