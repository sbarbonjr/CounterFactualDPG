"""Helper functions for generating visualizations in counterfactual experiments.

This module contains extracted visualization code to make the main experiment
runner more maintainable and LLM-friendly.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

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


def create_pca_pairplot(
    features: np.ndarray,
    feature_names_local: List[str],
    original_sample: Dict[str, Any],
    combination_viz: Dict[str, Any],
    sample_id: str,
    original_sample_predicted_class: int,
    target_class: int,
    class_colors_list: List[str],
    output_path: str,
) -> bool:
    """Create PCA pairplot showing evolution in principal component space.
    
    Args:
        features: Full dataset features array
        feature_names_local: List of feature names
        original_sample: Dictionary with original sample values
        combination_viz: Dictionary with replication data including evolution_history
        sample_id: Sample identifier for the plot title
        original_sample_predicted_class: Original predicted class label
        target_class: Target class label
        class_colors_list: List of colors for each class
        output_path: Full path where to save the PCA pairplot PNG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import seaborn as sns
        from sklearn.decomposition import PCA as PCA_pairplot
        from sklearn.preprocessing import StandardScaler as Scaler_pairplot

        # Determine number of PCs (one per feature, max 5)
        n_pcs = min(5, len(feature_names_local))

        if n_pcs < 2:
            print(
                f"INFO: Skipping PCA pairplot - not enough features for multiple PCs"
            )
            return False

        print(
            f"INFO: Creating PCA pairplot with {n_pcs} principal components..."
        )

        # Fit PCA on dataset
        dataset_numeric = pd.DataFrame(
            features, columns=feature_names_local
        ).select_dtypes(include=[np.number])
        scaler_pp = Scaler_pairplot()
        dataset_scaled = scaler_pp.fit_transform(dataset_numeric)

        pca_pp = PCA_pairplot(n_components=n_pcs)
        dataset_pcs = pca_pp.fit_transform(dataset_scaled)

        # Create PC column names
        pc_names = [f"PC{i + 1}" for i in range(n_pcs)]

        # Dataset PCs
        dataset_pc_df = pd.DataFrame(dataset_pcs, columns=pc_names)
        dataset_pc_df["point_type"] = "Dataset"

        # Original sample PCs
        orig_numeric = pd.DataFrame([original_sample])[
            feature_names_local
        ].select_dtypes(include=[np.number])
        orig_scaled = scaler_pp.transform(orig_numeric)
        orig_pcs = pca_pp.transform(orig_scaled)
        orig_pc_df = pd.DataFrame(orig_pcs, columns=pc_names)
        orig_pc_df["point_type"] = "Original"

        # Evolution and counterfactual PCs
        evolution_pc_rows = []
        cf_pc_rows = []

        for rep_idx, rep in enumerate(combination_viz["replication"]):
            evolution_history = rep.get("evolution_history", [])
            if evolution_history:
                for gen_idx, gen_sample in enumerate(evolution_history):
                    gen_numeric = pd.DataFrame([gen_sample])[
                        feature_names_local
                    ].select_dtypes(include=[np.number])
                    gen_scaled = scaler_pp.transform(gen_numeric)
                    gen_pcs = pca_pp.transform(gen_scaled)

                    is_final = gen_idx == len(evolution_history) - 1
                    pc_row = {pc_names[i]: gen_pcs[0, i] for i in range(n_pcs)}
                    pc_row["point_type"] = (
                        "Counterfactual" if is_final else "Evolution"
                    )

                    if is_final:
                        cf_pc_rows.append(pc_row)
                    else:
                        evolution_pc_rows.append(pc_row)

        evolution_pc_df = (
            pd.DataFrame(evolution_pc_rows)
            if evolution_pc_rows
            else pd.DataFrame(columns=pc_names + ["point_type"])
        )
        cf_pc_df = (
            pd.DataFrame(cf_pc_rows)
            if cf_pc_rows
            else pd.DataFrame(columns=pc_names + ["point_type"])
        )

        # Combine all data
        combined_pc_df = pd.concat(
            [dataset_pc_df, evolution_pc_df, orig_pc_df, cf_pc_df],
            ignore_index=True,
        )

        # Define colors
        orig_class_color = class_colors_list[
            original_sample_predicted_class % len(class_colors_list)
        ]
        target_class_color = class_colors_list[
            target_class % len(class_colors_list)
        ]

        # Define full palette and markers mapping
        full_palette_pc = {
            "Dataset": "lightgray",
            "Evolution": "gray",
            "Original": orig_class_color,
            "Counterfactual": target_class_color,
        }
        full_markers_pc = {
            "Dataset": "o",
            "Evolution": ".",
            "Original": "o",
            "Counterfactual": "s",
        }
        full_hue_order_pc = [
            "Dataset",
            "Evolution",
            "Original",
            "Counterfactual",
        ]

        sns.set_style("whitegrid")

        # Determine actual unique point types present in data
        actual_point_types_pc = combined_pc_df["point_type"].unique().tolist()

        # Filter hue_order, palette, and markers to only include actual values
        hue_order_pc = [h for h in full_hue_order_pc if h in actual_point_types_pc]
        palette_pc = {k: full_palette_pc[k] for k in hue_order_pc}
        markers_pc = [full_markers_pc[k] for k in hue_order_pc]

        g_pc = sns.pairplot(
            combined_pc_df,
            hue="point_type",
            hue_order=hue_order_pc,
            palette=palette_pc,
            markers=markers_pc,
            diag_kind="kde",
            plot_kws={"alpha": 0.6},
            diag_kws={"alpha": 0.5, "linewidth": 2},
            corner=False,
            height=2.5,
            aspect=1,
        )

        # Add explained variance to title
        explained_var = pca_pp.explained_variance_ratio_
        var_str = ", ".join(
            [f"PC{i + 1}:{v:.1%}" for i, v in enumerate(explained_var)]
        )

        g_pc.fig.suptitle(
            f"PCA Pairplot - Sample {sample_id}\n"
            f"Original (Class {original_sample_predicted_class}) → Counterfactual (Class {target_class})\n"
            f"Explained Variance: {var_str}",
            y=1.04,
            fontsize=11,
            weight="bold",
        )

        if g_pc._legend:
            g_pc._legend.set_title("Point Type")

        # Save the PCA pairplot
        g_pc.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(g_pc.fig)
        print(f"INFO: Successfully saved PCA pairplot to {output_path}")
        return True

    except Exception as exc:
        print(f"ERROR: Failed to create PCA pairplot: {exc}")
        traceback.print_exc()
        return False
