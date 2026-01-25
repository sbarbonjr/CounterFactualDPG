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


def create_radar_chart(
    features_for_radar: List[str],
    final_cf: Dict[str, Any],
    original_sample: Dict[str, Any],
    constraints: Dict[str, Any],
    original_sample_predicted_class: int,
    target_class: int,
    class_colors_list: List[str],
    output_path: str,
) -> bool:
    """Create radar chart showing feature changes with DPG constraints.
    
    Args:
        features_for_radar: List of feature names to include in radar chart
        final_cf: Dictionary with final counterfactual sample values
        original_sample: Dictionary with original sample values
        constraints: DPG constraints dictionary (by class)
        original_sample_predicted_class: Original predicted class label
        target_class: Target class label
        class_colors_list: List of colors for each class
        output_path: Full path where to save the radar chart PNG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not final_cf or len(features_for_radar) == 0:
            print(
                f"INFO: Skipping radar chart - no counterfactual or actionable features"
            )
            return False

        print(
            f"INFO: Creating radar chart for {len(features_for_radar)} features with changes..."
        )

        # Extract DPG constraints for radar chart
        orig_class_key = f"Class {original_sample_predicted_class}"
        target_class_key = f"Class {target_class}"

        orig_constraints = {}
        target_constraints = {}

        if orig_class_key in constraints:
            for constraint in constraints[orig_class_key]:
                feat = constraint.get("feature")
                if feat:
                    orig_constraints[feat] = {
                        "min": constraint.get("min"),
                        "max": constraint.get("max"),
                    }

        if target_class_key in constraints:
            for constraint in constraints[target_class_key]:
                feat = constraint.get("feature")
                if feat:
                    target_constraints[feat] = {
                        "min": constraint.get("min"),
                        "max": constraint.get("max"),
                    }

        # Prepare data for radar chart using filtered features
        categories = features_for_radar
        original_values = [original_sample.get(f, 0) for f in categories]
        cf_values = [final_cf.get(f, 0) for f in categories]

        # Collect constraint bounds for normalization
        constraint_values = []
        for f in categories:
            if f in orig_constraints:
                if orig_constraints[f]["min"] is not None:
                    constraint_values.append(orig_constraints[f]["min"])
                if orig_constraints[f]["max"] is not None:
                    constraint_values.append(orig_constraints[f]["max"])
            if f in target_constraints:
                if target_constraints[f]["min"] is not None:
                    constraint_values.append(target_constraints[f]["min"])
                if target_constraints[f]["max"] is not None:
                    constraint_values.append(target_constraints[f]["max"])

        # Normalize values to 0-1 range for better visualization
        all_values = original_values + cf_values + constraint_values
        min_val = min(all_values) if all_values else 0
        max_val = max(all_values) if all_values else 1
        value_range = max_val - min_val if max_val > min_val else 1.0

        # Normalize and clip to [0, 1] to avoid floating-point precision errors
        original_norm = [
            np.clip((v - min_val) / value_range, 0.0, 1.0)
            for v in original_values
        ]
        cf_norm = [
            np.clip((v - min_val) / value_range, 0.0, 1.0) for v in cf_values
        ]

        # Normalize constraint bounds
        orig_constraint_min = []
        orig_constraint_max = []
        target_constraint_min = []
        target_constraint_max = []

        for f in categories:
            # Original class constraints
            if f in orig_constraints:
                if orig_constraints[f]["min"] is not None:
                    orig_constraint_min.append(
                        np.clip(
                            (orig_constraints[f]["min"] - min_val) / value_range,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    orig_constraint_min.append(None)
                if orig_constraints[f]["max"] is not None:
                    orig_constraint_max.append(
                        np.clip(
                            (orig_constraints[f]["max"] - min_val) / value_range,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    orig_constraint_max.append(None)
            else:
                orig_constraint_min.append(None)
                orig_constraint_max.append(None)

            # Target class constraints
            if f in target_constraints:
                if target_constraints[f]["min"] is not None:
                    target_constraint_min.append(
                        np.clip(
                            (target_constraints[f]["min"] - min_val)
                            / value_range,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    target_constraint_min.append(None)
                if target_constraints[f]["max"] is not None:
                    target_constraint_max.append(
                        np.clip(
                            (target_constraints[f]["max"] - min_val)
                            / value_range,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    target_constraint_max.append(None)
            else:
                target_constraint_min.append(None)
                target_constraint_max.append(None)

        # Create radar chart
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the plot by appending first value
        original_norm += original_norm[:1]
        cf_norm += cf_norm[:1]
        angles += angles[:1]

        fig_radar, ax_radar = plt.subplots(
            figsize=(12, 12), subplot_kw=dict(projection="polar")
        )

        # Plot original and counterfactual
        orig_class_color = class_colors_list[
            original_sample_predicted_class % len(class_colors_list)
        ]
        target_class_color = class_colors_list[
            target_class % len(class_colors_list)
        ]

        # Plot constraint bounds as shaded regions
        for idx, (angle, orig_min, orig_max, target_min, target_max) in enumerate(
            zip(
                angles[:-1],
                orig_constraint_min,
                orig_constraint_max,
                target_constraint_min,
                target_constraint_max,
            )
        ):
            # Original class constraint region
            if orig_min is not None and orig_max is not None:
                ax_radar.plot(
                    [angle, angle],
                    [orig_min, orig_max],
                    color=orig_class_color,
                    linewidth=4,
                    alpha=0.4,
                    linestyle="-",
                    zorder=1,
                )
            elif orig_min is not None:
                ax_radar.plot(
                    [angle, angle],
                    [orig_min, 1.0],
                    color=orig_class_color,
                    linewidth=4,
                    alpha=0.3,
                    linestyle=":",
                    zorder=1,
                )
            elif orig_max is not None:
                ax_radar.plot(
                    [angle, angle],
                    [0.0, orig_max],
                    color=orig_class_color,
                    linewidth=4,
                    alpha=0.3,
                    linestyle=":",
                    zorder=1,
                )

            # Target class constraint region
            if target_min is not None and target_max is not None:
                ax_radar.plot(
                    [angle, angle],
                    [target_min, target_max],
                    color=target_class_color,
                    linewidth=4,
                    alpha=0.4,
                    linestyle="-",
                    zorder=1,
                )
            elif target_min is not None:
                ax_radar.plot(
                    [angle, angle],
                    [target_min, 1.0],
                    color=target_class_color,
                    linewidth=4,
                    alpha=0.3,
                    linestyle=":",
                    zorder=1,
                )
            elif target_max is not None:
                ax_radar.plot(
                    [angle, angle],
                    [0.0, target_max],
                    color=target_class_color,
                    linewidth=4,
                    alpha=0.3,
                    linestyle=":",
                    zorder=1,
                )

        ax_radar.plot(
            angles,
            original_norm,
            "o-",
            linewidth=2,
            label=f"Original (Class {original_sample_predicted_class})",
            color=orig_class_color,
            markersize=8,
            zorder=3,
        )
        ax_radar.fill(
            angles, original_norm, alpha=0.15, color=orig_class_color, zorder=2
        )

        ax_radar.plot(
            angles,
            cf_norm,
            "s-",
            linewidth=2,
            label=f"Counterfactual (Class {target_class})",
            color=target_class_color,
            markersize=8,
            zorder=3,
        )
        ax_radar.fill(
            angles, cf_norm, alpha=0.15, color=target_class_color, zorder=2
        )

        # Add numerical value annotations
        for idx, (
            angle,
            orig_val,
            cf_val,
            orig_norm_val,
            cf_norm_val,
            orig_min,
            orig_max,
            target_min,
            target_max,
        ) in enumerate(
            zip(
                angles[:-1],
                original_values,
                cf_values,
                original_norm[:-1],
                cf_norm[:-1],
                orig_constraint_min,
                orig_constraint_max,
                target_constraint_min,
                target_constraint_max,
            )
        ):
            # Annotate original value
            ax_radar.text(
                angle,
                orig_norm_val + 0.05,
                f"{orig_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=orig_class_color,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=orig_class_color,
                    alpha=0.7,
                    linewidth=0.5,
                ),
                zorder=4,
            )

            # Annotate counterfactual value
            ax_radar.text(
                angle,
                cf_norm_val - 0.05,
                f"{cf_val:.2f}",
                ha="center",
                va="top",
                fontsize=6,
                color=target_class_color,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=target_class_color,
                    alpha=0.7,
                    linewidth=0.5,
                ),
                zorder=4,
            )

            # Annotate original class constraints
            if orig_min is not None:
                orig_min_real = orig_constraints[categories[idx]]["min"]
                ax_radar.text(
                    angle,
                    orig_min - 0.08,
                    f"min:{orig_min_real:.2f}",
                    ha="center",
                    va="top",
                    fontsize=5,
                    color=orig_class_color,
                    alpha=0.8,
                    style="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=orig_class_color,
                        alpha=0.5,
                        linewidth=0.3,
                    ),
                    zorder=2,
                )

            if orig_max is not None:
                orig_max_real = orig_constraints[categories[idx]]["max"]
                ax_radar.text(
                    angle,
                    orig_max + 0.08,
                    f"max:{orig_max_real:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color=orig_class_color,
                    alpha=0.8,
                    style="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=orig_class_color,
                        alpha=0.5,
                        linewidth=0.3,
                    ),
                    zorder=2,
                )

            # Annotate target class constraints (offset slightly to avoid overlap)
            if target_min is not None:
                target_min_real = target_constraints[categories[idx]]["min"]
                # Offset angle slightly for readability
                offset_angle = angle + 0.05
                ax_radar.text(
                    offset_angle,
                    target_min - 0.08,
                    f"min:{target_min_real:.2f}",
                    ha="left",
                    va="top",
                    fontsize=5,
                    color=target_class_color,
                    alpha=0.8,
                    style="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=target_class_color,
                        alpha=0.5,
                        linewidth=0.3,
                    ),
                    zorder=2,
                )

            if target_max is not None:
                target_max_real = target_constraints[categories[idx]]["max"]
                # Offset angle slightly for readability
                offset_angle = angle + 0.05
                ax_radar.text(
                    offset_angle,
                    target_max + 0.08,
                    f"max:{target_max_real:.2f}",
                    ha="left",
                    va="bottom",
                    fontsize=5,
                    color=target_class_color,
                    alpha=0.8,
                    style="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=target_class_color,
                        alpha=0.5,
                        linewidth=0.3,
                    ),
                    zorder=2,
                )

        # Add custom legend entries for constraints
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D(
                [0],
                [0],
                color=orig_class_color,
                linewidth=4,
                alpha=0.4,
                label=f"Class {original_sample_predicted_class} Constraints",
            ),
            Line2D(
                [0],
                [0],
                color=target_class_color,
                linewidth=4,
                alpha=0.4,
                label=f"Class {target_class} Constraints",
            ),
        ]

        # Set category labels
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, size=8)

        # Set radial limits
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=7)
        ax_radar.grid(True, linestyle="--", alpha=0.3)

        # Add legend and title
        handles, labels = ax_radar.get_legend_handles_labels()
        handles.extend(custom_lines)
        ax_radar.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1),
            fontsize=9,
        )
        ax_radar.set_title(
            f"Feature Changes: Original vs Counterfactual\\n(Normalized values with DPG Constraints)",
            size=14,
            weight="bold",
            pad=20,
        )

        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings for complex polar plots
        fig_radar.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig_radar)
        print(f"INFO: Successfully saved radar chart")
        return True

    except Exception as exc:
        print(f"ERROR: Failed to save radar chart: {exc}")
        traceback.print_exc()
        return False


def create_pairwise_feature_evolution_plot(
    features_to_plot: List[str],
    combination_viz: Dict[str, Any],
    original_sample: Dict[str, Any],
    feature_names_local: List[str],
    model: Any,
    constraints: Dict[str, Any],
    original_sample_predicted_class: int,
    target_class: int,
    class_colors_list: List[str],
    output_path: str,
) -> bool:
    """Create 4D pairwise scatter matrix showing feature evolution with DPG constraints.
    
    Args:
        features_to_plot: List of top feature names to include
        combination_viz: Dictionary with replication data including evolution_history
        original_sample: Dictionary with original sample values
        feature_names_local: All feature names for prediction
        model: Trained model for predicting class of generations
        constraints: DPG constraints dictionary (by class)
        original_sample_predicted_class: Original predicted class label
        target_class: Target class label
        class_colors_list: List of colors for each class
        output_path: Full path where to save the plot PNG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if len(features_to_plot) == 0:
            print(
                f"INFO: Skipping pairwise plot - no actionable features with changes"
            )
            return False

        print(
            f"INFO: Creating pairwise feature evolution plot for top {len(features_to_plot)} changed features..."
        )
        
        import pandas as pd
        
        n_features = len(features_to_plot)
        fig_4d, axes = plt.subplots(n_features, n_features, figsize=(14, 14))

        # Use class colors from the main visualization
        orig_class_color = class_colors_list[
            original_sample_predicted_class % len(class_colors_list)
        ]
        target_class_color = class_colors_list[
            target_class % len(class_colors_list)
        ]

        # Extract DPG constraints for original and target classes
        orig_class_key = f"Class {original_sample_predicted_class}"
        target_class_key = f"Class {target_class}"

        orig_constraints = {}
        target_constraints = {}

        if orig_class_key in constraints:
            for constraint in constraints[orig_class_key]:
                feat = constraint.get("feature")
                if feat:
                    orig_constraints[feat] = {
                        "min": constraint.get("min"),
                        "max": constraint.get("max"),
                    }

        if target_class_key in constraints:
            for constraint in constraints[target_class_key]:
                feat = constraint.get("feature")
                if feat:
                    target_constraints[feat] = {
                        "min": constraint.get("min"),
                        "max": constraint.get("max"),
                    }

        for i, feat_y in enumerate(features_to_plot):
            for j, feat_x in enumerate(features_to_plot):
                ax = axes[i, j] if n_features > 1 else axes

                if i == j:
                    # Diagonal: show feature name
                    ax.text(
                        0.5,
                        0.5,
                        feat_x,
                        ha="center",
                        va="center",
                        fontsize=10,
                        weight="bold",
                        transform=ax.transAxes,
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis("off")
                else:
                    # Off-diagonal: scatter plot
                    # Plot original sample with its class color
                    ax.scatter(
                        original_sample[feat_x],
                        original_sample[feat_y],
                        marker="o",
                        s=200,
                        c=orig_class_color,
                        edgecolors="black",
                        linewidths=2,
                        zorder=10,
                        alpha=0.7,
                    )
                    ax.text(
                        original_sample[feat_x],
                        original_sample[feat_y],
                        "S",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        weight="bold",
                        zorder=11,
                    )

                    # Plot evolution for each replication
                    for rep_idx, rep in enumerate(combination_viz["replication"]):
                        evolution_history = rep.get("evolution_history", [])
                        if evolution_history:
                            # Plot path (use target class color)
                            x_vals = [
                                gen_sample.get(feat_x, np.nan)
                                for gen_sample in evolution_history
                            ]
                            y_vals = [
                                gen_sample.get(feat_y, np.nan)
                                for gen_sample in evolution_history
                            ]
                            ax.plot(
                                x_vals,
                                y_vals,
                                color=target_class_color,
                                alpha=0.3,
                                linewidth=1,
                                zorder=3,
                            )

                            # Plot generation points colored by predicted class
                            for gen_idx, gen_sample in enumerate(
                                evolution_history
                            ):
                                x_val = gen_sample.get(feat_x, np.nan)
                                y_val = gen_sample.get(feat_y, np.nan)

                                # Predict class for this generation
                                gen_sample_df = pd.DataFrame([gen_sample])[
                                    feature_names_local
                                ]
                                gen_pred_class = int(
                                    model.predict(gen_sample_df)[0]
                                )
                                gen_color = class_colors_list[
                                    gen_pred_class % len(class_colors_list)
                                ]

                                # Determine if this is the last generation (final counterfactual)
                                is_final = gen_idx == len(evolution_history) - 1
                                marker_size = 120 if is_final else 80
                                alpha_val = (
                                    1.0
                                    if is_final
                                    else 0.3
                                    + (
                                        0.5
                                        * gen_idx
                                        / max(1, len(evolution_history) - 1)
                                    )
                                )
                                # Clamp alpha to [0, 1] to avoid floating-point precision errors
                                alpha_val = np.clip(alpha_val, 0.0, 1.0)

                                ax.scatter(
                                    x_val,
                                    y_val,
                                    marker="o",
                                    s=marker_size,
                                    c="none",
                                    edgecolors=gen_color,
                                    linewidths=1.5 if is_final else 1,
                                    alpha=alpha_val,
                                    zorder=5,
                                )

                                # Add label
                                label = "C" if is_final else str(gen_idx + 1)
                                ax.text(
                                    x_val,
                                    y_val,
                                    label,
                                    ha="center",
                                    va="center",
                                    fontsize=7 if is_final else 6,
                                    color=gen_color,
                                    weight="bold",
                                    zorder=6,
                                    alpha=alpha_val,
                                )

                    # Now add constraint boundaries with labels (after data is plotted)
                    # Plot constraint lines/regions
                    # Original class constraints (dashed) - positioned at edges
                    if feat_x in orig_constraints:
                        x_min = orig_constraints[feat_x].get("min")
                        x_max = orig_constraints[feat_x].get("max")
                        if x_min is not None:
                            ax.axvline(
                                x=x_min,
                                color=orig_class_color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.5,
                                zorder=1,
                            )
                            ax.text(
                                x_min,
                                0.98,
                                f"C{original_sample_predicted_class} min={x_min:.2f}",
                                rotation=90,
                                va="top",
                                ha="right",
                                fontsize=4,
                                color=orig_class_color,
                                alpha=0.9,
                                transform=ax.get_xaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=orig_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )
                        if x_max is not None:
                            ax.axvline(
                                x=x_max,
                                color=orig_class_color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.5,
                                zorder=1,
                            )
                            ax.text(
                                x_max,
                                0.98,
                                f"C{original_sample_predicted_class} max={x_max:.2f}",
                                rotation=90,
                                va="top",
                                ha="left",
                                fontsize=4,
                                color=orig_class_color,
                                alpha=0.9,
                                transform=ax.get_xaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=orig_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )

                    if feat_y in orig_constraints:
                        y_min = orig_constraints[feat_y].get("min")
                        y_max = orig_constraints[feat_y].get("max")
                        if y_min is not None:
                            ax.axhline(
                                y=y_min,
                                color=orig_class_color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.5,
                                zorder=1,
                            )
                            ax.text(
                                0.98,
                                y_min,
                                f"C{original_sample_predicted_class} min={y_min:.2f}",
                                rotation=0,
                                va="bottom",
                                ha="right",
                                fontsize=4,
                                color=orig_class_color,
                                alpha=0.9,
                                transform=ax.get_yaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=orig_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )
                        if y_max is not None:
                            ax.axhline(
                                y=y_max,
                                color=orig_class_color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.5,
                                zorder=1,
                            )
                            ax.text(
                                0.98,
                                y_max,
                                f"C{original_sample_predicted_class} max={y_max:.2f}",
                                rotation=0,
                                va="top",
                                ha="right",
                                fontsize=4,
                                color=orig_class_color,
                                alpha=0.9,
                                transform=ax.get_yaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=orig_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )

                    # Target class constraints (solid) - positioned at opposite edges
                    if feat_x in target_constraints:
                        x_min = target_constraints[feat_x].get("min")
                        x_max = target_constraints[feat_x].get("max")
                        if x_min is not None:
                            ax.axvline(
                                x=x_min,
                                color=target_class_color,
                                linestyle="-",
                                linewidth=1.5,
                                alpha=0.6,
                                zorder=2,
                            )
                            ax.text(
                                x_min,
                                0.02,
                                f"C{target_class} min={x_min:.2f}",
                                rotation=90,
                                va="bottom",
                                ha="right",
                                fontsize=4,
                                color=target_class_color,
                                alpha=0.9,
                                transform=ax.get_xaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=target_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )
                        if x_max is not None:
                            ax.axvline(
                                x=x_max,
                                color=target_class_color,
                                linestyle="-",
                                linewidth=1.5,
                                alpha=0.6,
                                zorder=2,
                            )
                            ax.text(
                                x_max,
                                0.02,
                                f"C{target_class} max={x_max:.2f}",
                                rotation=90,
                                va="bottom",
                                ha="left",
                                fontsize=4,
                                color=target_class_color,
                                alpha=0.9,
                                transform=ax.get_xaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=target_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )

                    if feat_y in target_constraints:
                        y_min = target_constraints[feat_y].get("min")
                        y_max = target_constraints[feat_y].get("max")
                        if y_min is not None:
                            ax.axhline(
                                y=y_min,
                                color=target_class_color,
                                linestyle="-",
                                linewidth=1.5,
                                alpha=0.6,
                                zorder=2,
                            )
                            ax.text(
                                0.02,
                                y_min,
                                f"C{target_class} min={y_min:.2f}",
                                rotation=0,
                                va="bottom",
                                ha="left",
                                fontsize=4,
                                color=target_class_color,
                                alpha=0.9,
                                transform=ax.get_yaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=target_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )
                        if y_max is not None:
                            ax.axhline(
                                y=y_max,
                                color=target_class_color,
                                linestyle="-",
                                linewidth=1.5,
                                alpha=0.6,
                                zorder=2,
                            )
                            ax.text(
                                0.02,
                                y_max,
                                f"C{target_class} max={y_max:.2f}",
                                rotation=0,
                                va="top",
                                ha="left",
                                fontsize=4,
                                color=target_class_color,
                                alpha=0.9,
                                transform=ax.get_yaxis_transform(),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    edgecolor=target_class_color,
                                    alpha=0.7,
                                    linewidth=0.5,
                                ),
                            )

                    ax.set_xlabel(
                        feat_x if i == n_features - 1 else "", fontsize=8
                    )
                    ax.set_ylabel(feat_y if j == 0 else "", fontsize=8)
                    ax.tick_params(labelsize=7)

        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings
        fig_4d.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig_4d)
        print(
            f"INFO: Successfully saved pairwise feature evolution plot with {len(features_to_plot)} features"
        )
        return True

    except Exception as exc:
        print(f"ERROR: Failed to save pairwise feature evolution plot: {exc}")
        traceback.print_exc()
        return False
