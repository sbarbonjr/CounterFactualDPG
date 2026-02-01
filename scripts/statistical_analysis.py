#!/usr/bin/env python3
"""Statistical analysis functions for DPG vs DiCE comparison.

This module extracts functionality from notebooks/DPG_CF_statistics.ipynb
to enable statistical analysis and visualization of comparison results.

Main functions:
- load_and_clean_comparison_data: Load and clean comparison CSV data
- compute_difference_matrix: Calculate DPG - DiCE difference matrix
- wilcoxon_statistical_test: Run Wilcoxon signed-rank tests per metric
- generate_latex_table: Generate LaTeX table for paper
- plot_heatmap_differences: Create heatmap of normalized differences
- plot_metric_scatter: Create per-dataset scatter plot
- plot_pairwise_comparison: Create pairwise comparison scatter plot
- plot_median_difference_bars: Create bar chart of median differences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from scipy.stats import wilcoxon
from typing import Dict, List, Optional, Tuple, Any


# Default suffixes for DPG and DiCE columns
SUFFIX_DPG = "_DPG"
SUFFIX_DICE = "_DiCE"

# Metric name mapping (internal -> pretty)
METRIC_NAME_MAP = {
    "plausibility_nbr_cf": "Implausibility",
    "count_diversity_all": "Diversity",
    "avg_nbr_changes": "Sparsity",
    "accuracy_knn_sklearn": "Discriminative Power",
    "runtime": "Runtime",
    "distance_mh": "Distance",
    "perc_valid_cf_all": "Validity",
    "perc_actionable_cf_all": "Actionability",
}

# Metric goal mapping (internal -> (arrow, direction))
# direction: "higher" means higher is better, "lower" means lower is better
METRIC_GOAL_MAP = {
    "plausibility_nbr_cf": ("↓", "lower"),
    "count_diversity_all": ("↑", "higher"),
    "avg_nbr_changes": ("↓", "lower"),
    "accuracy_knn_sklearn": ("↑", "higher"),
    "runtime": ("↓", "lower"),
    "distance_mh": ("↓", "lower"),
    "perc_valid_cf_all": ("↑", "higher"),
    "perc_actionable_cf_all": ("↑", "higher"),
}


def load_and_clean_comparison_data(
    csv_path: str,
    bad_datasets: Optional[List[str]] = None,
    drop_nan_rows: bool = True,
) -> pd.DataFrame:
    """Load comparison CSV and clean data.

    Args:
        csv_path: Path to comparison_numeric_small.csv file.
        bad_datasets: List of dataset names to exclude (default: None).
        drop_nan_rows: Whether to drop rows with NaN values in metric columns.

    Returns:
        Cleaned DataFrame with 'Dataset' as index.
    """
    df = pd.read_csv(csv_path)

    # Default datasets to exclude
    if bad_datasets is None:
        bad_datasets = ["breast_cancer_wisconsin", "abalone_19", "wheat-seeds", "heart_disease_uci"]

    # Remove specific datasets
    df = df[~df["Dataset"].isin(bad_datasets)]

    # Identify metric columns (ones with _DPG / _DiCE suffixes)
    metric_cols = [
        c for c in df.columns if c.endswith(SUFFIX_DPG) or c.endswith(SUFFIX_DICE)
    ]

    # Drop rows with any NaN in metric columns
    if drop_nan_rows:
        df = df.dropna(subset=metric_cols)

    # Set Dataset as index
    df = df.set_index("Dataset")

    return df


def detect_metrics(df: pd.DataFrame) -> List[str]:
    """Detect metric base names present for both DPG and DiCE.

    Args:
        df: DataFrame with metric columns.

    Returns:
        List of metric base names (sorted).
    """
    cols_dpg = [c for c in df.columns if c.endswith(SUFFIX_DPG)]
    cols_dice = [c for c in df.columns if c.endswith(SUFFIX_DICE)]

    metrics = sorted(
        set(c.replace(SUFFIX_DPG, "") for c in cols_dpg)
        & set(c.replace(SUFFIX_DICE, "") for c in cols_dice)
    )

    return metrics


def compute_difference_matrix(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute difference matrix (DPG - DiCE) and normalized version.

    Args:
        df: DataFrame with metric columns indexed by Dataset.
        metrics: List of metric base names (auto-detected if None).

    Returns:
        Tuple of (raw difference matrix, normalized difference matrix [-1, 1]).
    """
    if metrics is None:
        metrics = detect_metrics(df)

    # Build matrix of differences (DPG - DiCE)
    diff_mat = pd.DataFrame(index=df.index)
    for metric in metrics:
        col_dpg = metric + SUFFIX_DPG
        col_dice = metric + SUFFIX_DICE
        if col_dpg in df.columns and col_dice in df.columns:
            diff_mat[metric] = df[col_dpg] - df[col_dice]

    # Column-wise normalization to [-1, 1] by max absolute value per metric
    diff_mat_norm = diff_mat.copy()
    for col in diff_mat_norm.columns:
        max_abs = np.nanmax(np.abs(diff_mat_norm[col].values))
        if max_abs > 0:
            diff_mat_norm[col] = diff_mat_norm[col] / max_abs
        else:
            diff_mat_norm[col] = 0.0

    return diff_mat, diff_mat_norm


def holm_bonferroni(p_vals: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni correction to p-values.

    Args:
        p_vals: Dictionary mapping metric name to raw p-value.

    Returns:
        Dictionary mapping metric name to corrected p-value.
    """
    items = sorted(p_vals.items(), key=lambda x: x[1])
    m = len(items)
    corrected_internal = {}
    prev = 0.0

    for i, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - i + 1) * p)
        adj = max(adj, prev)  # ensure monotonic
        prev = adj
        corrected_internal[name] = adj

    return {name: corrected_internal[name] for name in p_vals.keys()}


def wilcoxon_statistical_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run Wilcoxon signed-rank test per metric.

    Args:
        df: DataFrame with metric columns indexed by Dataset.

    Returns:
        DataFrame with test results including p-values and effect sizes.
    """
    metrics = detect_metrics(df)
    results = []
    raw_pvals = {}

    for metric in metrics:
        col_dpg = metric + SUFFIX_DPG
        col_dice = metric + SUFFIX_DICE

        x = df[col_dpg]
        y = df[col_dice]

        # Drop NaN pairs
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask].values
        y_clean = y[mask].values

        if len(x_clean) < 1:
            continue

        # Wilcoxon signed-rank test
        stat, p = wilcoxon(
            x_clean, y_clean,
            zero_method='wilcox',
            alternative='two-sided'
        )

        # Handle NaN p-value (occurs when all differences are zero)
        if np.isnan(p):
            p = 1.0

        diff = x_clean - y_clean
        median_diff = np.median(diff)

        # Effect size: (n_pos - n_neg) / (n_pos + n_neg)
        non_zero = diff[diff != 0]
        n_pos = np.sum(non_zero > 0)
        n_neg = np.sum(non_zero < 0)
        if (n_pos + n_neg) > 0:
            effect_sign = (n_pos - n_neg) / (n_pos + n_neg)
        else:
            effect_sign = 0.0

        raw_pvals[metric] = p

        results.append({
            "metric": metric,
            "n_datasets_used": len(x_clean),
            "median_diff(DPG - DiCE)": median_diff,
            "wilcoxon_stat": stat,
            "p_value_raw": p,
            "effect_size_sign": effect_sign
        })

    if not results:
        return pd.DataFrame()

    # Apply Holm-Bonferroni correction
    corrected = holm_bonferroni(raw_pvals)
    for r in results:
        r["p_value_holm"] = corrected[r["metric"]]

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("p_value_raw")

    # Round for readability
    for col in ["median_diff(DPG - DiCE)", "p_value_raw", "p_value_holm", "effect_size_sign"]:
        if col in res_df.columns:
            res_df[col] = res_df[col].round(4)

    return res_df


def generate_latex_table(
    res_df: pd.DataFrame,
    caption_prefix: str = "Paired Wilcoxon signed-rank comparison",
) -> str:
    """Generate LaTeX table for paper from statistical test results.

    Args:
        res_df: DataFrame with test results from wilcoxon_statistical_test().
        caption_prefix: Prefix for the caption text.

    Returns:
        LaTeX table as string.
    """
    # Sort by pretty name alphabetically
    res_df_sorted = res_df.copy()
    res_df_sorted["pretty_name"] = res_df_sorted["metric"].map(METRIC_NAME_MAP)
    res_df_sorted = res_df_sorted.sort_values("pretty_name")

    n_datasets = int(res_df_sorted.iloc[0]["n_datasets_used"]) if len(res_df_sorted) > 0 else 0

    # Build LaTeX table rows
    latex_rows = []

    for _, row in res_df_sorted.iterrows():
        metric = row["metric"]
        pretty_name = METRIC_NAME_MAP.get(metric, metric)
        goal_symbol_display, goal_direction = METRIC_GOAL_MAP.get(metric, ("", ""))
        
        # Convert display symbol to LaTeX
        goal_symbol_latex = "\\downarrow" if goal_symbol_display == "↓" else "\\uparrow"

        median_diff = row["median_diff(DPG - DiCE)"]
        p_value = row["p_value_raw"]

        is_significant = p_value < 0.05

        if abs(median_diff) < 1e-6:
            best_method = "Both"
        elif goal_direction == "higher":
            best_method = "DPG-CF" if median_diff > 0 else "DiCE"
        else:
            best_method = "DPG-CF" if median_diff < 0 else "DiCE"

        if is_significant and best_method != "Both":
            best_method = f"\\textbf{{{best_method}}}"

        latex_row = f"    \\texttt{{{pretty_name}}}  & ${goal_symbol_latex}$ & {median_diff:6.2f}  & {p_value:.2f} & {best_method}            \\\\"
        latex_rows.append(latex_row)

    latex_table = f"""\\begin{{table}}[b!]
  \\centering
  \\caption{{{caption_prefix} between DPG-CF and DiCE across
  {n_datasets} datasets. $\\Delta = \\text{{DPG-CF}} - \\text{{DiCE}}$. The ``Goal'' column indicates
  whether higher ($\\uparrow$) or lower ($\\downarrow$) values are desirable. The
  ``Best'' column reports the method with better median performance according to
  this goal; method names in bold denote statistically significant differences
  (Wilcoxon test, $p < 0.05$). All metrics computed using $n = {n_datasets}$ datasets.}}
  \\label{{tab:wilcoxon_results}}
  \\begin{{tabular}}{{lcccl}}
    \\toprule
    Metric & Goal & median $\\Delta$ & $p$-value & Best \\\\
    \\midrule
"""

    for row in latex_rows:
        latex_table += row + "\n"

    latex_table += r"""    \bottomrule
  \end{tabular}
\end{table}"""

    return latex_table


def plot_heatmap_differences(
    diff_mat: pd.DataFrame,
    diff_mat_norm: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "",
) -> matplotlib.figure.Figure:
    """Create heatmap of normalized differences with raw values as annotations.

    Args:
        diff_mat: Raw difference matrix (DPG - DiCE).
        diff_mat_norm: Normalized difference matrix [-1, 1].
        output_path: Path to save figure (PDF/PNG).
        figsize: Figure size.
        title: Title for the plot.

    Returns:
        matplotlib Figure object.
    """
    # Keep only metrics present and sort by pretty name
    metrics_present = [m for m in METRIC_NAME_MAP.keys() if m in diff_mat.columns]
    base_metrics_sorted = sorted(metrics_present, key=lambda m: METRIC_NAME_MAP[m])

    diff_mat_plot = diff_mat[base_metrics_sorted]
    diff_mat_norm_plot = diff_mat_norm[base_metrics_sorted]

    # Build x-axis labels with arrows
    x_labels = [
        f"{METRIC_NAME_MAP[m]} {METRIC_GOAL_MAP[m][0]}"
        for m in base_metrics_sorted
    ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        diff_mat_norm_plot,
        annot=diff_mat_plot.round(2),
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Column-wise scaled difference"},
        ax=ax
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Metric (goal)", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, format=output_path.split(".")[-1], bbox_inches="tight")

    return fig


def plot_metric_scatter(
    df: pd.DataFrame,
    res_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> matplotlib.figure.Figure:
    """Create scatter plot of per-dataset differences grouped by metric.

    Args:
        df: DataFrame with metric columns indexed by Dataset.
        res_df: DataFrame with test results (used for metric ordering).
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    metrics = detect_metrics(df)
    
    # Build long-form dataframe of differences
    rows = []
    for metric in metrics:
        col_dpg = metric + SUFFIX_DPG
        col_dice = metric + SUFFIX_DICE
        if col_dpg not in df.columns or col_dice not in df.columns:
            continue
        diffs = df[col_dpg] - df[col_dice]
        for dataset_name, dval in zip(df.index, diffs):
            rows.append({"metric": metric, "dataset": dataset_name, "diff": dval})

    diff_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    metrics_order = res_df["metric"].tolist() if len(res_df) > 0 else metrics
    positions = {m: i for i, m in enumerate(metrics_order)}

    for m in metrics_order:
        subset = diff_df[diff_df["metric"] == m]
        x = np.full(len(subset), positions[m])
        ax.scatter(x, subset["diff"], alpha=0.7)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(metrics_order)))
    ax.set_xticklabels(metrics_order, rotation=45, ha="right")
    ax.set_ylabel("Difference per dataset (DPG - DiCE)")
    ax.set_title("Per-dataset metric differences grouped by metric")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pairwise_comparison(
    df: pd.DataFrame,
    metric: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> matplotlib.figure.Figure:
    """Create pairwise comparison scatter plot for a specific metric.

    Points above the diagonal indicate DPG is better (for higher-is-better metrics)
    or worse (for lower-is-better metrics).

    Args:
        df: DataFrame with metric columns indexed by Dataset.
        metric: Base name of metric to plot.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    col_dpg = metric + SUFFIX_DPG
    col_dice = metric + SUFFIX_DICE

    x_dice = np.asarray(df[col_dice].values)
    x_dpg = np.asarray(df[col_dpg].values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_dice, x_dpg)

    # Diagonal line (equal performance)
    min_val = min(np.nanmin(x_dice), np.nanmin(x_dpg))
    max_val = max(np.nanmax(x_dice), np.nanmax(x_dpg))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")

    # Optional: draw lines from diagonal to points
    for xd, xp in zip(x_dice, x_dpg):
        ax.plot([xd, xd, xp], [xd, xp, xp], alpha=0.3, color="gray")

    ax.set_xlabel("DiCE")
    ax.set_ylabel("DPG-CF")
    pretty_name = METRIC_NAME_MAP.get(metric, metric)
    ax.set_title(f"{pretty_name}: per-dataset scores (DPG vs DiCE)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_median_difference_bars(
    res_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> matplotlib.figure.Figure:
    """Create bar chart of median differences with significance markers.

    Args:
        res_df: DataFrame with test results from wilcoxon_statistical_test().
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    res_df_plot = res_df.sort_values("p_value_raw")
    x = np.arange(len(res_df_plot))
    vals = np.asarray(res_df_plot["median_diff(DPG - DiCE)"].values)
    pvals = np.asarray(res_df_plot["p_value_holm"].values)

    # Get pretty names for x-axis
    metric_names = res_df_plot["metric"].tolist()
    pretty_names = [METRIC_NAME_MAP.get(m, m) or m for m in metric_names]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, vals)
    ax.axhline(0, color="black", linewidth=1)

    for i, (v, p) in enumerate(zip(vals, pvals)):
        if p < 0.001:
            mark = "***"
        elif p < 0.01:
            mark = "**"
        elif p < 0.05:
            mark = "*"
        else:
            mark = ""

        if mark:
            ax.text(i, v + np.sign(v) * 0.01, mark, ha="center",
                    va="bottom" if v >= 0 else "top")

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_names, rotation=45, ha="right")
    ax.set_ylabel("Median difference (DPG - DiCE)")
    ax.set_title("Per-metric median difference with Holm-corrected significance")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def run_full_analysis(
    csv_path: str,
    output_dir: str,
    bad_datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run complete statistical analysis and generate all outputs.

    Args:
        csv_path: Path to comparison_numeric_small.csv file.
        output_dir: Directory to save outputs.
        bad_datasets: List of dataset names to exclude.

    Returns:
        Dictionary with analysis results and paths to generated files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_comparison_data(csv_path, bad_datasets=bad_datasets)
    results["n_datasets"] = len(df)
    results["datasets"] = df.index.tolist()
    print(f"  → {len(df)} datasets after cleaning")

    # Compute difference matrix
    print("Computing difference matrices...")
    diff_mat, diff_mat_norm = compute_difference_matrix(df)
    results["diff_mat"] = diff_mat
    results["diff_mat_norm"] = diff_mat_norm

    # Save difference matrices as CSV
    diff_mat_path = os.path.join(output_dir, "difference_matrix.csv")
    diff_mat.to_csv(diff_mat_path)
    print(f"  ✓ Saved: {diff_mat_path}")

    diff_mat_norm_path = os.path.join(output_dir, "difference_matrix_normalized.csv")
    diff_mat_norm.to_csv(diff_mat_norm_path)
    print(f"  ✓ Saved: {diff_mat_norm_path}")

    # Run Wilcoxon tests
    print("Running Wilcoxon signed-rank tests...")
    res_df = wilcoxon_statistical_test(df)
    results["wilcoxon_results"] = res_df

    if len(res_df) > 0:
        wilcoxon_path = os.path.join(output_dir, "wilcoxon_results.csv")
        res_df.to_csv(wilcoxon_path, index=False)
        print(f"  ✓ Saved: {wilcoxon_path}")

        # Generate LaTeX table
        print("Generating LaTeX table...")
        latex_table = generate_latex_table(res_df)
        results["latex_table"] = latex_table

        latex_path = os.path.join(output_dir, "wilcoxon_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)
        print(f"  ✓ Saved: {latex_path}")

    # Generate visualizations
    print("Generating visualizations...")

    # Heatmap
    heatmap_path = os.path.join(output_dir, "heatmap_dpgcf_dice_metrics.pdf")
    fig = plot_heatmap_differences(diff_mat, diff_mat_norm, output_path=heatmap_path)
    plt.close(fig)
    print(f"  ✓ Saved: {heatmap_path}")
    
    heatmap_png_path = os.path.join(output_dir, "heatmap_dpgcf_dice_metrics.png")
    fig = plot_heatmap_differences(diff_mat, diff_mat_norm, output_path=heatmap_png_path)
    plt.close(fig)
    print(f"  ✓ Saved: {heatmap_png_path}")

    if len(res_df) > 0:
        # Metric scatter plot
        scatter_path = os.path.join(output_dir, "metric_scatter.png")
        fig = plot_metric_scatter(df, res_df, output_path=scatter_path)
        plt.close(fig)
        print(f"  ✓ Saved: {scatter_path}")

        # Median difference bars
        bars_path = os.path.join(output_dir, "median_difference_bars.png")
        fig = plot_median_difference_bars(res_df, output_path=bars_path)
        plt.close(fig)
        print(f"  ✓ Saved: {bars_path}")

    # Pairwise comparison for validity metric
    if "perc_valid_cf_all" + SUFFIX_DPG in df.columns:
        pairwise_path = os.path.join(output_dir, "pairwise_validity.png")
        fig = plot_pairwise_comparison(df, "perc_valid_cf_all", output_path=pairwise_path)
        plt.close(fig)
        print(f"  ✓ Saved: {pairwise_path}")

    results["output_dir"] = output_dir
    print(f"\n✓ Analysis complete. Outputs saved to: {output_dir}")

    return results


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run statistical analysis on DPG vs DiCE comparison data")
    parser.add_argument("csv_path", help="Path to comparison_numeric_small.csv file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same directory as input CSV)")
    parser.add_argument("--exclude-datasets", "-e", nargs="+", default=None,
                        help="Dataset names to exclude from analysis")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.csv_path), "statistics")

    run_full_analysis(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        bad_datasets=args.exclude_datasets,
    )
