def plot_explainer_summary(explainer, original_sample, counterfactual):
    """
    Display a text summary of the counterfactual explanation inside a matplotlib figure, styled like other plots.
    Args:
        explainer: CounterFactualExplainer object with explanation details.
        original_sample: dict of original sample feature values.
        counterfactual: dict of counterfactual feature values.
    """
    # Compose summary text (customize as needed)
    summary_lines = []
    summary_lines.append("Counterfactual Explanation Summary\n")
    summary_lines.append(f"Target class: {explainer.target_class}")
    summary_lines.append("")
    summary_lines.append("Feature changes:")
    for feature in original_sample:
        orig = original_sample[feature]
        cf = counterfactual[feature]
        if orig != cf:
            summary_lines.append(f"- {feature}: {orig} → {cf} (Δ {cf - orig:+.2f})")
        else:
            summary_lines.append(f"- {feature}: {orig} (no change)")
    summary_lines.append("")
    if hasattr(explainer, 'explanation') and explainer.explanation:
        summary_lines.append("Explanation:")
        summary_lines.append(str(explainer.explanation))
    elif hasattr(explainer, 'get_summary'):
        summary_lines.append("Explanation:")
        summary_lines.append(str(explainer.get_summary()))
    summary = "\n".join(summary_lines)

    # Plot the text inside a figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.set_frame_on(False)
    ax.text(0.5, 0.5, summary, fontsize=13, ha='center', va='center', wrap=True,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray', alpha=0.95),
            transform=ax.transAxes)
    ax.set_title('Counterfactual Explanation Summary', fontsize=15, fontweight='bold', pad=18)
    plt.tight_layout()
    plt.close(fig)
    return fig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Set larger font sizes globally
plt.rcParams['font.size'] = 12  # Adjusts the default font size
plt.rcParams['axes.labelsize'] = 16  # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 16  # Font size for the plot title
plt.rcParams['xtick.labelsize'] = 12  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12  # Font size for y-axis tick labels


def plot_constraints(constraints, overlapping=False, class_colors=None, class_colors_list=None, sample=None, sample_class=None):
    """
    Visualize feature constraints for each class using horizontal bar ranges.
    
    Parameters:
    -----------
    constraints : dict
        Dictionary of constraints per class
    overlapping : bool
        If True, show all classes overlapping on one plot.
        If False, show classes side-by-side (default).
    class_colors : dict, optional
        Dictionary mapping class names to colors
    class_colors_list : list, optional
        List of colors for classes by index
    sample : dict, optional
        Sample data to plot on the constraints graph.
        Keys should match feature names (e.g., 'sepal length (cm)')
    sample_class : int, optional
        The class of the sample. Used to color the sample point according to its class.
    """
    # Default colors if not provided
    if class_colors is None:
        class_colors = {}
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    
    # Extract unique features
    features = []
    for class_name in constraints:
        for constraint in constraints[class_name]:
            feature = constraint['feature'].replace('_', ' ')
            if feature not in features:
                features.append(feature)
    
    # Use provided class colors
    class_names = list(constraints.keys())
    colors = [class_colors.get(cn, class_colors_list[i % len(class_colors_list)]) for i, cn in enumerate(class_names)]
    
    if overlapping:
        # Single plot with overlapping bars
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        n_features = len(features)
        n_classes = len(constraints)
        
        # Calculate bar positions with offset for overlapping
        bar_height = 0.25
        y_positions = np.arange(n_features)
        
        for idx, class_name in enumerate(class_names):
            class_constraints = constraints[class_name]
            constraint_dict = {c['feature'].replace('_', ' '): c for c in class_constraints}
            
            # Offset each class slightly
            y_offset = (idx - n_classes/2 + 0.5) * bar_height
            
            for i, feature in enumerate(features):
                if feature in constraint_dict:
                    c = constraint_dict[feature]
                    min_val = c['min'] if c['min'] is not None else 0
                    max_val = c['max'] if c['max'] is not None else 10
                    
                    # Determine the range
                    if c['min'] is None:
                        # Only max constraint
                        ax.barh(i + y_offset, max_val, left=0, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
                    elif c['max'] is None:
                        # Only min constraint
                        range_width = 10 - min_val
                        ax.barh(i + y_offset, range_width, left=min_val, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
                    else:
                        # Both min and max
                        range_width = max_val - min_val
                        ax.barh(i + y_offset, range_width, left=min_val, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
        
        # Plot sample data if provided
        if sample is not None:
            sample_plotted = False
            # Determine sample color based on sample_class
            if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                sample_color = class_colors_list[sample_class]
            else:
                sample_color = 'red'
            
            for i, feature in enumerate(features):
                # Try to find matching feature in sample (handle variations in naming)
                sample_value = None
                for key in sample.keys():
                    if feature.lower() in key.lower() or key.lower() in feature.lower():
                        sample_value = sample[key]
                        break
                
                if sample_value is not None:
                    # Plot sample as a vertical line (red) and marker (class color)
                    ax.plot([sample_value, sample_value], [i - 0.4, i + 0.4], 
                           color='red', linewidth=3, linestyle='-', zorder=10,
                           label='Sample' if not sample_plotted else "")
                    ax.scatter([sample_value], [i], color=sample_color, s=100, 
                              marker='D', edgecolor='darkred', linewidth=1.5, zorder=11)
                    sample_plotted = True
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(features, fontsize=11)
        ax.set_xlabel('Value Range', fontsize=12, fontweight='bold')
        ax.set_title('Feature Constraints per Class (Overlapping View)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 8)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Side-by-side plots (original view)
        n_classes = len(constraints)
        n_features = len(features)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(16, 6), sharey=True)
        if n_classes == 1:
            axes = [axes]
        
        for idx, (class_name, ax) in enumerate(zip(constraints.keys(), axes)):
            class_constraints = constraints[class_name]
            
            # Create a dictionary for easy lookup
            constraint_dict = {c['feature'].replace('_', ' '): c for c in class_constraints}
            
            y_positions = np.arange(n_features)
            
            for i, feature in enumerate(features):
                if feature in constraint_dict:
                    c = constraint_dict[feature]
                    min_val = c['min'] if c['min'] is not None else 0
                    max_val = c['max'] if c['max'] is not None else 10
                    
                    # Determine the range
                    if c['min'] is None:
                        # Only max constraint
                        ax.barh(i, max_val, left=0, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(max_val/2, i, f'≤ {max_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
                    elif c['max'] is None:
                        # Only min constraint
                        range_width = 10 - min_val  # Arbitrary max for visualization
                        ax.barh(i, range_width, left=min_val, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(min_val + range_width/2, i, f'≥ {min_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
                    else:
                        # Both min and max
                        range_width = max_val - min_val
                        ax.barh(i, range_width, left=min_val, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(min_val + range_width/2, i, f'{min_val:.2f} - {max_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Plot sample data if provided
            if sample is not None:
                # Determine sample color based on sample_class
                if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                    sample_color = class_colors_list[sample_class]
                else:
                    sample_color = 'red'
                
                for i, feature in enumerate(features):
                    # Try to find matching feature in sample (handle variations in naming)
                    sample_value = None
                    for key in sample.keys():
                        if feature.lower() in key.lower() or key.lower() in feature.lower():
                            sample_value = sample[key]
                            break
                    
                    if sample_value is not None:
                        # Plot sample as a vertical line (red) and marker (class color)
                        ax.plot([sample_value, sample_value], [i - 0.3, i + 0.3], 
                               color='red', linewidth=3, linestyle='-', zorder=10)
                        ax.scatter([sample_value], [i], color=sample_color, s=100, 
                                  marker='D', edgecolor='darkred', linewidth=1.5, zorder=11)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(features)
            ax.set_xlabel('Value Range', fontsize=11, fontweight='bold')
            ax.set_title(f'{class_name}', fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim(0, 8)
        
        axes[0].set_ylabel('Features', fontsize=11, fontweight='bold')
        
        # Add legend for sample if provided
        if sample is not None:
            # Determine sample color for legend
            if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                sample_color = class_colors_list[sample_class]
            else:
                sample_color = 'red'
            
            # Create custom legend entry
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='D', color='w', 
                                     markerfacecolor=sample_color, markersize=10, 
                                     markeredgecolor='darkred', markeredgewidth=1.5,
                                     label='Sample')]
            axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.suptitle('Feature Constraints per Class', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def plot_pca_loadings(X, feature_names):
    """
    Plot PCA loadings to show feature contributions to principal components.
    
    Parameters:
    -----------
    X : array-like
        The feature data
    feature_names : list
        Names of the features
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # Get the loadings (components)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot arrows for each feature
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, 
               feature, ha='center', va='center', fontsize=10)
    
    # Set plot properties
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('PCA Loadings Plot', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_val = np.abs(loadings).max() * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    
    # Add variance information to the plot
    variance_text = (
        f"PC1 Variance: {pca.explained_variance_ratio_[0]:.2%}\n"
        f"PC2 Variance: {pca.explained_variance_ratio_[1]:.2%}\n"
        f"Total Variance: {sum(pca.explained_variance_ratio_):.2%}"
    )
    ax.text(0.02, 0.98, variance_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
def plot_pca_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a PCA visualization of the dataset with the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for PCA.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset)

    # Transform the original sample and counterfactual using the same PCA
    original_sample_pca = pca.transform(pd.DataFrame([sample]))
    counterfactual_pca = pca.transform(pd.DataFrame([counterfactual]))

    # Predict the class of the counterfactual
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]
    original_class = model.predict(pd.DataFrame([sample]))[0]

    # Plot the PCA results with class colors and 'x' marker for the counterfactual
    plt.figure(figsize=(10, 6))
    n_unique_classes = len(np.unique(target))
    default_palette = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    colors = default_palette[:n_unique_classes]  # Colors for the classes

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value],
            alpha=0.6
        )

    # Plot original sample filled with its class color and a thick black outline
    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color=colors[original_class % len(colors)], label='Original Sample',
        edgecolor='black', linewidths=2.5, s=150, zorder=10
    )
    plt.scatter(
        counterfactual_pca[:, 0], counterfactual_pca[:, 1],
        color=colors[counterfactual_class], marker='x', s=100, label='Counterfactual', linewidths=1.5
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactual')
    plt.legend()
    #plt.savefig("experiments/PCA_plot_CF.png")
    #plt.show()
    return(plt)

def plot_sample_and_counterfactual_comparison(model, sample, sample_df, counterfactual, constraints=None, class_colors_list=None):
    """
    Enhanced visualization combining original and counterfactual samples with:
    1. Side-by-side feature comparison with arrows showing direction of change
    2. Feature changes (signed changes)
    3. Class probability comparison
    
    Args:
        model: Trained scikit-learn model
        sample: Original sample as dictionary
        sample_df: Original sample as DataFrame
        counterfactual: Counterfactual sample as dictionary
        constraints: Dictionary of constraints per class (optional)
        class_colors_list: List of colors for classes (default: ['purple', 'green', 'orange'])
    """
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange']
    
    predicted_class = model.predict(sample_df)[0]
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]
    
    # Calculate metrics
    feature_list = list(sample.keys())
    original_values = np.array(list(sample.values()))
    counterfactual_values = np.array(list(counterfactual.values()))
    changes = counterfactual_values - original_values
    l2_distance = np.linalg.norm(changes)
    l1_distance = np.sum(np.abs(changes))
    
    # Create figure with custom layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Combined Feature Comparison with Arrows
    ax1 = axes[0]
    x_pos = np.arange(len(feature_list))
    width = 0.35
    
    # Bars for original and counterfactual
    bars1 = ax1.barh(x_pos - width/2, original_values, width, 
                     label='Original', color=class_colors_list[predicted_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax1.barh(x_pos + width/2, counterfactual_values, width, 
                     label='Counterfactual', color=class_colors_list[counterfactual_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add arrows showing direction of change
    for i, (orig, cf, change) in enumerate(zip(original_values, counterfactual_values, changes)):
        if abs(change) > 0.01:  # Only show arrow if change is significant
            arrow_color = 'darkgreen' if change < 0 else 'darkred'
            arrow_style = '<-' if change < 0 else '->'
            ax1.annotate('', xy=(cf, i + width/2), xytext=(orig, i - width/2),
                        arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, 
                                      lw=2, alpha=0.6))
            # Add change value
            mid_point = (orig + cf) / 2
            ax1.text(mid_point, i, f'{change:+.2f}', 
                    ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            width_bar = bar.get_width()
            ax1.text(width_bar, bar.get_y() + bar.get_height()/2,
                    f'{width_bar:.2f}', ha='left', va='center', 
                    fontsize=9, fontweight='bold')
    
    # Add constraint indicators if constraints are provided
    if constraints is not None:
        # Get the x-axis limits to determine the plotting range
        xlim = ax1.get_xlim()
        
        # Plot constraints for both classes
        for class_idx in [predicted_class, counterfactual_class]:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                class_constraints = constraints[class_name]
                
                # Create a dictionary for easy lookup by feature name
                constraint_dict = {c['feature']: c for c in class_constraints}
                
                # Determine vertical offset based on class
                y_offset = -width/2 if class_idx == predicted_class else width/2
                
                for i, feature in enumerate(feature_list):
                    # Try to match feature name (handle with and without spaces/underscores)
                    feature_key = None
                    for key in constraint_dict.keys():
                        if key.replace('_', ' ') == feature.replace(' (cm)', '').replace('_', ' '):
                            feature_key = key
                            break
                    
                    if feature_key and feature_key in constraint_dict:
                        c = constraint_dict[feature_key]
                        min_val = c['min'] if c['min'] is not None else xlim[0]
                        max_val = c['max'] if c['max'] is not None else xlim[1]
                        
                        # Draw constraint range as a horizontal line with markers
                        constraint_color = class_colors_list[class_idx]
                        alpha_constraint = 0.5
                        
                        # Draw the range line
                        if c['min'] is not None and c['max'] is not None:
                            # Both min and max
                            ax1.plot([min_val, max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=4, alpha=alpha_constraint, 
                                   linestyle='-', zorder=1)
                            # Add markers at boundaries
                            ax1.plot([min_val], [i + y_offset], marker='|', markersize=12, 
                                   color=constraint_color, alpha=0.8, markeredgewidth=3, zorder=2)
                            ax1.plot([max_val], [i + y_offset], marker='|', markersize=12, 
                                   color=constraint_color, alpha=0.8, markeredgewidth=3, zorder=2)
                        elif c['min'] is not None:
                            # Only min constraint
                            ax1.plot([min_val, xlim[1]], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=4, alpha=alpha_constraint, 
                                   linestyle='-', zorder=1)
                            ax1.plot([min_val], [i + y_offset], marker='|', markersize=12, 
                                   color=constraint_color, alpha=0.8, markeredgewidth=3, zorder=2)
                        elif c['max'] is not None:
                            # Only max constraint
                            ax1.plot([xlim[0], max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=4, alpha=alpha_constraint, 
                                   linestyle='-', zorder=1)
                            ax1.plot([max_val], [i + y_offset], marker='|', markersize=12, 
                                   color=constraint_color, alpha=0.8, markeredgewidth=3, zorder=2)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list])
    ax1.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'Feature Comparison\nClass {predicted_class} → Class {counterfactual_class}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # 2. Feature Changes (Signed changes like original)
    ax2 = axes[1]
    changes_values = changes  # Already calculated as counterfactual - original
    change_colors = ['green' if c < 0 else 'red' if c > 0 else 'gray' for c in changes_values]
    
    bars_change = ax2.barh(range(len(feature_list)), changes_values, 
                           color=change_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(feature_list)))
    ax2.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list])
    ax2.set_xlabel('Change Value', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Changes', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, change) in enumerate(zip(bars_change, changes_values)):
        if abs(change) > 0.01:
            label = f'{change:+.2f}'
            # Position text based on bar direction
            x_pos = change
            ha = 'left' if change > 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                    label, ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # 3. Class Probability Comparison
    ax3 = axes[2]
    original_probs = model.predict_proba(sample_df)[0]
    counterfactual_probs = model.predict_proba(pd.DataFrame([counterfactual]))[0]
    n_classes = len(original_probs)
    class_names = [f'Class {i}' for i in range(n_classes)]
    
    x_pos_prob = np.arange(n_classes)
    width_prob = 0.35
    
    bars1 = ax3.bar(x_pos_prob - width_prob/2, original_probs, width_prob, 
                   label='Original', color=class_colors_list[predicted_class], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x_pos_prob + width_prob/2, counterfactual_probs, width_prob, 
                   label='Counterfactual', color=class_colors_list[counterfactual_class], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x_pos_prob)
    ax3.set_xticklabels(class_names)
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Counterfactual Explanation Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_pairwise_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a Seaborn pairplot of the dataset, highlighting the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for the plot.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """
    # Convert the dataset into a DataFrame and add the target labels
    data_df = pd.DataFrame(dataset, columns=pd.DataFrame([sample]).columns)
    data_df['label'] = 'Dataset'

    # Convert the original sample and counterfactual to DataFrames
    original_sample_df = pd.DataFrame([sample])
    counterfactual_df = pd.DataFrame([counterfactual])

    # Add labels to distinguish the original sample and counterfactual in the plot
    original_sample_df['label'] = 'Original Sample'
    counterfactual_df['label'] = 'Counterfactual'

    # Combine the original sample and counterfactual with the dataset for plotting
    combined_df = pd.concat([data_df, original_sample_df, counterfactual_df], ignore_index=True)

    # Determine the class color for the original sample
    original_class = model.predict(pd.DataFrame([sample]))[0]
    default_palette = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    colors = default_palette

    # Plot the pairplot with Seaborn using the class color for the original sample
    sns.pairplot(combined_df, hue='label', palette={'Dataset': 'gray', 'Original Sample': colors[original_class % len(colors)], 'Counterfactual': 'blue'})
    plt.suptitle('Pairwise Plot with Original Sample and Counterfactual', y=1.02)
    return(plt)    

    #plt.show()


def plot_sample_and_counterfactual_heatmap(sample, class_sample, counterfactual, class_counterfactual, restrictions):
    """
    Plot the original sample, the differences, and the counterfactual as a heatmap,
    and indicate restrictions using icons.

    Args:
        sample (dict): Original sample values.
        counterfactual (dict): Counterfactual sample values.
        restrictions (dict): Restrictions applied to each feature.
    """
    # Create DataFrame from the samples
    sample_df = pd.DataFrame([sample], index=['Original'])
    cf_df = pd.DataFrame([counterfactual], index=['Counterfactual'])

    # Calculate differences
    differences = (cf_df.loc['Counterfactual'] - sample_df.loc['Original']).to_frame('Difference').T

    # Combine all data
    full_df = pd.concat([sample_df, differences, cf_df])

    # Map restrictions to symbols
    symbol_map = {
        'no_change': '⊝',  # Locked symbol for no change
        'non_increasing': '⬇️',  # Down arrow for non-increasing
        'non_decreasing': '⬆️'  # Up arrow for non-decreasing
    }
    restrictions_ser = pd.Series(restrictions).replace(symbol_map)

    mask = np.full_like(full_df, False, dtype=bool)  # Start with no masking
    mask[[0, -1], :] = True  # Only mask the first and last rows


    vmax = np.max(np.abs(full_df.values))
    vmin = -vmax

    # Plotting the heatmap for numeric data
    fig = plt.figure(figsize=(10, 5))
    ax = sns.heatmap(full_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=1.2, linecolor='k',
                     vmin=vmin, vmax=vmax, mask=mask)

    # Annotate with restrictions
    for i, (feat, restr) in enumerate(restrictions_ser.items()):
        ax.text(i + 0.5, 3.5, restr, ha='center', va='center', color='black', fontweight='bold', fontsize=14)

    annotations = full_df.round(2).copy().astype(str)
    for col in full_df.columns:
        annotations.loc['Difference', col] = f"Δ {full_df.loc['Difference', col]:.2f}"

    for (i, j), val in np.ndenumerate(full_df):
        if i == 1:
            continue
        ax.text(j + 0.5, i + 0.5, annotations.iloc[i, j],
                horizontalalignment='center', verticalalignment='center', color='black')


    plt.title(f'Original (Class {class_sample}), Counterfactual (Class {class_counterfactual}) with Restrictions')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, va="center")
    plt.close(fig)
    return fig

# Example usage
#sample = {'petal width (cm)': 6.1, 'petal length (cm)': 2.8, 'sepal length (cm)': 4.7, 'sepal width (cm)': 1.2}
#counterfactual = {'petal width (cm)': 5.78, 'petal length (cm)': 2.55, 'sepal length (cm)': 1.53, 'sepal width (cm)': 1.2}
#class_sample = model.predict(pd.DataFrame([sample]))[0]
#class_counterfactual = model.predict(pd.DataFrame([counterfactual]))[0]
#restrictions = {'petal width (cm)': 'non_decreasing', 'petal length (cm)': 'non_increasing', 'sepal length (cm)': 'non_increasing', 'sepal width (cm)': 'no_change'}
#plot_sample_and_counterfactual_heatmap(sample, class_sample,  counterfactual, class_counterfactual, restrictions)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pairwise_with_counterfactual_df(model, dataset, target, sample, counterfactual_df):
    """
    Plot pairwise plots of the dataset, highlighting the original sample and multiple counterfactuals using matplotlib.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactuals.
        dataset: The original dataset (features) used for the plot.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual_df: DataFrame containing several counterfactual samples.
    """
    # Convert the dataset into a DataFrame
    data_df = pd.DataFrame(dataset, columns=list(sample.keys()))

    # Convert the original sample to DataFrame
    sample_df = pd.DataFrame([sample], index=['Original'])

    # Combine the dataset with marked samples
    combined_df = pd.concat([data_df, sample_df, counterfactual_df])

    # Get feature names
    features = list(sample.keys())
    num_features = len(features)

    # Create a grid of plots
    fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(15, 15))

    # Determine class color for the original sample
    original_class = model.predict(pd.DataFrame([sample]))[0]
    colors = ['purple', 'green', 'orange']

    # Plot each pair of features
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            ax = axes[i, j]
            if i != j:
                # Scatter plot for different features

                # Adds a regression line
                sns.regplot(data=data_df, x=feature_i, y=feature_j, ax=ax, 
                           scatter_kws={'alpha': 0.5, 'color': 'gray', 's': 30}, 
                           line_kws={'color': 'darkgray', 'linewidth': 2})

                ax.scatter(data_df[feature_i], data_df[feature_j], c='gray', label='Dataset', alpha=0.5)
                ax.scatter(sample_df[feature_i], sample_df[feature_j], c=colors[original_class % len(colors)], label='Original Sample', edgecolors='black', linewidths=2.5, s=120)
                ax.scatter(counterfactual_df[feature_i], counterfactual_df[feature_j], c='blue', label='Counterfactuals', alpha=0.6, edgecolors='k', s=50)
                ax.set_ylabel('')
                ax.set_xlabel('')
            else:
                # Histogram on the diagonal
                ax.hist(data_df[feature_i], color='gray', bins=30, alpha=0.5)
                ax.hist(sample_df[feature_i], color='red', bins=1)
                ax.hist(counterfactual_df[feature_i], color='blue', bins=30, alpha=0.6)

            # Label axes if on the perimeter
            if i == num_features - 1:
                ax.set_xlabel(feature_j)
            if j == 0:
                ax.set_ylabel(feature_i)

    # Add legends and adjust layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    try:
        plt.tight_layout()
    except:
        pass  # Ignore tight_layout warnings for complex plots
    plt.close(fig)
    return fig

# Example usage
# Assuming you have a trained model, a dataset (as a DataFrame or array), target labels, original sample, and counterfactual_df
#plot_pairwise_with_counterfactual_df(model, X, y, sample, counterfactuals_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_with_counterfactuals(model, dataset, target, sample, counterfactuals_df, evolution_histories=None):
    """
    Plot a PCA visualization of the dataset with the original sample and multiple counterfactuals from a DataFrame.
    Shows the evolutionary process of GA with opacity gradient from initial to final generation.
    
    Args:
        model: Trained model for predictions
        dataset: Full dataset for PCA
        target: Target labels
        sample: Original sample dict
        counterfactuals_df: DataFrame of final counterfactuals
        evolution_histories: List of evolution histories (one per replication), each a list of dicts
    """
    # Standardize the dataset
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset_scaled)

    # Ensure the sample and counterfactuals are formatted as numeric DataFrame
    sample_df = pd.DataFrame([sample]).select_dtypes(include=[np.number])
    sample_df_scaled = scaler.transform(sample_df)
    if not isinstance(counterfactuals_df, pd.DataFrame):
        raise ValueError("counterfactuals_df must be a pandas DataFrame")

    numeric_cf_df = counterfactuals_df.select_dtypes(include=[np.number])
    numeric_cf_df_scaled = scaler.transform(numeric_cf_df)

    # Transform using PCA
    original_sample_pca = pca.transform(sample_df_scaled)
    counterfactuals_pca = pca.transform(numeric_cf_df_scaled)

    # Predict classes for counterfactuals
    counterfactual_classes = model.predict(numeric_cf_df)

    # Plot the PCA results
    fig = plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']
    # Determine original sample class for consistent styling
    original_class = model.predict(pd.DataFrame([sample]))[0]

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value % len(colors)],
            alpha=0.5
        )

    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color=colors[original_class % len(colors)], label='Original Sample',
        edgecolor='black', linewidths=2.5, s=150, zorder=10
    )

    # Plot evolution histories if provided
    if evolution_histories:
        for history in evolution_histories:
            if not history:
                continue
            
            # Convert evolution history to DataFrame and transform
            history_df = pd.DataFrame(history)
            if history_df.empty:
                continue
                
            history_numeric = history_df.select_dtypes(include=[np.number])
            history_scaled = scaler.transform(history_numeric)
            history_pca = pca.transform(history_scaled)
            
            # Predict classes for each generation
            history_classes = model.predict(history_numeric)
            
            num_generations = len(history)
            for gen_idx, (coords, gen_class) in enumerate(zip(history_pca, history_classes)):
                # Opacity increases from ~0.1 to 1.0
                alpha = 0.1 + (0.9 * gen_idx / max(1, num_generations - 1))
                # Clamp alpha to [0, 1] to avoid floating-point precision errors
                alpha = np.clip(alpha, 0.0, 1.0)
                # Size for circle outline
                size = 60 if gen_idx < num_generations - 1 else 100
                # Use class color
                color = colors[gen_class % len(colors)]

                # Plot circle outline marker
                plt.scatter(
                    coords[0], coords[1],
                    facecolors='none', edgecolors=color, marker='o', s=size,
                    alpha=alpha, linewidths=1.5,
                    zorder=5
                )

                # Add generation number as text inside the circle
                plt.text(
                    coords[0], coords[1], str(gen_idx + 1),
                    ha='center', va='center', fontsize=7 if gen_idx < num_generations - 1 else 9,
                    color=color, alpha=alpha, weight='bold',
                    zorder=6
                )

                # Add thicker circle around final generation
                if gen_idx == num_generations - 1:
                    plt.scatter(
                        coords[0], coords[1],
                        facecolors='none', edgecolors=color,
                        s=int(size * 1.3), linewidths=2.5, alpha=1.0,
                        zorder=6
                    )

            # Draw a line connecting all generations in sequence (first to last)
            try:
                if len(history_pca) > 1:
                    final_class = history_classes[-1]
                    line_color = colors[final_class % len(colors)]
                    # Extract all x and y coordinates for the path
                    x_coords = history_pca[:, 0]
                    y_coords = history_pca[:, 1]
                    plt.plot(x_coords, y_coords,
                             color=line_color, linewidth=1.0, alpha=0.6, zorder=4)
            except Exception:
                pass
    else:
        # Fallback: plot final counterfactuals only (original behavior)
        for idx, cf_class in enumerate(counterfactual_classes):
            plt.scatter(
                counterfactuals_pca[idx, 0], counterfactuals_pca[idx, 1],
                color=colors[cf_class % len(colors)], marker='x', s=100,
                linewidths=1.5, zorder=5
            )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with GA Evolution (opacity: initial→final)')
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[original_class % len(colors)], markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Original Sample'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=8,
               markeredgecolor='gray', markeredgewidth=1.5, label='GA Evolution (faint→solid)')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.close(fig)
    return fig

# Example usage:
# Assuming model, X, y, sample, counterfactuals_df are appropriately defined
#plot_pca_with_counterfactuals(model, pd.DataFrame(X), y, sample, counterfactuals_df)


