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
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance
    print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

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
    colors = ['purple', 'green', 'orange']  # Colors for the classes

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value],
            alpha=0.6
        )

    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color='red', label='Original Sample', edgecolor=colors[original_class]
    )
    plt.scatter(
        counterfactual_pca[:, 0], counterfactual_pca[:, 1],
        color=colors[counterfactual_class], marker='x', s=100, label='Counterfactual', edgecolor='black'
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactual')
    plt.legend()
    #plt.savefig("experiments/PCA_plot_CF.png")
    #plt.show()
    return(plt)

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

    # Plot the pairplot with Seaborn
    sns.pairplot(combined_df, hue='label', palette={'Dataset': 'gray', 'Original Sample': 'red', 'Counterfactual': 'blue'})
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
    plt.figure(figsize=(10, 5))
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
    #plt.savefig("experiments/Plot_CF.png")
    return(plt)
    #plt.show()

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

    # Plot each pair of features
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            ax = axes[i, j]
            if i != j:
                # Scatter plot for different features
                ax.scatter(data_df[feature_i], data_df[feature_j], c='gray', label='Dataset', alpha=0.5)
                ax.scatter(sample_df[feature_i], sample_df[feature_j], c='red', label='Original Sample', edgecolors='k', s=100)
                ax.scatter(counterfactual_df[feature_i], counterfactual_df[feature_j], c='blue', label='Counterfactuals', alpha=0.6, edgecolors='k', s=50)
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
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming you have a trained model, a dataset (as a DataFrame or array), target labels, original sample, and counterfactual_df
#plot_pairwise_with_counterfactual_df(model, X, y, sample, counterfactuals_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_with_counterfactuals(model, dataset, target, sample, counterfactuals_df):
    """
    Plot a PCA visualization of the dataset with the original sample and multiple counterfactuals from a DataFrame.
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
    plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']

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
        color='red', label='Original Sample', edgecolor='black', s=100
    )

    for idx, cf_class in enumerate(counterfactual_classes):
        plt.scatter(
            counterfactuals_pca[idx, 0], counterfactuals_pca[idx, 1],
            color=colors[cf_class % len(colors)], marker='x', s=100, label=f'Counterfactual (Class {cf_class})', edgecolor='black'
        )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactuals')
    #plt.legend()
    plt.show()

# Example usage:
# Assuming model, X, y, sample, counterfactuals_df are appropriately defined
#plot_pca_with_counterfactuals(model, pd.DataFrame(X), y, sample, counterfactuals_df)


