"""Dataset loading utilities for CounterFactualDPG experiments.

This module provides a unified interface for loading and preprocessing various datasets
used in counterfactual generation experiments. It handles:
- Generic CSV loading with config-driven preprocessing
- Feature type detection (continuous/categorical)
- Label encoding for categorical features
- Feature actionability configuration

All dataset-specific behavior is controlled via YAML config options:
- data.dataset_path: Path to CSV file
- data.target_column: Name of target column
- data.drop_columns: List of columns to drop (e.g., id columns)
- data.binarize_target: Whether to binarize target
- data.binarize_threshold: Threshold for binarization (>= threshold = 1)
- data.missing_values: Strategy for handling missing values ('fill' or 'drop')
- data.categorical_features: Explicit list of categorical feature names (auto-detected if not specified)
- data.continuous_features: Explicit list of continuous feature names (auto-detected if not specified)
- data.variable_features: List of actionable features (all features if not specified)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


def determine_feature_types(features_df, config=None):
    """Determine continuous and categorical feature indices from DataFrame.
    
    Args:
        features_df: DataFrame with features
        config: Optional config with explicit feature type specifications
        
    Returns:
        tuple: (continuous_indices, categorical_indices, variable_indices)
    """
    # Check if config explicitly specifies feature types
    if config and hasattr(config.data, 'continuous_features') and config.data.continuous_features:
        continuous_features = config.data.continuous_features
    else:
        # Auto-detect: numeric columns are continuous
        continuous_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if config and hasattr(config.data, 'categorical_features') and config.data.categorical_features:
        categorical_features = config.data.categorical_features
    else:
        # Auto-detect: non-numeric columns are categorical
        categorical_features = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Convert to indices
    all_cols = list(features_df.columns)
    continuous_indices = [all_cols.index(f) for f in continuous_features if f in all_cols]
    categorical_indices = [all_cols.index(f) for f in categorical_features if f in all_cols]
    
    # Variable features (actionable) - default to all features
    if config and hasattr(config.data, 'variable_features') and config.data.variable_features:
        variable_features = config.data.variable_features
        variable_indices = [all_cols.index(f) for f in variable_features if f in all_cols]
    else:
        # Default: all features are actionable
        variable_indices = list(range(len(all_cols)))
    
    return continuous_indices, categorical_indices, variable_indices


def _preprocess_diabetes_dataset(features_df):
    """Special preprocessing for diabetes dataset: replace zeros with feature mean.
    
    In the diabetes dataset, zero values in certain features (like Glucose, BloodPressure, etc.)
    represent missing data, not actual zeros. This function replaces those zeros with the
    mean of non-zero values for each feature.
    
    Args:
        features_df: DataFrame with diabetes features
        
    Returns:
        DataFrame with zeros replaced by feature means
    """
    features_df_processed = features_df.copy()
    
    # Process each feature
    for col in features_df_processed.columns:
        # Only process numeric columns
        if features_df_processed[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            # Find rows with zero values
            zero_mask = features_df_processed[col] == 0
            num_zeros = zero_mask.sum()
            
            if num_zeros > 0:
                # Calculate mean of non-zero values
                non_zero_values = features_df_processed[col][~zero_mask]
                if len(non_zero_values) > 0:
                    mean_value = non_zero_values.mean()
                    # Replace zeros with mean
                    features_df_processed.loc[zero_mask, col] = mean_value
                    print(f"INFO: Diabetes preprocessing - Replaced {num_zeros} zeros in '{col}' with mean {mean_value:.2f}")
    
    return features_df_processed


def _load_csv_dataset(config, repo_root=None):
    """Generic CSV dataset loader driven by config options.
    
    Config options used:
        data.dataset_path: Path to CSV file (required)
        data.target_column: Name of target column (required)
        data.drop_columns: List of columns to drop (optional)
        data.binarize_target: Whether to binarize target (optional, default False)
        data.binarize_threshold: Threshold for binarization (optional, default 1)
        data.missing_values: Strategy for missing values - 'fill' or 'drop' (optional, default 'fill')
        
    Returns:
        dict with features, labels, feature_names, features_df, label_encoders, and feature indices
    """
    dataset_name = config.data.dataset
    print(f"INFO: Loading {dataset_name} dataset...")
    
    # Load CSV
    dataset_path = config.data.dataset_path
    if not os.path.isabs(dataset_path) and repo_root:
        dataset_path = os.path.join(repo_root, dataset_path)
    
    df = pd.read_csv(dataset_path)
    
    # Drop columns if specified (e.g., id columns)
    drop_columns = getattr(config.data, 'drop_columns', None) or []
    if drop_columns:
        existing_drop_cols = [c for c in drop_columns if c in df.columns]
        if existing_drop_cols:
            print(f"INFO: Dropping columns: {existing_drop_cols}")
            df = df.drop(columns=existing_drop_cols)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        missing_strategy = getattr(config.data, 'missing_values', 'fill')
        print(f"INFO: Dataset has {missing_before} missing values, strategy: {missing_strategy}")
        
        if missing_strategy == 'drop':
            df = df.dropna()
            print(f"INFO: Dropped rows with missing values, {len(df)} rows remaining")
        else:  # 'fill' (default)
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Fill numerical columns with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Fill categorical columns with mode
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
            print(f"INFO: Filled missing values (numerical: median, categorical: mode)")
    
    # Extract target
    target_column = config.data.target_column
    labels = df[target_column].values
    features_df = df.drop(columns=[target_column])
    
    # DIABETES SPECIAL PREPROCESSING: Replace zeros with feature means
    if dataset_name.lower() == 'diabetes':
        print("INFO: Applying diabetes-specific preprocessing (replacing zeros with feature means)")
        features_df = _preprocess_diabetes_dataset(features_df)
    
    label_encoders = {}
    
    # Handle target encoding
    # 1. Binarize target if configured
    binarize_target = getattr(config.data, 'binarize_target', False)
    if binarize_target:
        threshold = getattr(config.data, 'binarize_threshold', 1)
        print(f"INFO: Binarizing target with threshold {threshold} (>= {threshold} = 1, < {threshold} = 0)")
        labels = (labels >= threshold).astype(int)
    
    # 2. Encode target if it's string/object type
    elif labels.dtype == 'object' or str(labels.dtype) == 'object':
        print("INFO: Encoding string target labels")
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        label_encoders['target'] = le
        print(f"INFO: Target label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 3. Encode target if it contains negative values (e.g., -1, 1)
    elif np.issubdtype(labels.dtype, np.number):
        unique_labels = np.unique(labels)
        if np.any(unique_labels < 0):
            print(f"INFO: Encoding target labels from {unique_labels} to non-negative integers")
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            label_encoders['target'] = le
            print(f"INFO: Target label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Encode categorical feature variables
    features_df_encoded = features_df.copy()
    
    for col in features_df.columns:
        col_dtype = features_df[col].dtype
        if col_dtype == 'object' or col_dtype.name == 'category':
            print(f"INFO: Encoding categorical feature: {col}")
            le = LabelEncoder()
            features_df_encoded[col] = le.fit_transform(features_df[col].astype(str))
            label_encoders[col] = le
        elif col_dtype == 'bool':
            features_df_encoded[col] = features_df[col].astype(int)
    
    features = features_df_encoded.values.astype(float)
    feature_names = list(features_df_encoded.columns)
    
    # Final check for NaN/Inf values
    if np.any(~np.isfinite(features)):
        nan_count = np.sum(~np.isfinite(features))
        print(f"WARNING: Found {nan_count} non-finite values in features, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Report stats
    print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
    if label_encoders:
        feature_encoders = {k: v for k, v in label_encoders.items() if k != 'target'}
        if feature_encoders:
            print(f"INFO: Encoded {len(feature_encoders)} categorical features")
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 10:  # Only show distribution for reasonable number of classes
        print(f"INFO: Classes: {unique_labels}, distribution: {np.bincount(labels)}")
    else:
        print(f"INFO: Classes: {len(unique_labels)} unique values")
    
    # Determine feature types
    continuous_indices, categorical_indices, variable_indices = determine_feature_types(features_df_encoded, config)
    
    return {
        'features': features,
        'labels': labels,
        'feature_names': feature_names,
        'features_df': features_df_encoded,
        'label_encoders': label_encoders,
        'continuous_indices': continuous_indices,
        'categorical_indices': categorical_indices,
        'variable_indices': variable_indices,
    }


def load_dataset(config, repo_root=None):
    """Load dataset based on config specification.
    
    Args:
        config: Configuration object with data specifications
        repo_root: Optional path to repository root (for resolving relative paths)
    
    Returns:
        dict with keys:
            - features: numpy array of feature values
            - labels: numpy array of target labels
            - feature_names: list of feature names
            - features_df: pandas DataFrame with feature names
            - label_encoders: dict of LabelEncoder objects (for categorical features)
            - continuous_indices: list of continuous feature indices
            - categorical_indices: list of categorical feature indices
            - variable_indices: list of actionable feature indices
    """
    dataset_name = config.data.dataset.lower()
    
    # Special case: iris (sklearn built-in, no CSV needed)
    if dataset_name == "iris":
        print("INFO: Loading Iris dataset (sklearn built-in)...")
        iris = load_iris()
        features = iris.data
        labels = iris.target
        feature_names = list(iris.feature_names)
        features_df = pd.DataFrame(features, columns=feature_names)
        label_encoders = {}
        
        continuous_indices, categorical_indices, variable_indices = determine_feature_types(features_df, config)
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'features_df': features_df,
            'label_encoders': label_encoders,
            'continuous_indices': continuous_indices,
            'categorical_indices': categorical_indices,
            'variable_indices': variable_indices,
        }
    
    # Generic CSV loader - works for any dataset with proper config
    if hasattr(config.data, 'dataset_path') and config.data.dataset_path:
        return _load_csv_dataset(config, repo_root)
    
    raise ValueError(
        f"Cannot load dataset '{dataset_name}': no dataset_path specified in config. "
        f"Add 'data.dataset_path' to your config YAML pointing to the CSV file."
    )
