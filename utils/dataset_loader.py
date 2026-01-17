"""Dataset loading utilities for CounterFactualDPG experiments.

This module provides a unified interface for loading and preprocessing various datasets
used in counterfactual generation experiments. It handles:
- Dataset-specific loading and preprocessing
- Feature type detection (continuous/categorical)
- Label encoding for categorical features
- Feature actionability configuration
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
    if config and hasattr(config.data, 'continuous_features'):
        continuous_features = config.data.continuous_features
    else:
        # Auto-detect: numeric columns are continuous
        continuous_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if config and hasattr(config.data, 'categorical_features'):
        categorical_features = config.data.categorical_features
    else:
        # Auto-detect: non-numeric columns are categorical
        categorical_features = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Convert to indices
    all_cols = list(features_df.columns)
    continuous_indices = [all_cols.index(f) for f in continuous_features if f in all_cols]
    categorical_indices = [all_cols.index(f) for f in categorical_features if f in all_cols]
    
    # Variable features (actionable) - default to all features
    if config and hasattr(config.data, 'variable_features'):
        variable_features = config.data.variable_features
        variable_indices = [all_cols.index(f) for f in variable_features if f in all_cols]
    else:
        # Default: all features are actionable
        variable_indices = list(range(len(all_cols)))
    
    return continuous_indices, categorical_indices, variable_indices


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
    
    if dataset_name == "iris":
        print("INFO: Loading Iris dataset...")
        iris = load_iris()
        features = iris.data
        labels = iris.target
        feature_names = list(iris.feature_names)
        features_df = pd.DataFrame(features, columns=feature_names)
        label_encoders = {}  # No categorical features in iris
        
        # Determine feature types (Iris: all continuous)
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
    
    elif dataset_name == "german_credit":
        print("INFO: Loading German Credit dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Encode categorical variables
        label_encoders = {}
        features_df_encoded = features_df.copy()
        
        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category':
                print(f"INFO: Encoding categorical feature: {col}")
                le = LabelEncoder()
                features_df_encoded[col] = le.fit_transform(features_df[col])
                label_encoders[col] = le
        
        features = features_df_encoded.values
        feature_names = list(features_df_encoded.columns)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Encoded {len(label_encoders)} categorical features")
        
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
    
    elif dataset_name == "wheat_seeds":
        print("INFO: Loading Wheat Seeds dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # All features are numerical - no encoding needed
        features = features_df.values
        feature_names = list(features_df.columns)
        label_encoders = {}
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "red_wine_quality":
        print("INFO: Loading Red Wine Quality dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Binarize target if configured (quality >= threshold = good, else bad)
        binarize_target = getattr(config.data, 'binarize_target', False)
        if binarize_target:
            threshold = getattr(config.data, 'binarize_threshold', 6)
            print(f"INFO: Binarizing target with threshold {threshold} (>= {threshold} = 1, < {threshold} = 0)")
            labels = (labels >= threshold).astype(int)
        
        # All features are numerical - no encoding needed
        features = features_df.values
        feature_names = list(features_df.columns)
        label_encoders = {}
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "breast_cancer_wisconsin":
        print("INFO: Loading Breast Cancer Wisconsin dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Drop columns if specified (e.g., id column)
        drop_columns = getattr(config.data, 'drop_columns', [])
        if drop_columns:
            print(f"INFO: Dropping columns: {drop_columns}")
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Encode target (M=1, B=0) if it's string
        label_encoders = {}
        if labels.dtype == 'object' or str(labels.dtype) == 'object':
            print("INFO: Encoding target variable (M=Malignant, B=Benign)")
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            label_encoders['target'] = le
            print(f"INFO: Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # All features are numerical - no encoding needed
        features = features_df.values.astype(float)
        feature_names = list(features_df.columns)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "banknote_authentication":
        print("INFO: Loading Banknote Authentication dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # All features are numerical (wavelet-transformed image features)
        features = features_df.values
        feature_names = list(features_df.columns)
        label_encoders = {}
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "diabetes":
        print("INFO: Loading Pima Indians Diabetes dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # All features are numerical
        features = features_df.values
        feature_names = list(features_df.columns)
        label_encoders = {}
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "heart_disease_uci":
        print("INFO: Loading Heart Disease UCI dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Drop columns if specified (e.g., id, dataset columns)
        drop_columns = getattr(config.data, 'drop_columns', [])
        if drop_columns:
            print(f"INFO: Dropping columns: {drop_columns}")
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        
        # Handle missing values - this dataset has many NaN values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"INFO: Dataset has {missing_before} missing values")
            # Option 1: Drop rows with any missing values
            # df = df.dropna()
            # Option 2: Fill missing values (preferred to keep more data)
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    # Fill numerical columns with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Fill categorical columns with mode
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
            print(f"INFO: Filled missing values (numerical: median, categorical: mode)")
            print(f"INFO: Remaining missing values: {df.isnull().sum().sum()}")
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Binarize target if configured (0 = no disease, >0 = disease)
        binarize_target = getattr(config.data, 'binarize_target', False)
        if binarize_target:
            threshold = getattr(config.data, 'binarize_threshold', 1)
            print(f"INFO: Binarizing target with threshold {threshold} (>= {threshold} = 1, < {threshold} = 0)")
            labels = (labels >= threshold).astype(int)
        
        # Encode categorical variables
        label_encoders = {}
        features_df_encoded = features_df.copy()
        
        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category' or features_df[col].dtype == 'bool':
                print(f"INFO: Encoding categorical feature: {col}")
                le = LabelEncoder()
                # Handle boolean columns
                if features_df[col].dtype == 'bool':
                    features_df_encoded[col] = features_df[col].astype(int)
                else:
                    features_df_encoded[col] = le.fit_transform(features_df[col].astype(str))
                    label_encoders[col] = le
        
        features = features_df_encoded.values.astype(float)
        feature_names = list(features_df_encoded.columns)
        
        # Final check for NaN/Inf values
        if np.any(~np.isfinite(features)):
            nan_count = np.sum(~np.isfinite(features))
            print(f"WARNING: Found {nan_count} non-finite values in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Encoded {len(label_encoders)} categorical features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
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
    
    elif dataset_name == "abalone":
        print("INFO: Loading Abalone dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Encode target labels if needed (handle -1, 1 or other values)
        label_encoders = {}
        unique_labels = np.unique(labels)
        if np.any(unique_labels < 0):
            print(f"INFO: Encoding target labels from {unique_labels} to non-negative integers")
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            label_encoders['target'] = le
            print(f"INFO: Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # All features are numerical
        features = features_df.values.astype(float)
        feature_names = list(features_df.columns)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
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
    
    elif dataset_name == "arrhythmia":
        print("INFO: Loading Arrhythmia dataset...")
        
        # Load CSV
        dataset_path = config.data.dataset_path
        if not os.path.isabs(dataset_path) and repo_root:
            dataset_path = os.path.join(repo_root, dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        # Extract target
        target_column = config.data.target_column
        labels = df[target_column].values
        features_df = df.drop(columns=[target_column])
        
        # Encode target labels if needed (handle -1, 1 or other values)
        label_encoders = {}
        unique_labels = np.unique(labels)
        if np.any(unique_labels < 0):
            print(f"INFO: Encoding target labels from {unique_labels} to non-negative integers")
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            label_encoders['target'] = le
            print(f"INFO: Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # All features are numerical
        features = features_df.values.astype(float)
        feature_names = list(features_df.columns)
        
        # Handle missing values if any
        if np.any(~np.isfinite(features)):
            nan_count = np.sum(~np.isfinite(features))
            print(f"WARNING: Found {nan_count} non-finite values in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"INFO: Loaded {len(df)} samples with {len(feature_names)} features")
        print(f"INFO: Classes: {np.unique(labels)}, distribution: {np.bincount(labels)}")
        
        # Determine feature types
        continuous_indices, categorical_indices, variable_indices = determine_feature_types(features_df, config)
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'features_df': pd.DataFrame(features, columns=feature_names),
            'label_encoders': label_encoders,
            'continuous_indices': continuous_indices,
            'categorical_indices': categorical_indices,
            'variable_indices': variable_indices,
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: iris, german_credit, wheat_seeds, red_wine_quality, breast_cancer_wisconsin, banknote_authentication, diabetes, heart_disease_uci, abalone, arrhythmia")
