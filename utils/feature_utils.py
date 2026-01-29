"""
Feature utility functions for counterfactual generation.

Provides shared feature name normalization and matching functions
used across CounterFactualModel, BoundaryAnalyzer, and ConstraintValidator.
"""

import re


def normalize_feature_name(feature: str) -> str:
    """
    Normalize feature name by stripping whitespace, removing units in parentheses,
    converting to lowercase, and replacing underscores with spaces. This helps match
    features that may have slight variations in naming (e.g., "sepal width" vs
    "sepal_width" vs "sepal width (cm)").

    Args:
        feature (str): The feature name to normalize.

    Returns:
        str: Normalized feature name.
    """
    # Remove anything in parentheses (like units)
    feature = re.sub(r"\s*\([^)]*\)", "", feature)
    # Replace underscores with spaces
    feature = feature.replace("_", " ")
    # Normalize multiple spaces to single space
    feature = re.sub(r"\s+", " ", feature)
    # Strip whitespace and convert to lowercase
    return feature.strip().lower()


def features_match(feature1: str, feature2: str) -> bool:
    """
    Check if two feature names match, using normalized comparison.

    Args:
        feature1 (str): First feature name.
        feature2 (str): Second feature name.

    Returns:
        bool: True if features match, False otherwise.
    """
    return normalize_feature_name(feature1) == normalize_feature_name(feature2)
