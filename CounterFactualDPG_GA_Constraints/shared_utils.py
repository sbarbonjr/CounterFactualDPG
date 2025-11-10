"""
Shared utilities for CounterFactual DPG notebooks.

This module contains common functions and classes used across multiple notebooks
in the CounterFactualDPG_GA_Constraints directory.
"""

import os
import ast
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.spatial.distance import cdist


# ============================================================================
# Path Setup
# ============================================================================

def setup_paths():
    """
    Set up paths for accessing constraints directory.
    
    Returns:
        tuple: (notebook_dir, constraints_dir, PATH)
            - notebook_dir: Current working directory
            - constraints_dir: Absolute path to constraints directory
            - PATH: Path to constraints directory with trailing slash
    """
    notebook_dir = os.getcwd()
    constraints_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'constraints'))
    PATH = constraints_dir + '/'
    return notebook_dir, constraints_dir, PATH


# ============================================================================
# Data Loading and Model Training
# ============================================================================

def load_and_train_model():
    """
    Load the Iris dataset, split it, and train a RandomForestClassifier.
    
    Returns:
        tuple: (iris, X, y, X_train, X_test, y_train, y_test, model)
            - iris: The Iris dataset object
            - X: Feature data
            - y: Target labels
            - X_train, X_test: Training and test feature sets
            - y_train, y_test: Training and test labels
            - model: Trained RandomForestClassifier
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier with 3 base learners
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X_train, y_train)
    
    return iris, X, y, X_train, X_test, y_train, y_test, model


# ============================================================================
# Constraint Loading
# ============================================================================

def load_constraints_source(PATH, filename="iris_l3_pv0.001_t2_dpg_metrics.txt"):
    """
    Load constraints source from a file.
    
    Args:
        PATH: Path to the constraints directory with trailing slash
        filename: Name of the constraints file to load
        
    Returns:
        str: The constraints source string (line 1 from the file)
    """
    with open(PATH + filename, 'r') as file:
        lines = file.readlines()
    
    constraints_source = lines[1]
    return constraints_source


# ============================================================================
# Constraint Parsing Functions
# ============================================================================

def parse_condition(condition):
    """Parse a single condition string into a list of dictionaries with feature, operator, and value."""
    # Split by logical operators while ignoring the first part if it's a standalone number
    parts = re.split(r" (<=|>=|<|>|==) ", condition.strip())

    if len(parts) == 3:
        # Simple case: single comparison
        feature, operator, value = parts
        return [{"feature": feature.strip(), "operator": operator, "value": float(value.strip())}]
    elif len(parts) == 5:
        # Complex case: range comparison
        value1, operator1, feature, operator2, value2 = parts
        return [
            {"feature": feature.strip(), "operator": operator1, "value": float(value1.strip())},
            {"feature": feature.strip(), "operator": operator2, "value": float(value2.strip())}
        ]
    else:
        # In case of an unexpected format, return None
        return None


def constraints_v1_to_dict(raw_string):
    """
    Convert raw constraint string to a nested dictionary.
    
    Args:
        raw_string: Raw string containing class bounds
        
    Returns:
        dict: Nested dictionary with class names as keys and constraint conditions as values
    """
    # Remove the prefix and newline character from the string
    stripped_string = raw_string.replace("Class Bounds: ", "").strip()

    # Use ast.literal_eval to safely parse the string into a dictionary
    parsed_dict = ast.literal_eval(stripped_string)

    # Create the nested dictionary with parsed conditions
    nested_dict = {}
    for class_name, conditions in parsed_dict.items():
        nested_conditions = []
        for condition in conditions:
            # Parse each condition into a list of dictionaries of feature, operator, and value
            parsed_conditions = parse_condition(condition)
            if parsed_conditions:
                nested_conditions.extend(parsed_conditions)
        nested_dict[class_name] = nested_conditions

    return nested_dict


def transform_by_feature(nested_dict):
    """
    Transform constraint dictionary to be organized by feature instead of by class.
    
    Args:
        nested_dict: Dictionary with class-based constraints
        
    Returns:
        dict: Dictionary with feature-based constraints
    """
    feature_dict = {}

    # Iterate through each class and its conditions
    for class_name, conditions in nested_dict.items():
        for condition in conditions:
            feature = condition["feature"]
            if feature not in feature_dict:
                feature_dict[feature] = []
            # Append the condition along with the class it belongs to
            feature_dict[feature].append({"class": class_name, "operator": condition["operator"], "value": condition["value"]})

    return feature_dict


def get_intervals_by_feature(feature_based_dict):
    """
    Get min/max intervals for each feature based on constraints.
    
    Args:
        feature_based_dict: Dictionary with feature-based constraints
        
    Returns:
        dict: Dictionary mapping features to (lower_bound, upper_bound) tuples
    """
    feature_intervals = {}

    for feature, conditions in feature_based_dict.items():
        # Initialize intervals
        lower_bound = float('-inf')
        upper_bound = float('inf')

        # Check each condition and update the bounds
        for condition in conditions:
            operator = condition["operator"]
            value = condition["value"]

            if operator == "<":
                upper_bound = min(upper_bound, value)
            elif operator == "<=":
                upper_bound = min(upper_bound, value)
            elif operator == ">":
                lower_bound = max(lower_bound, value)
            elif operator == ">=":
                lower_bound = max(lower_bound, value)

        # Store the interval for this feature
        feature_intervals[feature] = (lower_bound, upper_bound)

    return feature_intervals


def is_value_valid_for_class(class_name, feature, value, nested_dict):
    """
    Check if a value is valid for a specific class and feature.
    
    Args:
        class_name: Name of the class to check against
        feature: Feature name
        value: Value to validate
        nested_dict: Nested dictionary containing constraints
        
    Returns:
        bool: True if value satisfies all conditions, False otherwise
    """
    # Get the conditions for the given class
    conditions = nested_dict.get(class_name, [])

    # Iterate through each condition for the specified feature
    for condition in conditions:
        if condition["feature"] == feature:
            operator = condition["operator"]
            comparison_value = condition["value"]

            # Check if the value satisfies the condition
            if operator == "<" and not (value < comparison_value):
                return False
            elif operator == "<=" and not (value <= comparison_value):
                return False
            elif operator == ">" and not (value > comparison_value):
                return False
            elif operator == ">=" and not (value >= comparison_value):
                return False

    # If all conditions are satisfied, return True
    return True


def read_constraints_from_file(filename):
    """
    Read constraints from a file in JSON format.
    
    Args:
        filename: Path to the constraints file
        
    Returns:
        dict: Dictionary containing parsed constraints
    """
    constraints_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            # Remove new line characters and any leading/trailing whitespace
            line = line.strip()
            if not line:
                continue

            # Split the line at the first colon to separate the class label from the JSON data
            class_label, json_string = line.split(":", 1)

            # Clean up json_string by replacing single quotes with double quotes to make it valid JSON
            json_string = json_string.strip().replace("'", '"').replace("None", "null")

            try:
                # Convert the JSON string into a Python dictionary
                constraints_dict[class_label.strip()] = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {class_label}: {e}")

    return constraints_dict


  