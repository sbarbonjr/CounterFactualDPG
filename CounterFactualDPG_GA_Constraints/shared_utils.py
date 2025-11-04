"""
Shared utilities for CounterFactual DPG notebooks.

This module contains common functions and classes used across multiple notebooks
in the CounterFactualDPG_GA_Constraints directory.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
