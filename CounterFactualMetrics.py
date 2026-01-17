"""
Comprehensive counterfactual evaluation metrics.

This module provides a complete set of metrics for evaluating counterfactual explanations,
including validity, actionability, distance, diversity, plausibility, and model fidelity metrics.

Adapted from ECE/cf_eval/metrics.py for standalone use in CounterFactualDPG.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _safe_predict(model, X):
    """Safely predict, handling feature names if model was fitted with them.
    
    Args:
        model: sklearn model
        X: numpy array or DataFrame
        
    Returns:
        predictions array
    """
    # Check if model has feature_names_in_ attribute (was fitted with feature names)
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=feature_names)
    return model.predict(X)


class DummyScaler:
    """Simple passthrough scaler for compatibility."""
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return X


# ============================================================================
# Validity Metrics
# ============================================================================

def nbr_valid_cf(cf_list, model, y_val, y_desired=None):
    """Count number of valid counterfactuals (that changed the prediction).
    
    Args:
        cf_list: Array of counterfactual samples (n_samples, n_features)
        model: Trained classifier with predict() method
        y_val: Original predicted class
        y_desired: Optional desired target class
        
    Returns:
        int: Number of valid counterfactuals
    """
    if len(cf_list) == 0:
        return 0
    y_cf = _safe_predict(model, cf_list)
    idx = y_cf != y_val if y_desired is None else y_cf == y_desired
    return int(np.sum(idx))


def perc_valid_cf(cf_list, model, y_val, k=None, y_desired=None):
    """Percentage of valid counterfactuals.
    
    Args:
        cf_list: Array of counterfactual samples
        model: Trained classifier
        y_val: Original predicted class
        k: Denominator (defaults to len(cf_list))
        y_desired: Optional desired target class
        
    Returns:
        float: Percentage of valid counterfactuals
    """
    if len(cf_list) == 0:
        return 0.0
    n_val = nbr_valid_cf(cf_list, model, y_val, y_desired)
    k = len(cf_list) if k is None else k
    return n_val / k if k > 0 else 0.0


# ============================================================================
# Actionability Metrics
# ============================================================================

def nbr_actionable_cf(x, cf_list, variable_features):
    """Count counterfactuals that only modify actionable features.
    
    Args:
        x: Original sample (1D array)
        cf_list: Array of counterfactuals
        variable_features: List/set of actionable feature indices
        
    Returns:
        int: Number of actionable counterfactuals
    """
    if len(cf_list) == 0:
        return 0
    
    nbr_actionable = 0
    nbr_features = cf_list.shape[1]
    
    for cf in cf_list:
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_actionable += 1
    
    return nbr_actionable


def perc_actionable_cf(x, cf_list, variable_features, k=None):
    """Percentage of actionable counterfactuals."""
    if len(cf_list) == 0:
        return 0.0
    n_val = nbr_actionable_cf(x, cf_list, variable_features)
    k = len(cf_list) if k is None else k
    return n_val / k if k > 0 else 0.0


def nbr_valid_actionable_cf(x, cf_list, model, y_val, variable_features, y_desired=None):
    """Count counterfactuals that are both valid and actionable."""
    if len(cf_list) == 0:
        return 0
    
    y_cf = _safe_predict(model, cf_list)
    idx = y_cf != y_val if y_desired is None else y_cf == y_desired
    
    nbr_valid_actionable = 0
    nbr_features = cf_list.shape[1]
    
    for i, cf in enumerate(cf_list):
        if not idx[i]:
            continue
        
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        
        if not constraint_violated:
            nbr_valid_actionable += 1
    
    return nbr_valid_actionable


def perc_valid_actionable_cf(x, cf_list, model, y_val, variable_features, k=None, y_desired=None):
    """Percentage of counterfactuals that are both valid and actionable."""
    if len(cf_list) == 0:
        return 0.0
    n_val = nbr_valid_actionable_cf(x, cf_list, model, y_val, variable_features, y_desired)
    k = len(cf_list) if k is None else k
    return n_val / k if k > 0 else 0.0


def nbr_violations_per_cf(x, cf_list, variable_features):
    """Count constraint violations per counterfactual.
    
    Returns:
        Array of violation counts for each CF
    """
    if len(cf_list) == 0:
        return np.array([])
    
    nbr_features = cf_list.shape[1]
    nbr_violations = np.zeros(len(cf_list))
    
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                nbr_violations[i] += 1
    
    return nbr_violations


def avg_nbr_violations_per_cf(x, cf_list, variable_features):
    """Average number of violations per counterfactual."""
    if len(cf_list) == 0:
        return np.nan
    return float(np.mean(nbr_violations_per_cf(x, cf_list, variable_features)))


def avg_nbr_violations(x, cf_list, variable_features):
    """Average violations normalized by number of CFs and features."""
    if len(cf_list) == 0:
        return np.nan
    violations = nbr_violations_per_cf(x, cf_list, variable_features)
    nbr_cf, nbr_features = cf_list.shape
    return float(np.sum(violations) / (nbr_cf * nbr_features))


# ============================================================================
# Distance Metrics
# ============================================================================

def mad_cityblock(u, v, mad):
    """Manhattan distance normalized by Median Absolute Deviation."""
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = np.abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    """Compute distance between original and counterfactuals on continuous features.
    
    Args:
        x: Original sample (1D or 2D array)
        cf_list: Counterfactual samples (2D array)
        continuous_features: Indices of continuous features
        metric: Distance metric ('euclidean', 'manhattan', 'mad')
        X: Training data (required for 'mad' metric)
        agg: Aggregation function ('mean', 'min', 'max', None=mean)
        
    Returns:
        float: Distance value
    """
    if len(cf_list) == 0:
        return np.nan
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    continuous_features = list(continuous_features)
    
    if metric == 'mad':
        if X is None:
            raise ValueError("X (training data) required for MAD metric")
        mad = median_abs_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])
        
        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        
        dist = cdist(x[:, continuous_features], cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = cdist(x[:, continuous_features], cf_list[:, continuous_features], metric=metric)
    
    dist = dist.flatten()
    
    if agg == 'none':
        return dist
    elif agg is None or agg == 'mean':
        return float(np.mean(dist))
    elif agg == 'max':
        return float(np.max(dist))
    elif agg == 'min':
        return float(np.min(dist))
    else:
        return float(np.mean(dist))


def categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None):
    """Compute distance on categorical features."""
    if len(cf_list) == 0 or len(categorical_features) == 0:
        return np.nan
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    categorical_features = list(categorical_features)
    
    dist = cdist(x[:, categorical_features], cf_list[:, categorical_features], metric=metric)
    dist = dist.flatten()
    
    if agg == 'none':
        return dist
    elif agg is None or agg == 'mean':
        return float(np.mean(dist))
    elif agg == 'max':
        return float(np.max(dist))
    elif agg == 'min':
        return float(np.min(dist))
    else:
        return float(np.mean(dist))


def distance_l2j(x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    """Combined L2 (continuous) + Jaccard (categorical) distance."""
    if len(cf_list) == 0:
        return np.nan
    
    nbr_features = cf_list.shape[1]
    
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=agg)
    
    if np.isnan(dist_cont):
        dist_cont = 0.0
    if np.isnan(dist_cate):
        dist_cate = 0.0
    
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features if nbr_features > 0 else 0.5
        ratio_categorical = len(categorical_features) / nbr_features if nbr_features > 0 else 0.5
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    
    return float(ratio_continuous * dist_cont + ratio_categorical * dist_cate)


def distance_mh(x, cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    """Combined MAD (continuous) + Hamming (categorical) distance."""
    if len(cf_list) == 0:
        return np.nan
    
    nbr_features = cf_list.shape[1]
    
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='hamming', agg=agg)
    
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features if nbr_features > 0 else 0.5
        ratio_categorical = len(categorical_features) / nbr_features if nbr_features > 0 else 0.5
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    
    # Handle 'none' aggregation - return array of combined distances
    if agg == 'none':
        # Handle cases where one component might be nan or array
        if isinstance(dist_cont, np.ndarray) and isinstance(dist_cate, np.ndarray):
            return ratio_continuous * dist_cont + ratio_categorical * dist_cate
        elif isinstance(dist_cont, np.ndarray):
            dist_cate = np.zeros_like(dist_cont) if np.isnan(dist_cate) else dist_cate
            return ratio_continuous * dist_cont + ratio_categorical * dist_cate
        elif isinstance(dist_cate, np.ndarray):
            dist_cont = np.zeros_like(dist_cate) if np.isnan(dist_cont) else dist_cont
            return ratio_continuous * dist_cont + ratio_categorical * dist_cate
        else:
            # Both are scalars (shouldn't happen with agg='none', but handle gracefully)
            return np.array([ratio_continuous * (0.0 if np.isnan(dist_cont) else dist_cont) + 
                           ratio_categorical * (0.0 if np.isnan(dist_cate) else dist_cate)])
    
    # For aggregated results, handle NaN
    if np.isnan(dist_cont):
        dist_cont = 0.0
    if np.isnan(dist_cate):
        dist_cate = 0.0
    
    return float(ratio_continuous * dist_cont + ratio_categorical * dist_cate)


# ============================================================================
# Sparsity/Changes Metrics
# ============================================================================

def nbr_changes_per_cf(x, cf_list, continuous_features):
    """Count number of feature changes per counterfactual.
    
    Continuous features count as 1 change, categorical as 0.5.
    """
    if len(cf_list) == 0:
        return np.array([])
    
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j]:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    """Average number of feature changes per counterfactual."""
    if len(cf_list) == 0:
        return np.nan
    return float(np.mean(nbr_changes_per_cf(x, cf_list, continuous_features)))


def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    """Average changes normalized by number of features."""
    if len(cf_list) == 0:
        return np.nan
    changes = nbr_changes_per_cf(x, cf_list, continuous_features)
    nbr_cf = len(cf_list)
    return float(np.sum(changes) / (nbr_cf * nbr_features))


# ============================================================================
# Diversity Metrics
# ============================================================================

def continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    """Pairwise diversity among counterfactuals on continuous features."""
    if len(cf_list) <= 1:
        return 0.0
    
    continuous_features = list(continuous_features)
    
    if metric == 'mad':
        if X is None:
            raise ValueError("X required for MAD metric")
        mad = median_abs_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])
        
        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        
        dist = pdist(cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list[:, continuous_features], metric=metric)
    
    if agg is None or agg == 'mean':
        return float(np.mean(dist))
    elif agg == 'max':
        return float(np.max(dist))
    elif agg == 'min':
        return float(np.min(dist))
    else:
        return float(np.mean(dist))


def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None):
    """Pairwise diversity on categorical features."""
    if len(cf_list) <= 1 or len(categorical_features) == 0:
        return 0.0
    
    categorical_features = list(categorical_features)
    dist = pdist(cf_list[:, categorical_features], metric=metric)
    
    if agg is None or agg == 'mean':
        return float(np.mean(dist))
    elif agg == 'max':
        return float(np.max(dist))
    elif agg == 'min':
        return float(np.min(dist))
    else:
        return float(np.mean(dist))


def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    """Combined L2 + Jaccard diversity."""
    if len(cf_list) <= 1:
        return 0.0
    
    nbr_features = cf_list.shape[1]
    
    div_cont = continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    div_cate = categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=agg)
    
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features if nbr_features > 0 else 0.5
        ratio_categorical = len(categorical_features) / nbr_features if nbr_features > 0 else 0.5
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    
    return float(ratio_continuous * div_cont + ratio_categorical * div_cate)


def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    """Combined MAD + Hamming diversity."""
    if len(cf_list) <= 1:
        return 0.0
    
    nbr_features = cf_list.shape[1]
    
    div_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
    div_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
    
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features if nbr_features > 0 else 0.5
        ratio_categorical = len(categorical_features) / nbr_features if nbr_features > 0 else 0.5
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    
    return float(ratio_continuous * div_cont + ratio_categorical * div_cate)


def count_diversity(cf_list, features, nbr_features, continuous_features):
    """Count-based diversity measure."""
    if len(cf_list) <= 1:
        return 0.0
    
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    
    for i in range(nbr_cf):
        for j in range(i + 1, nbr_cf):
            for k in features:
                if cf_list[i][k] != cf_list[j][k]:
                    nbr_changes += 1 if k in continuous_features else 0.5
    
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features) if nbr_features > 0 else 0.0


def count_diversity_all(cf_list, nbr_features, continuous_features):
    """Count diversity across all features."""
    if len(cf_list) <= 1:
        return 0.0
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, continuous_features)


# ============================================================================
# Plausibility Metrics
# ============================================================================

def plausibility(x, model, cf_list, X_test, y_pred, continuous_features_all,
                 categorical_features_all, X_train, ratio_cont):
    """Compute plausibility as distance to nearest training sample of CF's predicted class.
    
    Args:
        x: Original sample
        model: Trained classifier
        cf_list: Counterfactual samples
        X_test: Test dataset
        y_pred: Predictions on X_test
        continuous_features_all: Continuous feature indices
        categorical_features_all: Categorical feature indices
        X_train: Training data
        ratio_cont: Ratio of continuous features
        
    Returns:
        float: Sum of plausibility distances
    """
    if len(cf_list) == 0:
        return 0.0
    
    sum_dist = 0.0
    
    for cf in cf_list:
        y_cf_val = _safe_predict(model, cf.reshape(1, -1))[0]
        X_test_y = X_test[y_cf_val == y_pred]
        
        if len(X_test_y) == 0:
            continue
        
        # Use agg='none' to get individual distances for nearest neighbor lookup
        # NOTE: We compute distance from CF to X_test_y (not from x), per the paper's definition:
        # impl = (1/|C|) * sum_{c in C} min_{x' in X} d(c, x')
        neigh_dist = distance_mh(cf.reshape(1, -1), X_test_y, continuous_features_all,
                                categorical_features_all, X_train, ratio_cont, agg='none')
        
        if neigh_dist is None or len(neigh_dist) == 0:
            continue
        
        # Flatten and filter out invalid distances
        neigh_dist_flat = np.asarray(neigh_dist).flatten()
        valid_mask = ~(np.isnan(neigh_dist_flat) | np.isinf(neigh_dist_flat))
        if not np.any(valid_mask):
            continue
        
        # Find the minimum distance (nearest neighbor to cf)
        # The distance is already computed, so just take the minimum
        d = np.min(neigh_dist_flat[valid_mask])
        
        if not np.isnan(d) and not np.isinf(d):
            sum_dist += d
    
    return float(sum_dist)


# ============================================================================
# Model Fidelity Metrics
# ============================================================================

def euclidean_jaccard(x, A, continuous_features, categorical_features, ratio_cont=None):
    """Combined Euclidean + Jaccard distance matrix."""
    nbr_features = A.shape[1]
    
    dist_cont = cdist(x.reshape(1, -1)[:, continuous_features], A[:, continuous_features], metric='euclidean')
    dist_cate = cdist(x.reshape(1, -1)[:, categorical_features], A[:, categorical_features], metric='jaccard')
    
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features if nbr_features > 0 else 0.5
        ratio_categorical = len(categorical_features) / nbr_features if nbr_features > 0 else 0.5
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    
    return ratio_continuous * dist_cont + ratio_categorical * dist_cate


def select_test_knn(x, model, X_test, continuous_features, categorical_features, scaler, test_size=5, get_normalized=False):
    """Select test samples for KNN evaluation."""
    y_val = _safe_predict(model, x.reshape(1, -1))[0]
    y_test = _safe_predict(model, X_test)
    
    nx = scaler.transform(x.reshape(1, -1))
    nX_test = scaler.transform(X_test)
    
    # Split by class
    same_class = y_test == y_val
    diff_class = y_test != y_val
    
    if not np.any(same_class) or not np.any(diff_class):
        # Fallback: return first test_size*2 samples
        idx = np.arange(min(test_size * 2, len(X_test)))
        if get_normalized:
            return X_test[idx], nX_test[idx]
        return X_test[idx]
    
    dist_f = euclidean_jaccard(nx, nX_test[same_class], continuous_features, categorical_features)
    dist_cf = euclidean_jaccard(nx, nX_test[diff_class], continuous_features, categorical_features)
    
    index_f = np.argsort(dist_f[0])[:test_size].tolist()
    index_cf = np.argsort(dist_cf[0])[:test_size].tolist()
    
    index = np.array(index_f + index_cf)
    
    if get_normalized:
        return X_test[index], nX_test[index]
    return X_test[index]


def accuracy_knn_sklearn(x, cf_list, model, X_test, continuous_features, categorical_features, scaler, test_size=5):
    """KNN accuracy using sklearn."""
    if len(cf_list) == 0:
        return 0.0
    
    try:
        clf = KNeighborsClassifier(n_neighbors=1)
        X_train = np.vstack([x.reshape(1, -1), cf_list])
        y_train = _safe_predict(model, X_train)
        clf.fit(X_train, y_train)
        
        X_test_knn = select_test_knn(x, model, X_test, continuous_features, categorical_features, scaler, test_size)
        y_test = _safe_predict(model, X_test_knn)
        y_pred = clf.predict(X_test_knn)
        
        return float(accuracy_score(y_test, y_pred))
    except Exception:
        return 0.0


def accuracy_knn_dist(x, cf_list, model, X_test, continuous_features, categorical_features, scaler, test_size=5):
    """KNN accuracy using distance-based prediction."""
    if len(cf_list) == 0:
        return 0.0
    
    try:
        X_train = np.vstack([x.reshape(1, -1), cf_list])
        y_train = _safe_predict(model, X_train)
        
        nX_train = scaler.transform(X_train)
        
        X_test_knn, nX_test_knn = select_test_knn(x, model, X_test, continuous_features, 
                                                   categorical_features, scaler, test_size, get_normalized=True)
        y_test = _safe_predict(model, X_test_knn)
        
        y_pred = []
        for nx_test in nX_test_knn:
            dist = euclidean_jaccard(nx_test, nX_train, continuous_features, categorical_features)
            idx = np.argmin(dist)
            y_pred.append(y_train[idx])
        
        return float(accuracy_score(y_test, y_pred))
    except Exception:
        return 0.0


def lof(x, cf_list, X, scaler):
    """Local Outlier Factor for counterfactuals."""
    if len(cf_list) == 0:
        return np.nan
    
    try:
        X_train = np.vstack([x.reshape(1, -1), X])
        nX_train = scaler.transform(X_train)
        ncf_list = scaler.transform(cf_list)
        
        clf = LocalOutlierFactor(n_neighbors=min(3, len(nX_train)), novelty=True)
        clf.fit(nX_train)
        
        lof_values = clf.predict(ncf_list)
        return float(np.mean(np.abs(lof_values)))
    except Exception:
        return np.nan


def delta_proba(x, cf_list, model, agg=None):
    """Difference in prediction probability/confidence."""
    if len(cf_list) == 0:
        return 0.0
    
    try:
        y_val = _safe_predict(model, x.reshape(1, -1))[0]
        y_cf = _safe_predict(model, cf_list)
        deltas = np.abs(y_cf - y_val)
        
        if agg is None or agg == 'mean':
            return float(np.mean(deltas))
        elif agg == 'max':
            return float(np.max(deltas))
        elif agg == 'min':
            return float(np.min(deltas))
        else:
            return float(np.mean(deltas))
    except Exception:
        return 0.0


# ============================================================================
# Boundary Escape Metrics (Dual-Boundary)
# ============================================================================

def boundary_escape_score(x, cf_list, original_constraints, target_constraints, feature_names=None):
    """
    Calculate boundary escape score: measures how well counterfactuals escape 
    original class bounds while entering target class bounds.
    
    Args:
        x: Original sample (dict or 1D array)
        cf_list: Array of counterfactual samples (n_samples, n_features)
        original_constraints: List of constraint dicts for original class 
                              [{"feature": "age", "min": 3.0, "max": 10.5}, ...]
        target_constraints: List of constraint dicts for target class
        feature_names: List of feature names (required if x is array)
        
    Returns:
        dict: Dictionary with escape metrics:
            - 'escape_rate': Fraction of constrained features that escaped original bounds
            - 'target_satisfaction_rate': Fraction of features satisfying target bounds
            - 'boundary_transition_score': Combined score (escaped AND in target) / total constrained
            - 'features_escaped': List of feature names that escaped original bounds
            - 'features_in_target': List of feature names satisfying target bounds
    """
    if len(cf_list) == 0:
        return {
            'escape_rate': 0.0,
            'target_satisfaction_rate': 0.0,
            'boundary_transition_score': 0.0,
            'features_escaped': [],
            'features_in_target': []
        }
    
    # Convert to dict format if needed
    if isinstance(x, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]
        x_dict = {feature_names[i]: x[i] for i in range(len(x))}
    else:
        x_dict = x
        if feature_names is None:
            feature_names = list(x.keys())
    
    # Build constraint lookup
    def _normalize_feature(f):
        import re
        f = re.sub(r'\s*\([^)]*\)', '', f)
        f = f.replace('_', ' ')
        f = re.sub(r'\s+', ' ', f)
        return f.strip().lower()
    
    orig_bounds = {}
    for c in original_constraints:
        norm_f = _normalize_feature(c.get("feature", ""))
        orig_bounds[norm_f] = {'min': c.get('min'), 'max': c.get('max')}
    
    target_bounds = {}
    for c in target_constraints:
        norm_f = _normalize_feature(c.get("feature", ""))
        target_bounds[norm_f] = {'min': c.get('min'), 'max': c.get('max')}
    
    # Analyze each counterfactual
    total_escape_count = 0
    total_target_count = 0
    total_both_count = 0
    total_constrained = 0
    
    features_escaped_all = set()
    features_in_target_all = set()
    
    for cf in cf_list:
        if isinstance(cf, np.ndarray):
            cf_dict = {feature_names[i]: cf[i] for i in range(len(cf))}
        else:
            cf_dict = cf
        
        for feature, cf_value in cf_dict.items():
            norm_f = _normalize_feature(feature)
            orig_value = x_dict.get(feature, cf_value)
            
            # Check original bounds
            has_orig_constraint = norm_f in orig_bounds
            escaped_original = True
            
            if has_orig_constraint:
                total_constrained += 1
                orig_min = orig_bounds[norm_f].get('min')
                orig_max = orig_bounds[norm_f].get('max')
                
                # Check if original value was in bounds
                orig_in_bounds = True
                if orig_min is not None and orig_value < orig_min:
                    orig_in_bounds = False
                if orig_max is not None and orig_value > orig_max:
                    orig_in_bounds = False
                
                # Check if cf escaped (if original was in bounds)
                if orig_in_bounds:
                    cf_in_orig_bounds = True
                    if orig_min is not None and cf_value < orig_min:
                        cf_in_orig_bounds = False
                    if orig_max is not None and cf_value > orig_max:
                        cf_in_orig_bounds = False
                    
                    if not cf_in_orig_bounds:
                        escaped_original = True
                        total_escape_count += 1
                        features_escaped_all.add(feature)
                    else:
                        escaped_original = False
                else:
                    # Original wasn't in bounds, so "escaped" by default
                    total_escape_count += 1
                    features_escaped_all.add(feature)
            
            # Check target bounds
            in_target = True
            if norm_f in target_bounds:
                target_min = target_bounds[norm_f].get('min')
                target_max = target_bounds[norm_f].get('max')
                
                if target_min is not None and cf_value < target_min:
                    in_target = False
                if target_max is not None and cf_value > target_max:
                    in_target = False
                
                if in_target:
                    total_target_count += 1
                    features_in_target_all.add(feature)
                    
                    if escaped_original and has_orig_constraint:
                        total_both_count += 1
    
    # Calculate rates
    n_cfs = len(cf_list)
    escape_rate = total_escape_count / total_constrained if total_constrained > 0 else 1.0
    target_rate = total_target_count / (n_cfs * len(target_bounds)) if len(target_bounds) > 0 else 1.0
    transition_score = total_both_count / total_constrained if total_constrained > 0 else 1.0
    
    return {
        'escape_rate': float(escape_rate),
        'target_satisfaction_rate': float(target_rate),
        'boundary_transition_score': float(transition_score),
        'features_escaped': list(features_escaped_all),
        'features_in_target': list(features_in_target_all)
    }


# ============================================================================
# Comprehensive Evaluation Function
# ============================================================================

def evaluate_cf_list(cf_list, x, model, y_val, max_nbr_cf, variable_features, continuous_features_all,
                     categorical_features_all, X_train, X_test, ratio_cont, nbr_features):
    """
    Comprehensive evaluation of a list of counterfactuals.
    
    Args:
        cf_list: Array of counterfactual samples (n_samples, n_features)
        x: Original sample (1D array)
        model: Trained classifier
        y_val: Original predicted class
        max_nbr_cf: Maximum number of CFs expected (for normalization)
        variable_features: List of actionable feature indices
        continuous_features_all: List of continuous feature indices
        categorical_features_all: List of categorical feature indices
        X_train: Training data
        X_test: Test data
        ratio_cont: Ratio of continuous features
        nbr_features: Total number of features
        
    Returns:
        dict: Dictionary with all computed metrics
    """
    
    nbr_cf_ = len(cf_list)
    
    if nbr_cf_ > 0:
        scaler = DummyScaler()
        scaler.fit(X_train)
        
        y_pred = _safe_predict(model, X_test)
        
        # Validity metrics
        nbr_valid_cf_ = nbr_valid_cf(cf_list, model, y_val)
        perc_valid_cf_ = perc_valid_cf(cf_list, model, y_val, k=nbr_cf_)
        perc_valid_cf_all_ = perc_valid_cf(cf_list, model, y_val, k=max_nbr_cf)
        
        # Actionability metrics
        nbr_actionable_cf_ = nbr_actionable_cf(x, cf_list, variable_features)
        perc_actionable_cf_ = perc_actionable_cf(x, cf_list, variable_features, k=nbr_cf_)
        perc_actionable_cf_all_ = perc_actionable_cf(x, cf_list, variable_features, k=max_nbr_cf)
        nbr_valid_actionable_cf_ = nbr_valid_actionable_cf(x, cf_list, model, y_val, variable_features)
        perc_valid_actionable_cf_ = perc_valid_actionable_cf(x, cf_list, model, y_val, variable_features, k=nbr_cf_)
        perc_valid_actionable_cf_all_ = perc_valid_actionable_cf(x, cf_list, model, y_val, variable_features, k=max_nbr_cf)
        avg_nbr_violations_per_cf_ = avg_nbr_violations_per_cf(x, cf_list, variable_features)
        avg_nbr_violations_ = avg_nbr_violations(x, cf_list, variable_features)
        
        # Plausibility
        plausibility_sum = plausibility(x, model, cf_list, X_test, y_pred, continuous_features_all,
                                       categorical_features_all, X_train, ratio_cont)
        plausibility_max_nbr_cf_ = plausibility_sum / max_nbr_cf if max_nbr_cf > 0 else 0.0
        plausibility_nbr_cf_ = plausibility_sum / nbr_cf_ if nbr_cf_ > 0 else 0.0
        plausibility_nbr_valid_cf_ = plausibility_sum / nbr_valid_cf_ if nbr_valid_cf_ > 0 else plausibility_sum
        plausibility_nbr_actionable_cf_ = plausibility_sum / nbr_actionable_cf_ if nbr_actionable_cf_ > 0 else plausibility_sum
        plausibility_nbr_valid_actionable_cf_ = plausibility_sum / nbr_valid_actionable_cf_ if nbr_valid_actionable_cf_ > 0 else plausibility_sum
        
        # Distance metrics
        distance_l2_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None)
        distance_mad_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train)
        distance_j_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard')
        distance_h_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming')
        distance_l2j_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont)
        distance_mh_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont)
        
        distance_l2_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
        distance_mad_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train, agg='min')
        distance_j_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='min')
        distance_h_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='min')
        distance_l2j_min_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
        distance_mh_min_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='min')
        
        distance_l2_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
        distance_mad_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train, agg='max')
        distance_j_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='max')
        distance_h_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='max')
        distance_l2j_max_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
        distance_mh_max_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='max')
        
        # Sparsity metrics
        avg_nbr_changes_per_cf_ = avg_nbr_changes_per_cf(x, cf_list, continuous_features_all)
        avg_nbr_changes_ = avg_nbr_changes(x, cf_list, nbr_features, continuous_features_all)
        
        # Delta metrics
        delta_ = delta_proba(x, cf_list, model, agg='mean')
        delta_min_ = delta_proba(x, cf_list, model, agg='min')
        delta_max_ = delta_proba(x, cf_list, model, agg='max')
        
        # Diversity metrics
        if len(cf_list) > 1:
            diversity_l2_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None)
            diversity_mad_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train)
            diversity_j_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard')
            diversity_h_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming')
            diversity_l2j_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont)
            diversity_mh_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont)
            
            diversity_l2_min_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
            diversity_mad_min_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train, agg='min')
            diversity_j_min_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='min')
            diversity_h_min_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='min')
            diversity_l2j_min_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
            diversity_mh_min_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='min')
            
            diversity_l2_max_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
            diversity_mad_max_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train, agg='max')
            diversity_j_max_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='max')
            diversity_h_max_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='max')
            diversity_l2j_max_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
            diversity_mh_max_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='max')
        else:
            diversity_l2_ = 0.0
            diversity_mad_ = 0.0
            diversity_j_ = 0.0
            diversity_h_ = 0.0
            diversity_l2j_ = 0.0
            diversity_mh_ = 0.0
            
            diversity_l2_min_ = 0.0
            diversity_mad_min_ = 0.0
            diversity_j_min_ = 0.0
            diversity_h_min_ = 0.0
            diversity_l2j_min_ = 0.0
            diversity_mh_min_ = 0.0
            
            diversity_l2_max_ = 0.0
            diversity_mad_max_ = 0.0
            diversity_j_max_ = 0.0
            diversity_h_max_ = 0.0
            diversity_l2j_max_ = 0.0
            diversity_mh_max_ = 0.0
        
        # Count diversity
        count_diversity_cont_ = count_diversity(cf_list, continuous_features_all, nbr_features, continuous_features_all)
        count_diversity_cate_ = count_diversity(cf_list, categorical_features_all, nbr_features, continuous_features_all)
        count_diversity_all_ = count_diversity_all(cf_list, nbr_features, continuous_features_all)
        
        # Model fidelity
        accuracy_knn_sklearn_ = accuracy_knn_sklearn(x, cf_list, model, X_test, continuous_features_all,
                                                      categorical_features_all, scaler, test_size=5)
        accuracy_knn_dist_ = accuracy_knn_dist(x, cf_list, model, X_test, continuous_features_all,
                                               categorical_features_all, scaler, test_size=5)
        
        lof_ = lof(x, cf_list, X_train, scaler)
        
        res = {
            'nbr_cf': nbr_cf_,
            'nbr_valid_cf': nbr_valid_cf_,
            'perc_valid_cf': perc_valid_cf_,
            'perc_valid_cf_all': perc_valid_cf_all_,
            'nbr_actionable_cf': nbr_actionable_cf_,
            'perc_actionable_cf': perc_actionable_cf_,
            'perc_actionable_cf_all': perc_actionable_cf_all_,
            'nbr_valid_actionable_cf': nbr_valid_actionable_cf_,
            'perc_valid_actionable_cf': perc_valid_actionable_cf_,
            'perc_valid_actionable_cf_all': perc_valid_actionable_cf_all_,
            'avg_nbr_violations_per_cf': avg_nbr_violations_per_cf_,
            'avg_nbr_violations': avg_nbr_violations_,
            'distance_l2': distance_l2_,
            'distance_mad': distance_mad_,
            'distance_j': distance_j_,
            'distance_h': distance_h_,
            'distance_l2j': distance_l2j_,
            'distance_mh': distance_mh_,
            'avg_nbr_changes_per_cf': avg_nbr_changes_per_cf_,
            'avg_nbr_changes': avg_nbr_changes_,
            'distance_l2_min': distance_l2_min_,
            'distance_mad_min': distance_mad_min_,
            'distance_j_min': distance_j_min_,
            'distance_h_min': distance_h_min_,
            'distance_l2j_min': distance_l2j_min_,
            'distance_mh_min': distance_mh_min_,
            'distance_l2_max': distance_l2_max_,
            'distance_mad_max': distance_mad_max_,
            'distance_j_max': distance_j_max_,
            'distance_h_max': distance_h_max_,
            'distance_l2j_max': distance_l2j_max_,
            'distance_mh_max': distance_mh_max_,
            'diversity_l2': diversity_l2_,
            'diversity_mad': diversity_mad_,
            'diversity_j': diversity_j_,
            'diversity_h': diversity_h_,
            'diversity_l2j': diversity_l2j_,
            'diversity_mh': diversity_mh_,
            'diversity_l2_min': diversity_l2_min_,
            'diversity_mad_min': diversity_mad_min_,
            'diversity_j_min': diversity_j_min_,
            'diversity_h_min': diversity_h_min_,
            'diversity_l2j_min': diversity_l2j_min_,
            'diversity_mh_min': diversity_mh_min_,
            'diversity_l2_max': diversity_l2_max_,
            'diversity_mad_max': diversity_mad_max_,
            'diversity_j_max': diversity_j_max_,
            'diversity_h_max': diversity_h_max_,
            'diversity_l2j_max': diversity_l2j_max_,
            'diversity_mh_max': diversity_mh_max_,
            'count_diversity_cont': count_diversity_cont_,
            'count_diversity_cate': count_diversity_cate_,
            'count_diversity_all': count_diversity_all_,
            'accuracy_knn_sklearn': accuracy_knn_sklearn_,
            'accuracy_knn_dist': accuracy_knn_dist_,
            'lof': lof_,
            'delta': delta_,
            'delta_min': delta_min_,
            'delta_max': delta_max_,
            'plausibility_sum': plausibility_sum,
            'plausibility_max_nbr_cf': plausibility_max_nbr_cf_,
            'plausibility_nbr_cf': plausibility_nbr_cf_,
            'plausibility_nbr_valid_cf': plausibility_nbr_valid_cf_,
            'plausibility_nbr_actionable_cf': plausibility_nbr_actionable_cf_,
            'plausibility_nbr_valid_actionable_cf': plausibility_nbr_valid_actionable_cf_,
        }
    
    else:
        # No counterfactuals generated
        res = {
            'nbr_cf': 0,
            'nbr_valid_cf': 0.0,
            'perc_valid_cf': 0.0,
            'perc_valid_cf_all': 0.0,
            'nbr_actionable_cf': 0.0,
            'perc_actionable_cf': 0.0,
            'perc_actionable_cf_all': 0.0,
            'nbr_valid_actionable_cf': 0.0,
            'perc_valid_actionable_cf': 0.0,
            'perc_valid_actionable_cf_all': 0.0,
            'avg_nbr_violations_per_cf': np.nan,
            'avg_nbr_violations': np.nan,
            'distance_l2': np.nan,
            'distance_mad': np.nan,
            'distance_j': np.nan,
            'distance_h': np.nan,
            'distance_l2j': np.nan,
            'distance_mh': np.nan,
            'distance_l2_min': np.nan,
            'distance_mad_min': np.nan,
            'distance_j_min': np.nan,
            'distance_h_min': np.nan,
            'distance_l2j_min': np.nan,
            'distance_mh_min': np.nan,
            'distance_l2_max': np.nan,
            'distance_mad_max': np.nan,
            'distance_j_max': np.nan,
            'distance_h_max': np.nan,
            'distance_l2j_max': np.nan,
            'distance_mh_max': np.nan,
            'avg_nbr_changes_per_cf': np.nan,
            'avg_nbr_changes': np.nan,
            'diversity_l2': np.nan,
            'diversity_mad': np.nan,
            'diversity_j': np.nan,
            'diversity_h': np.nan,
            'diversity_l2j': np.nan,
            'diversity_mh': np.nan,
            'diversity_l2_min': np.nan,
            'diversity_mad_min': np.nan,
            'diversity_j_min': np.nan,
            'diversity_h_min': np.nan,
            'diversity_l2j_min': np.nan,
            'diversity_mh_min': np.nan,
            'diversity_l2_max': np.nan,
            'diversity_mad_max': np.nan,
            'diversity_j_max': np.nan,
            'diversity_h_max': np.nan,
            'diversity_l2j_max': np.nan,
            'diversity_mh_max': np.nan,
            'count_diversity_cont': np.nan,
            'count_diversity_cate': np.nan,
            'count_diversity_all': np.nan,
            'accuracy_knn_sklearn': 0.0,
            'accuracy_knn_dist': 0.0,
            'lof': np.nan,
            'delta': 0.0,
            'delta_min': 0.0,
            'delta_max': 0.0,
            'plausibility_sum': 0.0,
            'plausibility_max_nbr_cf': 0.0,
            'plausibility_nbr_cf': 0.0,
            'plausibility_nbr_valid_cf': 0.0,
            'plausibility_nbr_actionable_cf': 0.0,
            'plausibility_nbr_valid_actionable_cf': 0.0,
        }
    
    return res
