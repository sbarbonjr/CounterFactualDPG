"""Constraint Score Metric for evaluating DPG constraint quality.

This module provides a metric to evaluate how well DPG constraints separate
different classes, which is crucial for counterfactual generation. The metric
balances two components:

1. **Coverage Score**: How many features and classes have constraints.
   More constrained features = more guidance for the genetic algorithm.
   
2. **Separation Score**: How well intervals are separated between classes.
   Less overlap = easier to find samples outside original class bounds.

Score range: [0, 1]
  - 0: No useful constraints (no coverage or complete overlap)
  - 1: Perfect constraints (full coverage with perfect separation)

The composite score is:
  score = COVERAGE_WEIGHT * coverage + SEPARATION_WEIGHT * separation

where coverage = sqrt(feature_coverage * class_coverage) (geometric mean)

Future enhancement: Add a precision component measuring tightness of bounds.
Tighter bounds provide more guidance, but adds complexity. Could weight by
1 / (max - min + epsilon) for bounded intervals.
"""

from __future__ import annotations

import json
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# SCORING WEIGHTS - Adjust these to tune the metric behavior
# =============================================================================

# Weight for coverage component (how many features/classes have constraints)
# Higher = prefer more constrained features even if they overlap
COVERAGE_WEIGHT = 0.6

# Weight for separation component (how well intervals are separated)
# Higher = prefer better separation even if fewer features are constrained
SEPARATION_WEIGHT = 0.4

# Ensure weights sum to 1.0
assert abs(COVERAGE_WEIGHT + SEPARATION_WEIGHT - 1.0) < 1e-9, "Weights must sum to 1.0"


def load_constraints_from_json(path: str) -> Dict:
    """Load constraints from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _get_interval_bounds(
    constraint: Dict[str, Optional[float]],
    feature_min: float,
    feature_max: float,
) -> Tuple[float, float]:
    """Extract interval bounds, using feature range for unbounded values.
    
    Args:
        constraint: Dict with 'min' and 'max' keys (can be None/null)
        feature_min: Minimum value observed for this feature across all classes
        feature_max: Maximum value observed for this feature across all classes
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower = constraint.get("min")
    upper = constraint.get("max")
    
    # Handle null/None bounds by using feature extremes
    if lower is None:
        lower = feature_min
    if upper is None:
        upper = feature_max
    
    return (float(lower), float(upper))


def _compute_interval_overlap(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
) -> float:
    """Compute the overlap between two intervals.
    
    Returns the length of the overlapping region.
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    
    overlap_start = max(lower1, lower2)
    overlap_end = min(upper1, upper2)
    
    return max(0.0, overlap_end - overlap_start)


def _compute_interval_union_length(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
) -> float:
    """Compute the length of the union of two intervals.
    
    Note: This is not the true union for non-overlapping intervals,
    but the span from min to max (convex hull).
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    
    union_start = min(lower1, lower2)
    union_end = max(upper1, upper2)
    
    return max(0.0, union_end - union_start)


def _compute_separation_score_for_pair(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
    feature_range: float,
) -> float:
    """Compute separation score between two intervals.
    
    The score is based on the "non-overlap ratio" normalized by feature range.
    
    Returns:
        Score in [0, 1] where:
        - 0 = complete overlap (one interval contains the other or identical)
        - 1 = no overlap (completely separated)
    """
    if feature_range <= 0:
        return 0.0  # Degenerate case: constant feature
    
    overlap = _compute_interval_overlap(interval1, interval2)
    
    # Compute lengths
    len1 = interval1[1] - interval1[0]
    len2 = interval2[1] - interval2[0]
    
    # Handle degenerate intervals (point intervals or inverted)
    if len1 <= 0 and len2 <= 0:
        # Both are points - check if same point
        return 0.0 if abs(interval1[0] - interval2[0]) < 1e-9 else 1.0
    
    min_len = min(len1, len2)
    if min_len <= 0:
        min_len = max(len1, len2)
    
    if min_len <= 0:
        return 0.0
    
    # Non-overlap ratio: 1 - (overlap / min_length)
    # Using min length because if a small interval is fully inside a large one,
    # the overlap ratio should reflect that the small one is completely overlapped
    overlap_ratio = overlap / min_len
    separation = 1.0 - min(1.0, overlap_ratio)
    
    return separation


def _get_feature_range_across_classes(
    constraints: Dict[str, Dict],
    feature: str,
) -> Tuple[float, float, float]:
    """Get the min, max, and range for a feature across all classes.
    
    Returns:
        Tuple of (min_value, max_value, range)
    """
    min_vals = []
    max_vals = []
    
    for class_label, features in constraints.items():
        if feature not in features:
            continue
        
        feat_constraint = features[feature]
        if feat_constraint.get("min") is not None:
            min_vals.append(feat_constraint["min"])
        if feat_constraint.get("max") is not None:
            max_vals.append(feat_constraint["max"])
    
    if not min_vals and not max_vals:
        return (0.0, 1.0, 1.0)  # Default range
    
    # Use the observed bounds
    feature_min = min(min_vals) if min_vals else (min(max_vals) - 1.0)
    feature_max = max(max_vals) if max_vals else (max(min_vals) + 1.0)
    
    # Ensure we have a valid range
    if feature_max <= feature_min:
        feature_max = feature_min + 1.0
    
    return (feature_min, feature_max, feature_max - feature_min)


def compute_constraint_score(
    constraints: Dict[str, Dict],
    n_total_features: Optional[int] = None,
    n_total_classes: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, Dict]]:
    """Compute the constraint quality score.
    
    This metric balances coverage (how many features/classes are constrained)
    with separation (how well intervals are separated between classes).
    Higher scores indicate better constraints for counterfactual generation.
    
    Args:
        constraints: Dictionary mapping class labels to feature constraints.
            Format: {
                "Class 0": {
                    "feature1": {"min": value, "max": value},
                    ...
                },
                ...
            }
        n_total_features: Total number of features in the dataset. If None,
            inferred as max features observed across all classes in constraints.
        n_total_classes: Total number of classes in the dataset. If None,
            inferred from the number of classes in constraints.
        verbose: If True, return detailed breakdown by feature and class pair.
    
    Returns:
        Dictionary containing:
            - "score": Overall constraint score in [0, 1]
            - "coverage_score": Coverage component [0, 1]
            - "separation_score": Separation component [0, 1]
            - "feature_coverage": Ratio of constrained features
            - "class_coverage": Ratio of constrained classes
            - "n_classes": Number of classes with constraints
            - "n_features": Number of features with constraints
            - "per_feature_scores": (if verbose) Dict of scores per feature
            - "per_pair_scores": (if verbose) Dict of scores per class pair
    """
    class_labels = list(constraints.keys())
    n_classes = len(class_labels)
    
    # Infer total classes if not provided
    if n_total_classes is None:
        n_total_classes = n_classes
    
    if n_classes < 2:
        return {
            "score": 0.0,  # Single class = not useful for counterfactuals
            "coverage_score": 0.0,
            "separation_score": 0.0,
            "feature_coverage": 0.0,
            "class_coverage": n_classes / max(n_total_classes, 1),
            "n_classes": n_classes,
            "n_total_classes": n_total_classes,
            "n_features": 0,
            "n_total_features": n_total_features or 0,
            "message": "Need at least 2 classes to compute separation score",
        }
    
    # Collect all features across all classes
    all_features = set()
    for class_label, features in constraints.items():
        all_features.update(features.keys())
    
    all_features = sorted(all_features)
    n_features = len(all_features)
    
    # Infer total features if not provided (Option B: use max observed)
    if n_total_features is None:
        # Use the number of unique features found in constraints as proxy
        # This is a lower bound - actual dataset may have more features
        n_total_features = n_features
    
    if n_features == 0:
        return {
            "score": 0.0,
            "coverage_score": 0.0,
            "separation_score": 0.0,
            "feature_coverage": 0.0,
            "class_coverage": n_classes / max(n_total_classes, 1),
            "n_classes": n_classes,
            "n_total_classes": n_total_classes,
            "n_features": 0,
            "n_total_features": n_total_features,
            "message": "No features with constraints found",
        }
    
    # ==========================================================================
    # COVERAGE SCORE
    # ==========================================================================
    
    # Feature coverage: ratio of features that have constraints
    feature_coverage = n_features / n_total_features if n_total_features > 0 else 0.0
    
    # Class coverage: ratio of classes that have constraints
    class_coverage = n_classes / n_total_classes if n_total_classes > 0 else 0.0
    
    # Combined coverage using geometric mean (rewards balance between both)
    coverage_score = np.sqrt(feature_coverage * class_coverage)
    
    # ==========================================================================
    # SEPARATION SCORE
    # ==========================================================================
    
    # Compute feature ranges
    feature_ranges = {}
    for feature in all_features:
        feature_ranges[feature] = _get_feature_range_across_classes(
            constraints, feature
        )
    
    # Compute pairwise separation scores for each feature
    class_pairs = list(combinations(class_labels, 2))
    
    per_feature_scores = {}
    per_pair_scores = {f"{c1} vs {c2}": {} for c1, c2 in class_pairs}
    
    for feature in all_features:
        feat_min, feat_max, feat_range = feature_ranges[feature]
        pair_scores = []
        
        for class1, class2 in class_pairs:
            # Get intervals for both classes
            feat_constraints1 = constraints[class1].get(feature, {})
            feat_constraints2 = constraints[class2].get(feature, {})
            
            # Skip if feature not constrained in either class
            if not feat_constraints1 or not feat_constraints2:
                continue
            
            interval1 = _get_interval_bounds(feat_constraints1, feat_min, feat_max)
            interval2 = _get_interval_bounds(feat_constraints2, feat_min, feat_max)
            
            score = _compute_separation_score_for_pair(
                interval1, interval2, feat_range
            )
            pair_scores.append(score)
            per_pair_scores[f"{class1} vs {class2}"][feature] = score
        
        if pair_scores:
            per_feature_scores[feature] = np.mean(pair_scores)
    
    # Compute separation score
    if per_feature_scores:
        separation_score = np.mean(list(per_feature_scores.values()))
    else:
        separation_score = 0.0
    
    # Compute per-pair average scores
    per_pair_avg = {}
    for pair_key, feature_scores in per_pair_scores.items():
        if feature_scores:
            per_pair_avg[pair_key] = np.mean(list(feature_scores.values()))
        else:
            per_pair_avg[pair_key] = 0.0
    
    # ==========================================================================
    # COMPOSITE SCORE
    # ==========================================================================
    
    overall_score = (
        COVERAGE_WEIGHT * coverage_score +
        SEPARATION_WEIGHT * separation_score
    )
    
    result = {
        "score": float(overall_score),
        "coverage_score": float(coverage_score),
        "separation_score": float(separation_score),
        "feature_coverage": float(feature_coverage),
        "class_coverage": float(class_coverage),
        "n_classes": n_classes,
        "n_total_classes": n_total_classes,
        "n_features": n_features,
        "n_total_features": n_total_features,
        "n_class_pairs": len(class_pairs),
    }
    
    if verbose:
        result["per_feature_scores"] = per_feature_scores
        result["per_pair_scores"] = per_pair_scores
        result["per_pair_average"] = per_pair_avg
    
    return result


def compute_constraint_score_from_file(
    json_path: str,
    n_total_features: Optional[int] = None,
    n_total_classes: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, Dict]]:
    """Compute constraint score from a JSON file.
    
    Args:
        json_path: Path to JSON file with constraints
        n_total_features: Total features in dataset (optional)
        n_total_classes: Total classes in dataset (optional)
        verbose: If True, return detailed breakdown
    
    Returns:
        Dictionary with score and metadata
    """
    constraints = load_constraints_from_json(json_path)
    return compute_constraint_score(
        constraints,
        n_total_features=n_total_features,
        n_total_classes=n_total_classes,
        verbose=verbose,
    )


def compare_constraints(
    constraints1: Dict[str, Dict],
    constraints2: Dict[str, Dict],
    name1: str = "Constraints 1",
    name2: str = "Constraints 2",
    n_total_features: Optional[int] = None,
    n_total_classes: Optional[int] = None,
) -> Dict:
    """Compare two constraint sets and return their scores.
    
    Args:
        constraints1: First constraint dictionary
        constraints2: Second constraint dictionary
        name1: Name for first constraints
        name2: Name for second constraints
        n_total_features: Total features in dataset (optional)
        n_total_classes: Total classes in dataset (optional)
    
    Returns:
        Dictionary with comparison results
    """
    result1 = compute_constraint_score(
        constraints1,
        n_total_features=n_total_features,
        n_total_classes=n_total_classes,
        verbose=True,
    )
    result2 = compute_constraint_score(
        constraints2,
        n_total_features=n_total_features,
        n_total_classes=n_total_classes,
        verbose=True,
    )
    
    return {
        name1: result1,
        name2: result2,
        "score_difference": result1["score"] - result2["score"],
        "better": name1 if result1["score"] > result2["score"] else name2,
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Compute constraint quality score for DPG constraints"
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        help="Path to constraints JSON file",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two constraint files",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Total number of features in dataset (for accurate coverage)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=None,
        help="Total number of classes in dataset (for accurate coverage)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed breakdown",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        file1, file2 = args.compare
        constraints1 = load_constraints_from_json(file1)
        constraints2 = load_constraints_from_json(file2)
        
        result = compare_constraints(
            constraints1, constraints2,
            name1=file1, name2=file2,
            n_total_features=args.n_features,
            n_total_classes=args.n_classes,
        )
        
        print(f"\n{'='*70}")
        print("CONSTRAINT SCORE COMPARISON")
        print(f"{'='*70}")
        print(f"Weights: coverage={COVERAGE_WEIGHT:.1f}, separation={SEPARATION_WEIGHT:.1f}")
        print(f"{'='*70}\n")
        
        for name in [file1, file2]:
            r = result[name]
            print(f"{name}:")
            print(f"  SCORE: {r['score']:.4f}")
            print(f"    Coverage:   {r['coverage_score']:.4f} (feature={r['feature_coverage']:.2%}, class={r['class_coverage']:.2%})")
            print(f"    Separation: {r['separation_score']:.4f}")
            print(f"  Features: {r['n_features']}/{r['n_total_features']}, Classes: {r['n_classes']}/{r['n_total_classes']}")
            if args.verbose and "per_feature_scores" in r:
                print("  Per-feature separation:")
                for feat, score in sorted(r["per_feature_scores"].items()):
                    print(f"    {feat}: {score:.4f}")
            print()
        
        print(f"{'='*70}")
        print(f"Score difference: {result['score_difference']:+.4f}")
        print(f"Better constraints: {result['better']}")
        print(f"{'='*70}\n")
        
    elif args.json_file:
        result = compute_constraint_score_from_file(
            args.json_file,
            n_total_features=args.n_features,
            n_total_classes=args.n_classes,
            verbose=args.verbose,
        )
        
        print(f"\n{'='*70}")
        print("CONSTRAINT QUALITY SCORE")
        print(f"{'='*70}")
        print(f"Weights: coverage={COVERAGE_WEIGHT:.1f}, separation={SEPARATION_WEIGHT:.1f}")
        print(f"{'='*70}\n")
        print(f"File: {args.json_file}")
        print(f"\nSCORE: {result['score']:.4f}")
        print(f"  Coverage:   {result['coverage_score']:.4f} (feature={result['feature_coverage']:.2%}, class={result['class_coverage']:.2%})")
        print(f"  Separation: {result['separation_score']:.4f}")
        print(f"\nFeatures: {result['n_features']}/{result['n_total_features']}")
        print(f"Classes: {result['n_classes']}/{result['n_total_classes']}")
        
        if args.verbose and "per_feature_scores" in result:
            print(f"\nPer-feature separation scores:")
            for feat, score in sorted(result["per_feature_scores"].items()):
                print(f"  {feat}: {score:.4f}")
            
            print(f"\nPer-class-pair average scores:")
            for pair, score in sorted(result["per_pair_average"].items()):
                print(f"  {pair}: {score:.4f}")
        
        print(f"\n{'='*70}\n")
    else:
        parser.print_help()
        sys.exit(1)
