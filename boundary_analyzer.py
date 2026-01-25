"""
BoundaryAnalyzer: Analyzes constraint boundaries between classes.

Extracted from CounterFactualModel.py to provide focused boundary analysis
functionality for counterfactual generation.
"""

import re


class BoundaryAnalyzer:
    """
    Analyzes boundary overlap between original and target class constraints.
    Identifies features where boundaries don't overlap (clear escape paths).
    """

    def __init__(self, constraints, verbose=False):
        """
        Initialize the BoundaryAnalyzer.

        Args:
            constraints (dict): Feature constraints per class.
            verbose (bool): If True, prints detailed analysis information.
        """
        self.constraints = constraints
        self.verbose = verbose
        self._boundary_analysis_cache = {}

    def _normalize_feature_name(self, feature):
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

    def _features_match(self, feature1, feature2):
        """
        Check if two feature names match, using normalized comparison.

        Args:
            feature1 (str): First feature name.
            feature2 (str): Second feature name.

        Returns:
            bool: True if features match, False otherwise.
        """
        return self._normalize_feature_name(feature1) == self._normalize_feature_name(
            feature2
        )

    def analyze_boundary_overlap(self, original_class, target_class):
        """
        Analyze boundary overlap between original and target class constraints.
        Identifies features where boundaries don't overlap (clear escape paths).

        Args:
            original_class (int): The original class of the sample.
            target_class (int): The target class for counterfactual.

        Returns:
            dict: Analysis results with 'non_overlapping', 'overlapping', and 'escape_direction' per feature.
        """
        cache_key = (original_class, target_class)
        if cache_key in self._boundary_analysis_cache:
            return self._boundary_analysis_cache[cache_key]

        original_constraints = self.constraints.get(f"Class {original_class}", [])
        target_constraints = self.constraints.get(f"Class {target_class}", [])

        analysis = {
            "non_overlapping": [],  # Features with clear escape path
            "overlapping": [],  # Features with overlapping bounds
            "escape_direction": {},  # Direction to escape: 'increase', 'decrease', or 'both'
            "feature_bounds": {},  # Store both bounds for each feature
        }

        # Build lookup dict for original constraints
        orig_bounds = {}
        for c in original_constraints:
            feature = c.get("feature", "")
            norm_feature = self._normalize_feature_name(feature)
            orig_bounds[norm_feature] = {
                "min": c.get("min"),
                "max": c.get("max"),
                "original_name": feature,
            }

        # Analyze each target constraint
        for tc in target_constraints:
            feature = tc.get("feature", "")
            norm_feature = self._normalize_feature_name(feature)
            target_min = tc.get("min")
            target_max = tc.get("max")

            # Store bounds info
            analysis["feature_bounds"][norm_feature] = {
                "target_min": target_min,
                "target_max": target_max,
                "original_min": None,
                "original_max": None,
                "feature_name": feature,
            }

            if norm_feature in orig_bounds:
                orig_min = orig_bounds[norm_feature].get("min")
                orig_max = orig_bounds[norm_feature].get("max")

                analysis["feature_bounds"][norm_feature]["original_min"] = orig_min
                analysis["feature_bounds"][norm_feature]["original_max"] = orig_max

                # Determine escape direction based on constraint comparison
                # Key insight: We need to move FROM original bounds TO target bounds
                non_overlapping = False
                escape_dir = "both"

                # First check if constraints are identical (100% overlap, no discrimination)
                if target_min == orig_min and target_max == orig_max:
                    # Identical constraints - maximally overlapping, not useful for discrimination
                    non_overlapping = False
                    escape_dir = "both"
                else:
                    # Case 1: Target has upper bound, Original has lower bound
                    # Example: target_max=5.45, orig_min=5.45 -> must DECREASE to escape
                    # Target_max <= orig_min means target requires values at/below where origin starts
                    # Use <= to catch boundary cases where constraints meet exactly
                    if target_max is not None and orig_min is not None:
                        if target_max <= orig_min:
                            # Target's max is at or below origin's min - clear escape by decreasing
                            non_overlapping = True
                            escape_dir = (
                                "decrease"  # Must decrease to get at/below target_max
                            )
                        elif target_max < orig_min + (
                            orig_max - orig_min if orig_max else 1
                        ):
                            escape_dir = "decrease"  # Prefer decreasing

                    # Case 2: Target has lower bound, Original has upper bound
                    # Example: target_min=5, orig_max=4 -> must INCREASE to escape
                    # Target_min >= orig_max means target requires values at/above where origin ends
                    # Use >= to catch boundary cases where constraints meet exactly
                    if target_min is not None and orig_max is not None:
                        if target_min >= orig_max:
                            # Target's min is at or above origin's max - clear escape by increasing
                            non_overlapping = True
                            escape_dir = (
                                "increase"  # Must increase to get at/above target_min
                            )
                        elif target_min > orig_min if orig_min else 0:
                            escape_dir = "increase"  # Prefer increasing

                    # Case 3: Both have same type of bound - compare values
                    if (
                        target_min is not None
                        and orig_min is not None
                        and target_max is None
                        and orig_max is None
                    ):
                        if target_min > orig_min:
                            escape_dir = "increase"  # Target requires higher minimum
                        elif target_min < orig_min:
                            escape_dir = "decrease"  # Target allows lower values

                    if (
                        target_max is not None
                        and orig_max is not None
                        and target_min is None
                        and orig_min is None
                    ):
                        if target_max < orig_max:
                            escape_dir = "decrease"  # Target requires lower maximum
                        elif target_max > orig_max:
                            escape_dir = "increase"  # Target allows higher values

                if non_overlapping:
                    analysis["non_overlapping"].append(feature)
                else:
                    analysis["overlapping"].append(feature)

                analysis["escape_direction"][norm_feature] = escape_dir
            else:
                # No original constraint for this feature - it's in overlapping (no restriction)
                analysis["overlapping"].append(feature)
                analysis["escape_direction"][norm_feature] = "both"

        # Warn if no non-overlapping features found (constraints are non-discriminative)
        if len(analysis["non_overlapping"]) == 0 and len(target_constraints) > 0:
            if self.verbose:
                print(
                    f"WARNING: No non-overlapping boundaries found between Class {original_class} and Class {target_class}."
                )
                print(f"  The DPG constraints are nearly identical for both classes.")
                print(
                    f"  This may indicate the dataset lacks clear class-separating features in the constraint space."
                )
                print(
                    f"  Counterfactual generation may be difficult or produce poor results."
                )

        self._boundary_analysis_cache[cache_key] = analysis
        return analysis

    def calculate_original_escape_penalty(
        self, individual, sample, original_class, target_class=None
    ):
        """
        Calculate penalty for features still within original class bounds.
        Penalizes individuals that haven't escaped the original class boundaries.

        Enhanced: Only penalizes non-overlapping features where escaping is meaningful.
        For overlapping features, being within both class bounds is acceptable.

        Args:
            individual (dict): The individual to evaluate.
            sample (dict): Original sample.
            original_class (int): The original class of the sample.
            target_class (int, optional): The target class for overlap analysis.

        Returns:
            float: Penalty score (higher = worse, more features still in original bounds).
        """
        original_constraints = self.constraints.get(f"Class {original_class}", [])
        if not original_constraints:
            return 0.0

        # Get non-overlapping features if target_class is provided
        non_overlapping_features = set()
        if target_class is not None:
            boundary_analysis = self.analyze_boundary_overlap(
                original_class, target_class
            )
            non_overlapping_features = set(
                self._normalize_feature_name(f)
                for f in boundary_analysis.get("non_overlapping", [])
            )

        penalty = 0.0
        features_checked = 0

        for feature, value in individual.items():
            norm_feature = self._normalize_feature_name(feature)

            # Only apply escape penalty for non-overlapping features
            # For overlapping features, being within original bounds is OK if also in target bounds
            if (
                target_class is not None
                and norm_feature not in non_overlapping_features
            ):
                continue

            original_value = sample.get(feature, value)

            # Find matching original constraint
            matching_constraint = next(
                (
                    c
                    for c in original_constraints
                    if self._features_match(c.get("feature", ""), feature)
                ),
                None,
            )

            if matching_constraint:
                orig_min = matching_constraint.get("min")
                orig_max = matching_constraint.get("max")

                # Check if value is still within original class bounds
                # For single-bound constraints, only that bound matters
                in_original_bounds = True

                if orig_min is not None and orig_max is not None:
                    # Both bounds: check if inside the range
                    if value < orig_min or value > orig_max:
                        in_original_bounds = False
                elif orig_min is not None:
                    # Only min bound: original class requires value >= orig_min
                    if value < orig_min:
                        in_original_bounds = False
                elif orig_max is not None:
                    # Only max bound: original class requires value <= orig_max
                    if value > orig_max:
                        in_original_bounds = False

                if in_original_bounds:
                    # Calculate how deep inside the original bounds the value is
                    if orig_min is not None and orig_max is not None:
                        range_size = orig_max - orig_min
                        if range_size > 0:
                            # Normalized distance from boundary (0 = at boundary, 1 = at center)
                            center = (orig_min + orig_max) / 2
                            dist_from_boundary = 1.0 - abs(value - center) / (
                                range_size / 2
                            )
                            penalty += max(0, dist_from_boundary)
                    elif orig_min is not None:
                        # Single min bound - penalize being above it (deeper inside = worse)
                        # Use distance from the boundary relative to original value
                        if original_value > orig_min:
                            range_estimate = original_value - orig_min
                            dist_inside = (
                                (value - orig_min) / range_estimate
                                if range_estimate > 0
                                else 0.5
                            )
                            penalty += max(0, min(1, dist_inside))
                        else:
                            penalty += 0.5
                    elif orig_max is not None:
                        # Single max bound - penalize being below it (deeper inside = worse)
                        if original_value < orig_max:
                            range_estimate = orig_max - original_value
                            dist_inside = (
                                (orig_max - value) / range_estimate
                                if range_estimate > 0
                                else 0.5
                            )
                            penalty += max(0, min(1, dist_inside))
                        else:
                            penalty += 0.5

        return penalty
