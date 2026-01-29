"""
SampleGenerator: Generates valid samples and finds nearest counterfactuals.

Extracted from CounterFactualModel.py to provide focused sample generation
functionality for counterfactual generation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from constants import (
    MUTATION_EPSILON,
    SAMPLE_GEN_RANGE_SCALE,
    SAMPLE_GEN_ESCAPE_BIAS,
    ACTIONABILITY_RANGE_ADJUST,
)


class SampleGenerator:
    """
    Generates valid samples and finds nearest counterfactuals.
    """

    def __init__(
        self,
        model,
        constraints,
        dict_non_actionable=None,
        feature_names=None,
        escape_pressure=0.5,
        X_train=None,
        y_train=None,
        min_probability_margin=0.001,
        verbose=False,
        boundary_analyzer=None,
        constraint_validator=None,
    ):
        """
        Initialize the SampleGenerator.

        Args:
            model: The machine learning model used for predictions.
            constraints (dict): Feature constraints per class.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
            feature_names: List of feature names from the model.
            escape_pressure (float): Balance between escaping original (1.0) vs approaching target (0.0).
            min_probability_margin (float): Minimum margin for accepting counterfactuals.
            verbose (bool): If True, prints detailed information.
            boundary_analyzer: Optional BoundaryAnalyzer instance for overlap analysis.
            constraint_validator: Optional ConstraintValidator instance for validation.
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable
        self.feature_names = feature_names
        self.escape_pressure = escape_pressure
        self.min_probability_margin = min_probability_margin
        self.verbose = verbose
        self.boundary_analyzer = boundary_analyzer
        self.constraint_validator = constraint_validator

    def _normalize_feature_name(self, feature):
        """
        Normalize feature name (delegates to boundary_analyzer if available).
        """
        if self.boundary_analyzer:
            return self.boundary_analyzer._normalize_feature_name(feature)
        # Fallback implementation
        import re

        feature = re.sub(r"\s*\([^)]*\)", "", feature)
        feature = feature.replace("_", " ")
        feature = re.sub(r"\s+", " ", feature)
        return feature.strip().lower()

    def _features_match(self, feature1, feature2):
        """
        Check if two feature names match (delegates to boundary_analyzer if available).
        """
        if self.boundary_analyzer:
            return self.boundary_analyzer._features_match(feature1, feature2)
        # Fallback implementation
        return self._normalize_feature_name(feature1) == self._normalize_feature_name(
            feature2
        )

    def _validate_sample_prediction(self, adjusted_sample, target_class, sample_keys):
        """
        Validate that a sample is predicted as the target class.

        Args:
            adjusted_sample (dict): The sample to validate.
            target_class (int): The expected target class.
            sample_keys (list): Feature names in order.

        Returns:
            tuple: (is_valid: bool, margin: float, pred: int)
                - is_valid: True if predicted as target_class with sufficient margin
                - margin: Probability margin between target and second-best class
                - pred: The predicted class
        """
        try:
            if self.feature_names is not None:
                adjusted_df = pd.DataFrame([adjusted_sample], columns=self.feature_names)
                pred = self.model.predict(adjusted_df)[0]
                proba = self.model.predict_proba(adjusted_df)[0]
            else:
                adjusted_array = np.array([[adjusted_sample[f] for f in sample_keys]])
                pred = self.model.predict(adjusted_array)[0]
                proba = self.model.predict_proba(adjusted_array)[0]

            if pred != target_class:
                return False, 0.0, pred

            # Calculate probability margin
            if hasattr(self.model, "classes_"):
                class_list = list(self.model.classes_)
                if target_class in class_list:
                    target_idx = class_list.index(target_class)
                else:
                    target_idx = target_class
            else:
                target_idx = target_class

            target_prob = proba[target_idx]
            sorted_probs = np.sort(proba)[::-1]
            second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            margin = target_prob - second_best_prob

            is_valid = margin >= self.min_probability_margin
            return is_valid, margin, pred

        except Exception as e:
            if self.verbose:
                print(f"[VERBOSE-DPG] Prediction validation failed: {e}")
            return False, 0.0, None

    def _binary_search_feature(self, sample, feature, v_min, v_max, target_class, 
                                original_value, escape_dir, eps=0.01, max_iter=100):
        """
        Binary search within [v_min, v_max] to find a value for feature that
        results in the sample being classified as target_class.

        Args:
            sample (dict): Current sample values.
            feature (str): Feature name to search.
            v_min (float): Minimum bound for search.
            v_max (float): Maximum bound for search.
            target_class (int): Target class for validation.
            original_value (float): Original feature value (for minimal change preference).
            escape_dir (str): Escape direction ('increase', 'decrease', or 'both').
            eps (float): Minimum interval size to stop search.
            max_iter (int): Maximum iterations.

        Returns:
            float or None: Valid feature value, or None if not found.
        """
        sample_keys = list(sample.keys())
        test_sample = sample.copy()
        
        # Determine search direction based on escape_dir
        # For 'increase': search from v_min toward v_max (prefer values closer to v_min)
        # For 'decrease': search from v_max toward v_min (prefer values closer to v_max)
        if escape_dir == "increase":
            low, high = v_min, v_max
        elif escape_dir == "decrease":
            low, high = v_min, v_max
        else:
            low, high = v_min, v_max

        best_valid_value = None
        best_distance = float('inf')
        
        for iteration in range(max_iter):
            if abs(high - low) < eps:
                break
                
            mid = (low + high) / 2
            test_sample[feature] = mid
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            
            if is_valid:
                # Found valid value - track it and search for closer one
                distance = abs(mid - original_value)
                if distance < best_distance:
                    best_valid_value = mid
                    best_distance = distance
                
                # Search toward original value for minimal change
                if escape_dir == "increase":
                    # Valid at mid, try lower (closer to original if escape is increase)
                    high = mid
                elif escape_dir == "decrease":
                    # Valid at mid, try higher (closer to original if escape is decrease)
                    low = mid
                else:
                    # No clear direction - search toward original
                    if mid > original_value:
                        high = mid
                    else:
                        low = mid
            else:
                # Not valid - search deeper into target bounds
                if escape_dir == "increase":
                    low = mid  # Need higher values
                elif escape_dir == "decrease":
                    high = mid  # Need lower values
                else:
                    # Try moving away from original
                    if mid > original_value:
                        low = mid
                    else:
                        high = mid
            
            if self.verbose and iteration % 5 == 0:
                status = "valid" if is_valid else "invalid"
                print(f"[VERBOSE-DPG]     Binary search iter {iteration}: {feature}={mid:.4f} ({status})")

        # If binary search found a valid value, return it
        if best_valid_value is not None:
            if self.verbose:
                print(f"[VERBOSE-DPG]     Binary search found valid value: {feature}={best_valid_value:.4f}")
            return best_valid_value

        # Fallback: try boundary values directly
        for boundary_val in [v_min + eps, v_max - eps, (v_min + v_max) / 2]:
            if v_min <= boundary_val <= v_max:
                test_sample[feature] = boundary_val
                is_valid, margin, pred = self._validate_sample_prediction(
                    test_sample, target_class, sample_keys
                )
                if is_valid:
                    if self.verbose:
                        print(f"[VERBOSE-DPG]     Boundary fallback found: {feature}={boundary_val:.4f}")
                    return boundary_val

        return None

    def _progressive_depth_search(self, sample, feature_bounds_info, target_class, 
                                   eps=0.01, max_iter=30):
        """
        Progressive depth search: scale ALL features together from minimal to maximal
        values within their target bounds, binary searching for the minimum depth
        that achieves the target class.

        Args:
            sample (dict): Original sample values.
            feature_bounds_info (dict): Feature bounds info from initial pass.
            target_class (int): Target class for validation.
            eps (float): Minimum depth interval to stop search.
            max_iter (int): Maximum iterations.

        Returns:
            dict or None: Valid sample at minimal depth, or None if not found.
        """
        sample_keys = list(sample.keys())
        
        # Collect features that can be varied (have valid bounds and are searchable)
        searchable_features = []
        for feature, bounds in feature_bounds_info.items():
            # Skip no_change features
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                if self.dict_non_actionable[feature] == "no_change":
                    continue
            
            v_min = bounds['min']
            v_max = bounds['max']
            escape_dir = bounds['escape_dir']
            original_value = bounds['original']
            
            # Apply actionability constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    v_min = max(v_min, original_value)
                elif actionability == "non_increasing":
                    v_max = min(v_max, original_value)
            
            # Fix inverted bounds (can happen with decrease escape direction)
            if v_min > v_max:
                v_min, v_max = v_max, v_min
            
            # Skip invalid bounds (after potential swap)
            if v_min >= v_max:
                continue
            
            # Calculate start (minimal change) and end (maximal change) values
            # based on escape direction
            if escape_dir == "increase":
                start_val = v_min + eps  # Just inside target bounds
                end_val = v_max - eps    # Deep inside target bounds
            elif escape_dir == "decrease":
                start_val = v_max - eps  # Just inside target bounds  
                end_val = v_min + eps    # Deep inside target bounds
            else:
                # For 'both' direction, move toward midpoint
                start_val = original_value if v_min <= original_value <= v_max else (v_min + v_max) / 2
                end_val = (v_min + v_max) / 2
            
            searchable_features.append({
                'feature': feature,
                'start': start_val,
                'end': end_val,
                'original': original_value,
            })
        
        if not searchable_features:
            if self.verbose:
                print("[VERBOSE-DPG]   No searchable features for progressive depth search")
            return None
        
        if self.verbose:
            print(f"[VERBOSE-DPG]   Progressive depth search on {len(searchable_features)} features")
        
        # Binary search on depth (0 = minimal change, 1 = maximal change)
        low_depth, high_depth = 0.0, 1.0
        best_valid_sample = None
        best_depth = float('inf')
        
        for iteration in range(max_iter):
            if abs(high_depth - low_depth) < eps:
                break
            
            mid_depth = (low_depth + high_depth) / 2
            
            # Build test sample at this depth
            test_sample = sample.copy()
            for feat_info in searchable_features:
                feature = feat_info['feature']
                start = feat_info['start']
                end = feat_info['end']
                # Interpolate between start and end based on depth
                test_sample[feature] = start + mid_depth * (end - start)
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            
            if is_valid:
                # Found valid - record and search for lower depth (minimal change)
                if mid_depth < best_depth:
                    best_valid_sample = test_sample.copy()
                    best_depth = mid_depth
                high_depth = mid_depth  # Try to find even lower depth that works
            else:
                low_depth = mid_depth  # Need deeper into target bounds
            
            if self.verbose and iteration % 5 == 0:
                status = "valid" if is_valid else "invalid"
                print(f"[VERBOSE-DPG]     Depth search iter {iteration}: depth={mid_depth:.3f} ({status})")
        
        if best_valid_sample is not None:
            if self.verbose:
                print(f"[VERBOSE-DPG]   Progressive depth search found valid at depth {best_depth:.3f}")
                for feat_info in searchable_features:
                    feature = feat_info['feature']
                    orig = feat_info['original']
                    new_val = best_valid_sample[feature]
                    delta = new_val - orig
                    print(f"[VERBOSE-DPG]     {feature}: {orig:.4f} → {new_val:.4f} (Δ={delta:+.4f})")
            return best_valid_sample
        
        # Try extremes as fallback
        for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
            test_sample = sample.copy()
            for feat_info in searchable_features:
                feature = feat_info['feature']
                start = feat_info['start']
                end = feat_info['end']
                test_sample[feature] = start + depth * (end - start)
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            if is_valid:
                if self.verbose:
                    print(f"[VERBOSE-DPG]   Fallback depth {depth} found valid")
                return test_sample
        
        return None

    def _random_sample_search(self, sample, feature_bounds_info, target_class, n_samples=100):
        """
        Random sampling within constraint space as last resort.
        
        Args:
            sample (dict): Original sample values.
            feature_bounds_info (dict): Feature bounds info from initial pass.
            target_class (int): Target class for validation.
            n_samples (int): Number of random samples to try.
            
        Returns:
            dict or None: Valid sample closest to original, or None if not found.
        """
        sample_keys = list(sample.keys())
        
        # Collect searchable features with valid bounds
        searchable_features = []
        for feature, bounds in feature_bounds_info.items():
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                if self.dict_non_actionable[feature] == "no_change":
                    continue
            
            v_min = bounds['min']
            v_max = bounds['max']
            original_value = bounds['original']
            
            # Apply actionability constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    v_min = max(v_min, original_value)
                elif actionability == "non_increasing":
                    v_max = min(v_max, original_value)
            
            # Fix inverted bounds
            if v_min > v_max:
                v_min, v_max = v_max, v_min
            
            if v_min < v_max:
                searchable_features.append({
                    'feature': feature,
                    'min': v_min,
                    'max': v_max,
                    'original': original_value,
                })
        
        if not searchable_features:
            return None
        
        best_sample = None
        best_distance = float('inf')
        
        for i in range(n_samples):
            test_sample = sample.copy()
            
            for feat_info in searchable_features:
                feature = feat_info['feature']
                v_min = feat_info['min']
                v_max = feat_info['max']
                # Random value within bounds
                test_sample[feature] = np.random.uniform(v_min, v_max)
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            
            if is_valid:
                # Calculate distance to original
                distance = sum(
                    (test_sample[f['feature']] - f['original'])**2 
                    for f in searchable_features
                )**0.5
                
                if distance < best_distance:
                    best_sample = test_sample.copy()
                    best_distance = distance
                    
                    if self.verbose and i % 20 == 0:
                        print(f"[VERBOSE-DPG]     Random sample {i}: found valid (distance={distance:.4f})")
        
        if best_sample is not None and self.verbose:
            print(f"[VERBOSE-DPG]   Best random sample distance: {best_distance:.4f}")
            for feat_info in searchable_features:
                feature = feat_info['feature']
                orig = feat_info['original']
                new_val = best_sample[feature]
                delta = new_val - orig
                print(f"[VERBOSE-DPG]     {feature}: {orig:.4f} → {new_val:.4f} (Δ={delta:+.4f})")
        
        return best_sample

    def get_valid_sample(self, sample, target_class, original_class):
        """
        Generate a valid sample that meets all constraints for the specified target class
        while respecting actionable changes.

        Enhanced with dual-boundary support: the sample is biased to move away from original class bounds toward target class bounds.

        Args:
            sample (dict): The sample with feature values.
            target_class (int): The target class for filtering constraints.
            original_class (int): The original class for escape-aware generation.

        Returns:
            dict: A valid sample that meets all constraints for the target class
                  and respects actionable changes.
        """
        if self.verbose:
            print(f"[VERBOSE-DPG] Generating valid sample for target class {target_class} - get_valid_sample")
            if original_class is not None:
                print(f"[VERBOSE-DPG]   Original class: {original_class} (escape-aware generation)")
        
        adjusted_sample = sample.copy()  # Start with the original values
        feature_bounds_info = {}  # Track bounds for retry
        # Filter the constraints for the specified target class
        class_constraints = self.constraints.get(f"Class {target_class}", [])
        original_constraints = (self.constraints.get(f"Class {original_class}", []))

        # Get boundary analysis for escape direction if original class provided
        boundary_analysis = None
        if original_class is not None and self.boundary_analyzer:
            boundary_analysis = self.boundary_analyzer.analyze_boundary_overlap(
                original_class, target_class
            )

        for feature, original_value in sample.items():
            min_value = -np.inf
            max_value = np.inf
            escape_dir = "both"
            orig_min, orig_max = None, None
            raw_target_min, raw_target_max = None, None  # Store raw constraints

            # Find the constraints for this feature using direct lookup
            matching_constraint = next(
                (
                    condition
                    for condition in class_constraints
                    if self._features_match(condition["feature"], feature)
                ),
                None,
            )

            if matching_constraint:
                raw_target_min = matching_constraint.get("min")
                raw_target_max = matching_constraint.get("max")
                min_value = (
                    raw_target_min
                    if raw_target_min is not None
                    else -np.inf
                )
                max_value = (
                    raw_target_max
                    if raw_target_max is not None
                    else np.inf
                )

            # Get original class bounds for escape direction
            if original_constraints:
                matching_orig = next(
                    (
                        c
                        for c in original_constraints
                        if self._features_match(c.get("feature", ""), feature)
                    ),
                    None,
                )
                if matching_orig:
                    orig_min = matching_orig.get("min")
                    orig_max = matching_orig.get("max")

            # Get escape direction from boundary analysis
            if boundary_analysis:
                norm_feature = self._normalize_feature_name(feature)
                escape_dir = boundary_analysis.get("escape_direction", {}).get(
                    norm_feature, "both"
                )

            # Incorporate non-actionable constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]

                if actionability == "non_decreasing":
                    min_value = max(min_value, original_value)
                    if min_value > max_value:
                        max_value = (
                            min_value + min_value * ACTIONABILITY_RANGE_ADJUST
                        )  # Adjust to ensure valid range
                elif actionability == "non_increasing":
                    max_value = min(max_value, original_value)
                    if max_value < min_value:
                        min_value = (
                            max_value + max_value * ACTIONABILITY_RANGE_ADJUST
                        )  # Adjust to ensure valid range
                elif actionability == "no_change":
                    adjusted_sample[feature] = original_value
                    min_value = original_value
                    max_value = original_value
                    continue

            # If no explicit min/max constraints, use range around original value
            if min_value == -np.inf:
                min_value = original_value - SAMPLE_GEN_RANGE_SCALE * (
                    abs(original_value) + 1.0
                )
            if max_value == np.inf:
                max_value = original_value + SAMPLE_GEN_RANGE_SCALE * (
                    abs(original_value) + 1.0
                )

            # Determine target value based on escape direction and dual-boundary awareness
            # Key insight: we need to move FROM original bounds TO target bounds
            # Use a small epsilon to step just outside origin bounds
            epsilon = MUTATION_EPSILON

            if escape_dir == "increase":
                # Target requires higher values than origin allows
                # Example: orig_max=4, target_min=5 -> value should be just above target_min
                if orig_max is not None and max_value is not None:
                    # Step just above origin's max (into target range)
                    if min_value is not None and min_value > orig_max:
                        # Target's min is above origin's max - step to just at/above target's min
                        target_value = min_value + epsilon
                    else:
                        # No clear boundary gap - go to origin's max + epsilon
                        target_value = orig_max + epsilon
                elif min_value is not None:
                    # Only have target min bound - go just above it
                    target_value = min_value + epsilon
                else:
                    # Bias toward upper bound to escape original class
                    target_value = min_value + (max_value - min_value) * (
                        0.5 + SAMPLE_GEN_ESCAPE_BIAS * self.escape_pressure
                    )

            elif escape_dir == "decrease":
                # Target requires lower values than origin allows
                # Example: orig_min=5.45, target_max=5.45 -> value should be 5.44 (just below origin's min)
                if orig_min is not None and max_value is not None:
                    # Step just below origin's min (into target range)
                    if max_value is not None and max_value < orig_min:
                        # Target's max is below origin's min - step to just at/below target's max
                        target_value = max_value - epsilon
                    elif max_value is not None and max_value <= orig_min:
                        # Target's max equals or is just at origin's min boundary
                        # Step just below origin's min to escape
                        target_value = orig_min - epsilon
                    else:
                        # No clear boundary gap - go to origin's min - epsilon
                        target_value = orig_min - epsilon
                elif max_value is not None:
                    # Only have target max bound - go just below it
                    target_value = max_value - epsilon
                else:
                    # Bias toward lower bound to escape original class
                    target_value = min_value + (max_value - min_value) * (
                        0.5 - SAMPLE_GEN_ESCAPE_BIAS * self.escape_pressure
                    )
            else:
                # Default: keep original value if within bounds, otherwise use midpoint
                if min_value <= original_value <= max_value:
                    target_value = original_value
                else:
                    target_value = (min_value + max_value) / 2

            # Clip to target bounds and set
            adjusted_sample[feature] = np.clip(target_value, min_value, max_value)
            
            # Store feature bounds info for potential retry
            # Use RAW target constraints for search space (no expansion beyond DPG bounds)
            search_min = raw_target_min if raw_target_min is not None else min_value
            search_max = raw_target_max if raw_target_max is not None else max_value
            
            feature_bounds_info[feature] = {
                'min': search_min,
                'max': search_max,
                'escape_dir': escape_dir,
                'original': original_value,
            }
            
            if self.verbose:
                delta = adjusted_sample[feature] - original_value
                escape_info = f" (escape: {escape_dir})" if escape_dir != "both" else ""
                actionable_info = ""
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionable_info = f" [{self.dict_non_actionable[feature]}]"
                print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {adjusted_sample[feature]:.4f} (Δ={delta:+.4f}){escape_info}{actionable_info}")
        if self.verbose:
            print(f"[VERBOSE-DPG] --------------------------------------------------------") 

        # Validate initial sample
        sample_keys = list(sample.keys())
        is_valid, margin, pred = self._validate_sample_prediction(
            adjusted_sample, target_class, sample_keys
        )

        if is_valid:
            if self.verbose:
                print(f"[VERBOSE-DPG] ✓ Sample correctly predicted as class {pred} (margin: {margin:.3f})")
            return adjusted_sample

        # Initial attempt failed - retry with binary search on non-overlapping features
        if self.verbose:
            print(f"[VERBOSE-DPG] Initial sample predicted as class {pred}, not target {target_class}")
            print(f"[VERBOSE-DPG] Starting binary search retry on features with clear escape directions...")

        # Get non-overlapping features (clearest escape path) from boundary analysis
        retry_features = []
        if boundary_analysis:
            non_overlapping = boundary_analysis.get("non_overlapping", [])
            escape_directions = boundary_analysis.get("escape_direction", {})
            
            for feature in sample.keys():
                norm_feature = self._normalize_feature_name(feature)
                
                # Skip no_change features
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    if self.dict_non_actionable[feature] == "no_change":
                        continue
                
                # Prioritize non-overlapping features with clear escape directions
                escape_dir = escape_directions.get(norm_feature, "both")
                if norm_feature in non_overlapping or escape_dir in ["increase", "decrease"]:
                    bounds = feature_bounds_info.get(feature, {})
                    if bounds:
                        retry_features.append({
                            'feature': feature,
                            'escape_dir': escape_dir,
                            'min': bounds['min'],
                            'max': bounds['max'],
                            'original': bounds['original'],
                            'priority': 0 if norm_feature in non_overlapping else 1
                        })
        
        # Sort by priority (non-overlapping first)
        retry_features.sort(key=lambda x: x['priority'])
        
        if self.verbose:
            print(f"[VERBOSE-DPG] Features to retry: {[f['feature'] for f in retry_features]}")

        # Binary search each feature until we find a valid sample
        for feat_info in retry_features:
            feature = feat_info['feature']
            v_min = feat_info['min']
            v_max = feat_info['max']
            escape_dir = feat_info['escape_dir']
            original_value = feat_info['original']
            
            # Apply actionability constraints to search bounds
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    v_min = max(v_min, original_value)
                elif actionability == "non_increasing":
                    v_max = min(v_max, original_value)
            
            # Fix inverted bounds (can happen with decrease escape direction)
            if v_min > v_max:
                v_min, v_max = v_max, v_min
                if self.verbose:
                    print(f"[VERBOSE-DPG]   Swapped bounds for {feature}: [{v_min:.4f}, {v_max:.4f}]")
            
            if v_min >= v_max:
                if self.verbose:
                    print(f"[VERBOSE-DPG]   Skipping {feature}: invalid bounds [{v_min}, {v_max}]")
                continue
            
            if self.verbose:
                print(f"[VERBOSE-DPG]   Searching {feature} in [{v_min:.4f}, {v_max:.4f}] (escape: {escape_dir})")
            
            found_value = self._binary_search_feature(
                adjusted_sample, feature, v_min, v_max, target_class,
                original_value, escape_dir, eps=0.01, max_iter=20
            )
            
            if found_value is not None:
                adjusted_sample[feature] = found_value
                
                # Validate the updated sample
                is_valid, margin, pred = self._validate_sample_prediction(
                    adjusted_sample, target_class, sample_keys
                )
                
                if is_valid:
                    if self.verbose:
                        delta = found_value - original_value
                        print(f"[VERBOSE-DPG] ✓ Binary search success! {feature}: {original_value:.4f} → {found_value:.4f} (Δ={delta:+.4f})")
                        print(f"[VERBOSE-DPG] ✓ Sample now predicted as class {pred} (margin: {margin:.3f})")
                    return adjusted_sample
                else:
                    if self.verbose:
                        print(f"[VERBOSE-DPG]   {feature} updated but sample still predicted as {pred}")
            else:
                if self.verbose:
                    print(f"[VERBOSE-DPG]   No valid value found for {feature}")

        # Single-feature search failed - try progressive depth search (all features together)
        if self.verbose:
            print("[VERBOSE-DPG] Single-feature search exhausted, trying progressive depth search...")
        
        progressive_result = self._progressive_depth_search(
            sample, feature_bounds_info, target_class, eps=0.01, max_iter=30
        )
        
        if progressive_result is not None:
            is_valid, margin, pred = self._validate_sample_prediction(
                progressive_result, target_class, sample_keys
            )
            if is_valid:
                if self.verbose:
                    print(f"[VERBOSE-DPG] ✓ Progressive depth search success! (margin: {margin:.3f})")
                return progressive_result

        # Last resort: random sampling within constraint space
        if self.verbose:
            print("[VERBOSE-DPG] Progressive depth failed, trying random sampling...")
        
        random_result = self._random_sample_search(
            sample, feature_bounds_info, target_class, n_samples=500
        )
        
        if random_result is not None:
            is_valid, margin, pred = self._validate_sample_prediction(
                random_result, target_class, sample_keys
            )
            if is_valid:
                if self.verbose:
                    print(f"[VERBOSE-DPG] ✓ Random sampling success! (margin: {margin:.3f})")
                return random_result

        # All retries exhausted
        if self.verbose:
            print(f"[VERBOSE-DPG] WARNING: All search methods exhausted, returning best attempt (predicted as {pred})")
            
        return adjusted_sample

    def find_nearest_counterfactual(
        self,
        sample,
        target_class,
        X_train=None,
        y_train=None,
        metric="euclidean",
        validate_prediction=True,
    ):
        """
        Find the nearest training sample that is predicted as the target class.
        This is a simple but effective fallback when GA-based search fails.

        Args:
            sample (dict): The original sample with feature values
            target_class (int): The target class for counterfactual
            X_train (DataFrame): Training data features (optional, uses model's training data if available)
            y_train (Series): Training data labels (optional)
            metric (str): Distance metric to use
            validate_prediction (bool): If True, only return samples that model predicts as target_class

        Returns:
            dict: The nearest valid counterfactual or None
        """
        feature_names = list(sample.keys())
        sample_array = np.array([sample[f] for f in feature_names]).reshape(1, -1)

        # Try to get training data from the model if not provided
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            if self.verbose:
                print("Warning: No training data available for nearest neighbor search")
            return None

        # Ensure consistent format
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
            feature_names = list(X_train.columns)
        else:
            X_train_array = np.array(X_train)

        y_train_array = np.array(y_train)

        # Get samples of the target class
        target_mask = y_train_array == target_class
        target_samples = X_train_array[target_mask]
        target_indices = np.where(target_mask)[0]

        if len(target_samples) == 0:
            if self.verbose:
                print(f"No samples of target class {target_class} in training data")
            return None

        # Compute distances to all target class samples
        distances = cdist(sample_array, target_samples, metric=metric)[0]

        # Sort by distance
        sorted_indices = np.argsort(distances)

        # Find the nearest sample that is predicted as target class
        for idx in sorted_indices:
            candidate = target_samples[idx].reshape(1, -1)

            if validate_prediction:
                # Check that the model predicts this as target class with sufficient margin
                try:
                    if self.feature_names is not None:
                        candidate_df = pd.DataFrame(
                            candidate, columns=self.feature_names
                        )
                        pred = self.model.predict(candidate_df)[0]
                        proba = self.model.predict_proba(candidate_df)[0]
                    else:
                        pred = self.model.predict(candidate)[0]
                        proba = self.model.predict_proba(candidate)[0]

                    if pred != target_class:
                        continue  # Skip samples misclassified by model

                    # Check probability margin
                    # NOTE: proba is indexed by position, not by class label.
                    # Use model.classes_ to find the correct index for target_class
                    if hasattr(self.model, "classes_"):
                        class_list = list(self.model.classes_)
                        if target_class in class_list:
                            target_idx = class_list.index(target_class)
                        else:
                            target_idx = target_class  # Fallback to direct indexing
                    else:
                        target_idx = target_class
                    target_prob = proba[target_idx]
                    sorted_probs = np.sort(proba)[::-1]
                    second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                    margin = target_prob - second_best_prob

                    if margin < self.min_probability_margin:
                        if self.verbose:
                            print(
                                f"  Skipping candidate with weak margin: {margin:.3f} < {self.min_probability_margin}"
                            )
                        continue  # Skip samples with weak probability margin
                except Exception as e:
                    if self.verbose:
                        print(f"Prediction failed: {e}")
                    continue

            # Convert to dict
            candidate_dict = {
                feature_names[i]: candidate[0][i] for i in range(len(feature_names))
            }

            if self.verbose:
                print(f"Found nearest counterfactual at distance {distances[idx]:.2f}")
                # Check constraint validity
                if self.constraints and self.constraint_validator:
                    original_class = self.model.predict(pd.DataFrame([sample]))[0]
                    is_valid, penalty = self.constraint_validator.validate_constraints(
                        candidate_dict,
                        sample,
                        target_class,
                        original_class=original_class,
                    )
                    print(f"  DPG constraint valid: {is_valid}, penalty: {penalty:.2f}")

            return candidate_dict

        if self.verbose:
            print(
                f"No valid counterfactual found among {len(target_samples)} target class samples"
            )
        return None
