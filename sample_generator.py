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

            # Calculate probability margin (even if prediction is wrong, for debugging)
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
            
            if pred != target_class:
                # Return negative margin to show how far we are from flipping
                # (target_prob - max_other_prob will be negative)
                max_other_prob = max(p for i, p in enumerate(proba) if i != target_idx)
                margin = target_prob - max_other_prob
                return False, margin, pred

            # Calculate positive positive probability margin
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

    def _asymmetric_depth_search(self, sample, feature_bounds_info, target_class, eps=0.01):
        """
        Asymmetric depth search: Try pushing INDIVIDUAL features before combinations.
        This finds minimal counterfactuals by discovering which features matter most.
        
        Strategy:
        1. Try each feature individually at increasing depths (others at original/minimal)
        2. Try pairs of features with one at high depth, other at medium
        3. Returns the first valid sample with smallest total change
        
        This is more efficient than uniform progressive search for finding minimal changes.
        """
        sample_keys = list(sample.keys())
        
        # Build searchable features list (same logic as progressive depth search)
        searchable_features = []
        fixed_features = {}
        
        for feature, bounds in feature_bounds_info.items():
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                if self.dict_non_actionable[feature] == "no_change":
                    continue
            
            v_min = bounds['min']
            v_max = bounds['max']
            escape_dir = bounds['escape_dir']
            original_value = bounds['original']
            raw_target_min = bounds.get('raw_target_min')
            raw_target_max = bounds.get('raw_target_max')
            
            # Override escape direction based on actual sample position
            if raw_target_min is not None and raw_target_max is not None:
                if original_value < raw_target_min and escape_dir == "decrease":
                    escape_dir = "increase"
                elif original_value > raw_target_max and escape_dir == "increase":
                    escape_dir = "decrease"
            elif raw_target_min is not None and original_value < raw_target_min and escape_dir == "decrease":
                escape_dir = "increase"
            elif raw_target_max is not None and original_value > raw_target_max and escape_dir == "increase":
                escape_dir = "decrease"
            
            # Apply actionability constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    v_min = max(v_min, original_value)
                elif actionability == "non_increasing":
                    v_max = min(v_max, original_value)
            
            if v_min > v_max:
                v_min, v_max = v_max, v_min
            
            if abs(v_min - v_max) < 1e-6:
                fixed_features[feature] = v_min
                continue
            
            if v_min >= v_max:
                continue
            
            # Calculate target value based on escape direction
            if escape_dir == "increase":
                target_val = v_max - eps
            elif escape_dir == "decrease":
                target_val = v_min + eps
            else:
                # For "both", default to max but will try both directions
                target_val = v_max - eps
            
            searchable_features.append({
                'feature': feature,
                'original': original_value,
                'target': target_val,
                'v_min': v_min,
                'v_max': v_max,
                'escape_dir': escape_dir,
            })
        
        if not searchable_features:
            return None
        
        if self.verbose:
            print(f"[VERBOSE-DPG] Asymmetric search on {len(searchable_features)} features:")
            for f in searchable_features:
                print(f"[VERBOSE-DPG]   {f['feature']}: orig={f['original']:.2f} → target={f['target']:.2f} (escape={f['escape_dir']})")
        
        best_result = None
        best_change = float('inf')
        
        # Phase 1: Try each feature individually at high depth (others at original)
        if self.verbose:
            print(f"[VERBOSE-DPG] Phase 1: Single-feature push...")
        
        for primary_feat in searchable_features:
            test_sample = sample.copy()
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            
            # Push primary feature to target, others stay at original
            test_sample[primary_feat['feature']] = primary_feat['target']
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            
            if is_valid:
                total_change = abs(primary_feat['target'] - primary_feat['original'])
                if total_change < best_change:
                    best_change = total_change
                    best_result = test_sample.copy()
                    if self.verbose:
                        print(f"[VERBOSE-DPG]   ✓ {primary_feat['feature']} alone works! (Δ={total_change:.2f})")
        
        if best_result is not None:
            return best_result
        
        # Phase 2: Try pairs of features (one at high, one at medium)
        if self.verbose:
            print(f"[VERBOSE-DPG] Phase 2: Feature pairs...")
        
        import itertools
        for feat1, feat2 in itertools.combinations(searchable_features, 2):
            test_sample = sample.copy()
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            
            # Both at target depth
            test_sample[feat1['feature']] = feat1['target']
            test_sample[feat2['feature']] = feat2['target']
            
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            
            if is_valid:
                total_change = (abs(feat1['target'] - feat1['original']) + 
                               abs(feat2['target'] - feat2['original']))
                if total_change < best_change:
                    best_change = total_change
                    best_result = test_sample.copy()
                    if self.verbose:
                        print(f"[VERBOSE-DPG]   ✓ {feat1['feature']}+{feat2['feature']} works! (Δ={total_change:.2f})")
        
        if best_result is not None:
            return best_result
        
        # Phase 3: Try triplets
        if len(searchable_features) >= 3:
            if self.verbose:
                print(f"[VERBOSE-DPG] Phase 3: Feature triplets...")
            
            for feat1, feat2, feat3 in itertools.combinations(searchable_features, 3):
                test_sample = sample.copy()
                for feature, fixed_val in fixed_features.items():
                    test_sample[feature] = fixed_val
                
                test_sample[feat1['feature']] = feat1['target']
                test_sample[feat2['feature']] = feat2['target']
                test_sample[feat3['feature']] = feat3['target']
                
                is_valid, margin, pred = self._validate_sample_prediction(
                    test_sample, target_class, sample_keys
                )
                
                if is_valid:
                    total_change = (abs(feat1['target'] - feat1['original']) + 
                                   abs(feat2['target'] - feat2['original']) +
                                   abs(feat3['target'] - feat3['original']))
                    if total_change < best_change:
                        best_change = total_change
                        best_result = test_sample.copy()
                        if self.verbose:
                            print(f"[VERBOSE-DPG]   ✓ Triplet works! (Δ={total_change:.2f})")
        
        return best_result

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
        # Also track features with fixed bounds (must be set to exact value)
        searchable_features = []
        fixed_features = {}  # feature -> fixed_value
        
        for feature, bounds in feature_bounds_info.items():
            # Skip no_change features
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                if self.dict_non_actionable[feature] == "no_change":
                    continue
            
            v_min = bounds['min']
            v_max = bounds['max']
            escape_dir = bounds['escape_dir']
            original_value = bounds['original']
            raw_target_min = bounds.get('raw_target_min')
            raw_target_max = bounds.get('raw_target_max')
            
            # OVERRIDE escape direction based on actual sample position
            # (same logic as in get_valid_sample)
            if raw_target_min is not None and raw_target_max is not None:
                if original_value < raw_target_min and escape_dir == "decrease":
                    escape_dir = "increase"
                elif original_value > raw_target_max and escape_dir == "increase":
                    escape_dir = "decrease"
            elif raw_target_min is not None and original_value < raw_target_min and escape_dir == "decrease":
                escape_dir = "increase"
            elif raw_target_max is not None and original_value > raw_target_max and escape_dir == "increase":
                escape_dir = "decrease"
            
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
            
            # Handle equal bounds: feature has a FIXED required value
            if abs(v_min - v_max) < 1e-6:  # Essentially equal
                fixed_features[feature] = v_min  # Must be this exact value
                if self.verbose:
                    print(f"[VERBOSE-DPG]   Fixed feature {feature} = {v_min:.4f} (equal bounds)")
                continue
            
            # Skip invalid bounds (after potential swap)
            if v_min >= v_max:
                continue
            
            # Calculate start (minimal change) and end (maximal change) values
            # based on escape direction
            # start = value at depth=0 (minimal change from original)
            # end = value at depth=1 (maximal exploration of bounds)
            #
            # KEY INSIGHT: If the escape direction can't be followed within bounds
            # (e.g., escape=decrease but original is already at v_min), we should
            # explore the OPPOSITE direction instead - any movement is better than none!
            
            if escape_dir == "increase":
                # Need to increase: start near original, end at v_max
                start_val = max(original_value, v_min + eps)
                end_val = v_max - eps
                
                # If no room to increase (original already at or near v_max), try decreasing instead
                # Use a generous threshold - if range is less than 1% of total bounds, reverse
                range_threshold = max(eps * 2, (v_max - v_min) * 0.01)
                if abs(end_val - start_val) <= range_threshold:
                    start_val = min(original_value, v_max - eps)
                    end_val = v_min + eps
                    if self.verbose:
                        print(f"[VERBOSE-DPG]       {feature}: Can't increase (orig={original_value:.2f} near v_max={v_max:.2f}), reversing to decrease")
                        
            elif escape_dir == "decrease":
                # Need to decrease: start near original, end at v_min
                start_val = min(original_value, v_max - eps)
                end_val = v_min + eps
                
                # If no room to decrease (original already at or near v_min), try increasing instead
                # Use a generous threshold - if range is less than 1% of total bounds, reverse
                range_threshold = max(eps * 2, (v_max - v_min) * 0.01)
                if abs(end_val - start_val) <= range_threshold:
                    start_val = max(original_value, v_min + eps)
                    end_val = v_max - eps
                    if self.verbose:
                        print(f"[VERBOSE-DPG]       {feature}: Can't decrease (orig={original_value:.2f} near v_min={v_min:.2f}), reversing to increase")
            else:
                # 'both' direction: we don't know which way improves classification
                # Test BOTH directions with a quick probe to see which improves margin
                start_val = original_value if v_min <= original_value <= v_max else (v_min + v_max) / 2
                
                # Default: try toward max first
                end_val = v_max - eps
                
                # We'll determine the best direction during the search phase
                # For now, mark this feature for bidirectional testing
            
            # Ensure we have a meaningful search range
            range_threshold = max(eps * 2, (v_max - v_min) * 0.01)
            if abs(start_val - end_val) <= range_threshold:
                # If start and end are too close, try the opposite direction
                if end_val > start_val:
                    end_val = v_min + eps
                else:
                    end_val = v_max - eps
            
            if self.verbose:
                direction = "→ increase" if end_val > start_val else "→ decrease"
                print(f"[VERBOSE-DPG]     Searchable: {feature} start={start_val:.4f} end={end_val:.4f} {direction} (escape={escape_dir}, bounds=[{v_min:.4f}, {v_max:.4f}])")
            
            searchable_features.append({
                'feature': feature,
                'start': start_val,
                'end': end_val,
                'v_min': v_min,
                'v_max': v_max,
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
            
            # First set all fixed features to their required values
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            
            # Then interpolate searchable features based on depth
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
                status = "valid" if is_valid else f"invalid (pred={pred}, margin={margin:+.4f})"
                print(f"[VERBOSE-DPG]     Depth search iter {iteration}: depth={mid_depth:.3f} {status}")
        
        # If binary search failed, try extremes including depth=1.0
        if best_valid_sample is None and self.verbose:
            print(f"[VERBOSE-DPG]   Binary search completed without finding valid sample. Trying depth=1.0...")
            test_sample = sample.copy()
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            for feat_info in searchable_features:
                feature = feat_info['feature']
                test_sample[feature] = feat_info['end']  # Maximum depth
            is_valid, margin, pred = self._validate_sample_prediction(
                test_sample, target_class, sample_keys
            )
            print(f"[VERBOSE-DPG]     Depth=1.0: pred={pred}, margin={margin:+.4f}, target={target_class}")
            if is_valid:
                best_valid_sample = test_sample.copy()
                best_depth = 1.0
        
        # If initial direction failed, try REVERSING all search directions
        if best_valid_sample is None:
            if self.verbose:
                print(f"[VERBOSE-DPG]   Initial direction failed, trying REVERSED directions...")
            
            # Reverse all searchable features to try opposite direction
            reversed_features = []
            for feat_info in searchable_features:
                v_min = feat_info.get('v_min', feat_info['start'])
                v_max = feat_info.get('v_max', feat_info['end'])
                original = feat_info['original']
                start = feat_info['start']
                end = feat_info['end']
                
                # Reverse: if was going up, go down; if was going down, go up
                if end > start:
                    # Was going up, now go down
                    new_end = v_min + eps
                else:
                    # Was going down, now go up
                    new_end = v_max - eps
                
                new_start = original if v_min <= original <= v_max else start
                
                if self.verbose:
                    direction = "→ increase" if new_end > new_start else "→ decrease"
                    print(f"[VERBOSE-DPG]     REVERSED {feat_info['feature']}: {new_start:.4f} to {new_end:.4f} {direction}")
                
                reversed_features.append({
                    'feature': feat_info['feature'],
                    'start': new_start,
                    'end': new_end,
                    'original': original,
                })
            
            # Run binary search again with reversed directions
            low_depth, high_depth = 0.0, 1.0
            for iteration in range(max_iter):
                if abs(high_depth - low_depth) < eps:
                    break
                
                mid_depth = (low_depth + high_depth) / 2
                test_sample = sample.copy()
                
                for feature, fixed_val in fixed_features.items():
                    test_sample[feature] = fixed_val
                
                for feat_info in reversed_features:
                    feature = feat_info['feature']
                    start = feat_info['start']
                    end = feat_info['end']
                    test_sample[feature] = start + mid_depth * (end - start)
                
                is_valid, margin, pred = self._validate_sample_prediction(
                    test_sample, target_class, sample_keys
                )
                
                if is_valid:
                    if mid_depth < best_depth:
                        best_valid_sample = test_sample.copy()
                        best_depth = mid_depth
                    high_depth = mid_depth
                else:
                    low_depth = mid_depth
                
                if self.verbose and iteration % 5 == 0:
                    status = "valid" if is_valid else f"invalid (pred={pred}, margin={margin:+.4f})"
                    print(f"[VERBOSE-DPG]     Reversed depth iter {iteration}: depth={mid_depth:.3f} {status}")
            
            # Try depth=1.0 with reversed directions
            if best_valid_sample is None:
                test_sample = sample.copy()
                for feature, fixed_val in fixed_features.items():
                    test_sample[feature] = fixed_val
                for feat_info in reversed_features:
                    feature = feat_info['feature']
                    test_sample[feature] = feat_info['end']
                is_valid, margin, pred = self._validate_sample_prediction(
                    test_sample, target_class, sample_keys
                )
                if self.verbose:
                    print(f"[VERBOSE-DPG]     Reversed depth=1.0: pred={pred}, margin={margin:+.4f}")
                if is_valid:
                    best_valid_sample = test_sample.copy()
                    best_depth = 1.0
        
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
            
            # First set all fixed features to their required values
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            
            # Then set searchable features based on depth
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
        # Also track features with fixed bounds (must be set to exact value)
        searchable_features = []
        fixed_features = {}  # feature -> fixed_value
        
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
            
            # Handle equal bounds: feature has a FIXED required value
            if abs(v_min - v_max) < 1e-6:  # Essentially equal
                fixed_features[feature] = v_min  # Must be this exact value
                continue
            
            if v_min < v_max:
                searchable_features.append({
                    'feature': feature,
                    'min': v_min,
                    'max': v_max,
                    'original': original_value,
                })
        
        if not searchable_features and not fixed_features:
            return None
        
        best_sample = None
        best_distance = float('inf')
        
        for i in range(n_samples):
            test_sample = sample.copy()
            
            # First set all fixed features to their required values
            for feature, fixed_val in fixed_features.items():
                test_sample[feature] = fixed_val
            
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

    def get_valid_sample(self, sample, target_class, original_class, weak_constraints):
        """
        Generate a valid sample that meets all constraints for the specified target class
        while respecting actionable changes.

        Enhanced with dual-boundary support: the sample is biased to move away from original class bounds toward target class bounds.

        Args:
            sample (dict): The sample with feature values.
            target_class (int): The target class for filtering constraints.
            original_class (int): The original class for escape-aware generation.
            weak_constraints (bool): If True, search bounds span from original value to target constraint,
                                    allowing CFs to be found along the path. If False, search only within
                                    exact DPG target bounds.

        Returns:
            dict: A valid sample that meets all constraints for the target class
                  and respects actionable changes.
        """
        if self.verbose:
            print(f"[VERBOSE-DPG] Generating valid sample for target class {target_class} - get_valid_sample")
            if original_class is not None:
                print(f"[VERBOSE-DPG]   Original class: {original_class} (escape-aware generation)")
            print(f"[VERBOSE-DPG]   weak_constraints={weak_constraints}")
        
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

            # If no constraint exists for this feature, keep original value unchanged
            if matching_constraint is None:
                adjusted_sample[feature] = original_value
                if self.verbose:
                    print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {original_value:.4f} (Δ=+0.0000) [no constraint]")
                continue

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
            
            # CRITICAL FIX: Override escape direction based on actual sample value position
            # The boundary-based escape_dir only considers class constraints, not where
            # the actual sample value is. If the sample is outside target bounds,
            # the direction must be toward those bounds regardless of class-based escape_dir.
            if raw_target_min is not None and raw_target_max is not None:
                if original_value < raw_target_min:
                    # Sample is BELOW target bounds - must INCREASE to reach target
                    if escape_dir == "decrease":
                        if self.verbose:
                            print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                        escape_dir = "increase"
                elif original_value > raw_target_max:
                    # Sample is ABOVE target bounds - must DECREASE to reach target
                    if escape_dir == "increase":
                        if self.verbose:
                            print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                        escape_dir = "decrease"
            elif raw_target_min is not None and original_value < raw_target_min:
                # Only min bound exists and sample is below it
                if escape_dir == "decrease":
                    if self.verbose:
                        print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                    escape_dir = "increase"
            elif raw_target_max is not None and original_value > raw_target_max:
                # Only max bound exists and sample is above it
                if escape_dir == "increase":
                    if self.verbose:
                        print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                    escape_dir = "decrease"

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
                # Target requires higher values - start at target's min boundary
                # (Ignore original escape logic, go directly to target bounds)
                if min_value is not None and min_value != -np.inf:
                    target_value = min_value + epsilon
                else:
                    # No target min bound - use midpoint biased upward
                    target_value = min_value + (max_value - min_value) * (
                        0.5 + SAMPLE_GEN_ESCAPE_BIAS * self.escape_pressure
                    )

            elif escape_dir == "decrease":
                # Target requires lower values - start at target's max boundary
                # (Ignore original escape logic, go directly to target bounds)
                if max_value is not None and max_value != np.inf:
                    target_value = max_value - epsilon
                else:
                    # No target max bound - use midpoint biased downward
                    target_value = min_value + (max_value - min_value) * (
                        0.5 - SAMPLE_GEN_ESCAPE_BIAS * self.escape_pressure
                    )
            else:
                # Default (escape_dir == "both"): minimal change principle
                if weak_constraints:
                    # WEAK CONSTRAINTS: bounds are extended to include original value
                    # So original is always "within bounds" - start there and let search find minimum change
                    target_value = original_value
                elif min_value <= original_value <= max_value:
                    # Already within target bounds - keep original
                    target_value = original_value
                elif original_value < min_value:
                    # Below target bounds - step just inside the minimum
                    target_value = min_value + epsilon
                else:
                    # Above target bounds - step just inside the maximum
                    target_value = max_value - epsilon

            # Clip to target bounds and set
            adjusted_sample[feature] = np.clip(target_value, min_value, max_value)
            
            # Store feature bounds info for potential retry
            # Determine search bounds based on weak_constraints mode
            if weak_constraints:
                # WEAK CONSTRAINTS: Extend DPG bounds to include original value
                # This bridges the gap between original sample and DPG target bounds
                if raw_target_min is not None:
                    search_min = min(raw_target_min, original_value)
                else:
                    search_min = original_value - 10.0 * (abs(original_value) + 1.0)
                
                if raw_target_max is not None:
                    search_max = max(raw_target_max, original_value)
                else:
                    search_max = original_value + 10.0 * (abs(original_value) + 1.0)
            else:
                # STRICT CONSTRAINTS: Search only within exact DPG target bounds
                if raw_target_min is not None:
                    search_min = raw_target_min
                else:
                    search_min = original_value - 10.0 * (abs(original_value) + 1.0)
                
                if raw_target_max is not None:
                    search_max = raw_target_max
                else:
                    search_max = original_value + 10.0 * (abs(original_value) + 1.0)
            
            feature_bounds_info[feature] = {
                'min': search_min,
                'max': search_max,
                'escape_dir': escape_dir,
                'original': original_value,
                'raw_target_min': raw_target_min,
                'raw_target_max': raw_target_max,
            }
            
            if self.verbose:
                delta = adjusted_sample[feature] - original_value
                escape_info = f" (escape: {escape_dir})" if escape_dir != "both" else ""
                actionable_info = ""
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionable_info = f" [{self.dict_non_actionable[feature]}]"
                print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {adjusted_sample[feature]:.4f} (Δ={delta:+.4f}){escape_info}{actionable_info}    --- [search: {search_min:.4f}, {search_max:.4f}]")
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
                        original_value = bounds['original']
                        raw_target_min = bounds.get('raw_target_min')
                        raw_target_max = bounds.get('raw_target_max')
                        
                        # OVERRIDE escape direction based on actual sample position
                        # (same logic as in main loop)
                        if raw_target_min is not None and raw_target_max is not None:
                            if original_value < raw_target_min and escape_dir == "decrease":
                                if self.verbose:
                                    print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                                escape_dir = "increase"
                            elif original_value > raw_target_max and escape_dir == "increase":
                                if self.verbose:
                                    print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                                escape_dir = "decrease"
                        elif raw_target_min is not None and original_value < raw_target_min and escape_dir == "decrease":
                            if self.verbose:
                                print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                            escape_dir = "increase"
                        elif raw_target_max is not None and original_value > raw_target_max and escape_dir == "increase":
                            if self.verbose:
                                print(f"[VERBOSE-DPG]     OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                            escape_dir = "decrease"
                        
                        retry_features.append({
                            'feature': feature,
                            'escape_dir': escape_dir,
                            'min': bounds['min'],
                            'max': bounds['max'],
                            'original': original_value,
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

        # Single-feature search failed - try asymmetric depth search (feature combinations)
        if self.verbose:
            print("[VERBOSE-DPG] Single-feature search exhausted, trying asymmetric depth search...")
        
        asymmetric_result = self._asymmetric_depth_search(
            sample, feature_bounds_info, target_class, eps=0.01
        )
        
        if asymmetric_result is not None:
            is_valid, margin, pred = self._validate_sample_prediction(
                asymmetric_result, target_class, sample_keys
            )
            if is_valid:
                if self.verbose:
                    print(f"[VERBOSE-DPG] ✓ Asymmetric depth search success! (margin: {margin:.3f})")
                return asymmetric_result
        
        # Asymmetric search failed - try uniform progressive depth search (all features together)
        if self.verbose:
            print("[VERBOSE-DPG] Asymmetric search failed, trying uniform progressive depth search...")
        
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
