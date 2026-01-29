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

    def get_valid_sample(self, sample, target_class, original_class):
        """
        Generate a valid sample that meets all constraints for the specified target class
        while respecting actionable changes.

        Enhanced with dual-boundary support: when original_class is provided, the sample
        is biased to move away from original class bounds toward target class bounds.

        Args:
            sample (dict): The sample with feature values.
            target_class (int): The target class for filtering constraints.
            original_class (int, optional): The original class for escape-aware generation.

        Returns:
            dict: A valid sample that meets all constraints for the target class
                  and respects actionable changes.
        """
        if self.verbose:
            print(f"[VERBOSE-DPG] Generating valid sample for target class {target_class} - get_valid_sample")
            if original_class is not None:
                print(f"[VERBOSE-DPG]   Original class: {original_class} (escape-aware generation)")
        
        adjusted_sample = sample.copy()  # Start with the original values
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
                min_value = (
                    matching_constraint.get("min")
                    if matching_constraint.get("min") is not None
                    else -np.inf
                )
                max_value = (
                    matching_constraint.get("max")
                    if matching_constraint.get("max") is not None
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
            
            if self.verbose:
                delta = adjusted_sample[feature] - original_value
                escape_info = f" (escape: {escape_dir})" if escape_dir != "both" else ""
                actionable_info = ""
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionable_info = f" [{self.dict_non_actionable[feature]}]"
                print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {adjusted_sample[feature]:.4f} (Δ={delta:+.4f}){escape_info}{actionable_info}")
        if self.verbose:
            print(f"[VERBOSE-DPG] --------------------------------------------------------") 
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
