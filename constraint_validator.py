"""
ConstraintValidator: Validates constraints and actionability for counterfactuals.

Extracted from CounterFactualModel.py to provide focused constraint validation
functionality for counterfactual generation.
"""

import numpy as np
import pandas as pd

from utils.feature_utils import normalize_feature_name, features_match


class ConstraintValidator:
    """
    Validates constraints and checks actionability of counterfactual changes.
    """

    def __init__(
        self,
        model,
        constraints,
        dict_non_actionable=None,
        feature_names=None,
        boundary_analyzer=None,
        verbose=False,
    ):
        """
        Initialize the ConstraintValidator.

        Args:
            model: The machine learning model used for predictions.
            constraints (dict): Feature constraints per class.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints
                (non_decreasing, non_increasing, no_change).
            feature_names: List of feature names from the model.
            boundary_analyzer: Optional BoundaryAnalyzer instance for overlap analysis.
            verbose (bool): Whether to print detailed logging.
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable
        self.feature_names = feature_names
        self.boundary_analyzer = boundary_analyzer
        self.verbose = verbose

    def _normalize_feature_name(self, feature):
        """Delegate to feature_utils for feature name normalization."""
        return normalize_feature_name(feature)

    def _features_match(self, feature1, feature2):
        """Delegate to feature_utils for feature matching."""
        return features_match(feature1, feature2)

    def is_actionable_change(self, counterfactual_sample, original_sample):
        """
        Check if changes in features are actionable based on constraints.

        Args:
            counterfactual_sample (dict): The modified sample with new feature values.
            original_sample (dict): The original sample with feature values.

        Returns:
            bool: True if all changes are actionable, False otherwise.
        """
        if not self.dict_non_actionable:
            return True

        for feature, new_value in counterfactual_sample.items():
            if feature not in self.dict_non_actionable:
                continue

            original_value = original_sample.get(feature)
            constraint = self.dict_non_actionable[feature]

            if constraint == "non_decreasing" and new_value < original_value:
                if self.verbose:
                    print(f"[VERBOSE-DPG] Actionability violation: {feature} = {new_value:.4f} < {original_value:.4f} (non_decreasing constraint)")
                return False
            if constraint == "non_increasing" and new_value > original_value:
                if self.verbose:
                    print(f"[VERBOSE-DPG] Actionability violation: {feature} = {new_value:.4f} > {original_value:.4f} (non_increasing constraint)")
                return False
            if constraint == "no_change" and new_value != original_value:
                if self.verbose:
                    print(f"[VERBOSE-DPG] Actionability violation: {feature} = {new_value:.4f} != {original_value:.4f} (no_change constraint)")
                return False

        return True

    def check_validity(self, counterfactual_sample, original_sample, desired_class):
        """
        Checks the validity of a counterfactual sample.

        Parameters:
        - counterfactual_sample: Array-like, shape (n_features,), the counterfactual sample.
        - original_sample: Array-like, shape (n_features,), the original input sample.
        - desired_class: The desired class label.

        Returns:
        - True if the predicted class matches the desired class and the sample is different from the original.
        - False if the predicted class does not match the desired class or the sample is identical to the original.
        """
        # Ensure the input samples are numpy arrays
        counterfactual_sample = np.array(counterfactual_sample).reshape(1, -1)
        original_sample = np.array(original_sample).reshape(1, -1)

        # Check if the counterfactual sample is different from the original sample
        if np.array_equal(counterfactual_sample, original_sample):
            return False  # Return False if the samples are identical

        # Predict the class for the counterfactual sample
        # Convert to DataFrame with feature names if available for model compatibility
        if self.feature_names is not None:
            counterfactual_df = pd.DataFrame(
                counterfactual_sample, columns=self.feature_names
            )
            predicted_class = self.model.predict(counterfactual_df)[0]
        else:
            predicted_class = self.model.predict(counterfactual_sample)[0]

        # Check if the predicted class matches the desired class
        if predicted_class == desired_class:
            return True
        else:
            return False

    def validate_constraints(
        self, S_prime, sample, target_class, original_class=None, strict_mode=True
    ):
        """
        Validate if the modified sample S_prime meets all constraints for the specified target class.

        Args:
            S_prime (dict): Modified sample with feature values.
            sample (dict): The original sample with feature values.
            target_class (int): The target class for filtering constraints.
            original_class (int, optional): The original class (for overlap analysis).
            strict_mode (bool): If True, penalize values in non-target class bounds.
                               If False (relaxed mode), only check target class constraints.

        Returns:
            (bool, float): Tuple of whether the changes are valid and a penalty score.
        """
        penalty = 0.0
        valid_change = True

        # Filter the constraints for the specified target class
        class_constraints = self.constraints.get(str("Class " + str(target_class)), [])

        if self.verbose:
            is_cf_valid = self.check_validity(
                [S_prime[f] for f in self.feature_names],
                [sample[f] for f in self.feature_names],
                target_class,
            )
            validity_str = "valid" if is_cf_valid else "invalid"
            validity_sign = "✓" if is_cf_valid else "✗"
            print(f"[VERBOSE-DPG] Attempting to validate a {validity_str} counterfactual against constraints.     -  {validity_sign}")
            print(f"[VERBOSE-DPG] Validating constraints for target class {target_class}")

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Validate numerical constraints specific to the target class
                matching_constraint = next(
                    (
                        condition
                        for condition in class_constraints
                        if self._features_match(condition["feature"], feature)
                    ),
                    None,
                )

                if matching_constraint:
                    min_val = matching_constraint.get("min")
                    max_val = matching_constraint.get("max")

                    delta = new_value - original_value
                    if self.verbose:
                        in_bounds = "✓" if (min_val is None or new_value >= min_val) and (max_val is None or new_value <= max_val) else "✗"
                        print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {new_value:.4f} (Δ={delta:+.4f}) [{min_val}, {max_val}] {in_bounds}")

                    # Check if the new value violates min constraint
                    if min_val is not None and new_value < min_val:
                        valid_change = False
                        violation = abs(new_value - min_val)
                        penalty += violation
                        if self.verbose:
                            print(f"[VERBOSE-DPG]     Min violation: {new_value:.4f} < {min_val:.4f}, penalty += {violation:.4f}")

                    # Check if the new value violates max constraint
                    if max_val is not None and new_value > max_val:
                        valid_change = False
                        violation = abs(new_value - max_val)
                        penalty += violation
                        if self.verbose:
                            print(f"[VERBOSE-DPG]     Max violation: {new_value:.4f} > {max_val:.4f}, penalty += {violation:.4f}")
                elif self.verbose and new_value != original_value:
                    delta = new_value - original_value
                    print(f"[VERBOSE-DPG]   {feature}: {original_value:.4f} → {new_value:.4f} (Δ={delta:+.4f}) [no constraint]")

        # In relaxed mode, skip non-target class penalty (used when constraints overlap significantly)
        if not strict_mode:
            return valid_change, penalty

        # Get boundary overlap analysis to identify non-overlapping features
        # Only penalize non-target class violations for NON-OVERLAPPING features
        non_overlapping_features = set()
        if original_class is not None and self.boundary_analyzer is not None:
            boundary_analysis = self.boundary_analyzer.analyze_boundary_overlap(
                original_class, target_class
            )
            non_overlapping_features = set(
                self._normalize_feature_name(f)
                for f in boundary_analysis.get("non_overlapping", [])
            )

        # Collect all constraints that are NOT related to the target class
        non_target_class_constraints = [
            condition
            for class_name, conditions in self.constraints.items()
            if class_name
            != "Class " + str(target_class)  # Exclude the target class constraints
            for condition in conditions
        ]

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)
            norm_feature = self._normalize_feature_name(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Only apply non-target penalty for NON-OVERLAPPING features
                # For overlapping features, being within both class constraints is acceptable
                if (
                    original_class is not None
                    and norm_feature not in non_overlapping_features
                ):
                    # This feature has overlapping constraints - skip non-target penalty
                    continue

                # Validate numerical constraints NOT related to the target class
                matching_constraint = next(
                    (
                        condition
                        for condition in non_target_class_constraints
                        if self._features_match(condition["feature"], feature)
                    ),
                    None,
                )

                if matching_constraint:
                    min_val = matching_constraint.get("min")
                    max_val = matching_constraint.get("max")

                    # Check if the new value should NOT satisfy min constraint (inverse logic for non-target classes)
                    if min_val is not None and new_value >= min_val:
                        valid_change = False
                        penalty += abs(new_value - min_val)

                    # Check if the new value should NOT satisfy max constraint (inverse logic for non-target classes)
                    if max_val is not None and new_value <= max_val:
                        valid_change = False
                        penalty += abs(new_value - max_val)

        # print('Total Penalty:', penalty)
        return valid_change, penalty
