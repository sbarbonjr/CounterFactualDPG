"""
MutationStrategy: Handles genetic algorithm mutation and crossover operations.

Extracted from CounterFactualModel.py to provide focused mutation strategy
functionality for counterfactual generation.
"""

import numpy as np
from deap import creator

from constants import (
    MUTATION_EPSILON,
    MUTATION_RANGE_SCALE_ACTIONABLE,
    MUTATION_RANGE_SCALE_GENERAL,
    MUTATION_RANGE_SCALE_BOUNDARY_PUSH,
    MUTATION_RANGE_DEFAULT,
    MUTATION_RANGE_MIN,
    MUTATION_RATE_BOOST_NON_OVERLAPPING,
    FEATURE_VALUE_PRECISION,
    UNCONSTRAINED_MUTATION_RATE_FACTOR,
)


class MutationStrategy:
    """
    Handles mutation and crossover operations for the genetic algorithm.
    """

    def __init__(
        self,
        constraints,
        dict_non_actionable=None,
        escape_pressure=0.5,
        prioritize_non_overlapping=True,
        boundary_analyzer=None,
    ):
        """
        Initialize the MutationStrategy.

        Args:
            constraints (dict): Feature constraints per class.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
            escape_pressure (float): Balance between escaping original (1.0) vs approaching target (0.0).
            prioritize_non_overlapping (bool): Prioritize mutating features with non-overlapping boundaries.
            boundary_analyzer: Optional BoundaryAnalyzer instance for overlap analysis.
        """
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable
        self.escape_pressure = escape_pressure
        self.prioritize_non_overlapping = prioritize_non_overlapping
        self.boundary_analyzer = boundary_analyzer

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

    def create_deap_individual(self, sample_dict, feature_names):
        """Create a DEAP individual from a dictionary."""
        individual = creator.Individual(sample_dict)
        return individual

    def mutate_individual(
        self,
        individual,
        sample,
        feature_names,
        mutation_rate,
        target_class=None,
        original_class=None,
        boundary_analysis=None,
    ):
        """Custom mutation operator that respects actionability and uses dual DPG constraint boundaries.

        Enhanced with dual-boundary support: mutates features to escape original class bounds
        while moving toward target class bounds, with configurable escape_pressure.

        Args:
            individual: The individual to mutate
            sample: Original sample
            feature_names: List of feature names
            mutation_rate: Probability of mutating each feature
            target_class: Target class for constraint-aware mutation
            original_class: Original class for escape-aware mutation
            boundary_analysis: Pre-computed boundary analysis (optional, computed if not provided)
        """
        # Get target class constraints if available (list of constraint dicts)
        target_constraints = []
        original_constraints = []
        if target_class is not None and self.constraints:
            target_constraints = self.constraints.get(f"Class {target_class}", [])
        if original_class is not None and self.constraints:
            original_constraints = self.constraints.get(f"Class {original_class}", [])

        # Get or compute boundary analysis for prioritization
        if (
            boundary_analysis is None
            and original_class is not None
            and target_class is not None
            and self.boundary_analyzer
        ):
            boundary_analysis = self.boundary_analyzer.analyze_boundary_overlap(
                original_class, target_class
            )

        # Determine feature mutation priority based on non-overlapping boundaries
        non_overlapping_features = set()
        escape_directions = {}
        if boundary_analysis and self.prioritize_non_overlapping:
            non_overlapping_features = set(
                self._normalize_feature_name(f)
                for f in boundary_analysis.get("non_overlapping", [])
            )
            escape_directions = boundary_analysis.get("escape_direction", {})

        # Identify features with target class constraints
        target_constrained_features = set()
        if target_class is not None and target_constraints:
            for constraint in target_constraints:
                feature_name = constraint.get("feature", "")
                if feature_name:
                    target_constrained_features.add(self._normalize_feature_name(feature_name))

        for feature in feature_names:
            norm_feature = self._normalize_feature_name(feature)

            # Adjust mutation rate: higher for non-overlapping features
            effective_mutation_rate = mutation_rate
            if (
                self.prioritize_non_overlapping
                and norm_feature in non_overlapping_features
            ):
                # Boost mutation rate for features with clear escape paths
                effective_mutation_rate = min(
                    1.0, mutation_rate * MUTATION_RATE_BOOST_NON_OVERLAPPING
                )

            # Reduce mutation rate for unconstrained features (change as last resort)
            if norm_feature not in target_constrained_features and target_constrained_features:
                effective_mutation_rate *= UNCONSTRAINED_MUTATION_RATE_FACTOR

            if np.random.rand() < effective_mutation_rate:
                # Get target constraint boundaries for this feature
                target_min, target_max = None, None
                if target_constraints:
                    matching_constraint = next(
                        (
                            c
                            for c in target_constraints
                            if self._features_match(c.get("feature", ""), feature)
                        ),
                        None,
                    )
                    if matching_constraint:
                        target_min = matching_constraint.get("min")
                        target_max = matching_constraint.get("max")

                # Get original constraint boundaries for escape direction
                orig_min, orig_max = None, None
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

                # Determine escape direction based on analysis
                escape_dir = escape_directions.get(norm_feature, "both")

                # Apply mutation based on actionability constraints
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionability = self.dict_non_actionable[feature]
                    original_value = sample[feature]

                    if actionability == "non_decreasing":
                        # Only allow increase - use escape pressure to bias toward target upper bound
                        if target_max is not None:
                            mutation_range = min(
                                MUTATION_RANGE_DEFAULT,
                                (target_max - individual[feature])
                                * MUTATION_RANGE_SCALE_ACTIONABLE,
                            )
                        else:
                            mutation_range = MUTATION_RANGE_DEFAULT
                        individual[feature] += np.random.uniform(0, mutation_range)

                    elif actionability == "non_increasing":
                        # Only allow decrease - use escape pressure to bias toward target lower bound
                        if target_min is not None:
                            mutation_range = min(
                                MUTATION_RANGE_DEFAULT,
                                (individual[feature] - target_min)
                                * MUTATION_RANGE_SCALE_ACTIONABLE,
                            )
                        else:
                            mutation_range = MUTATION_RANGE_DEFAULT
                        individual[feature] += np.random.uniform(-mutation_range, 0)

                    elif actionability == "no_change":
                        individual[feature] = original_value  # Do not change
                    else:
                        # Apply dual-boundary mutation
                        individual[feature] = self._dual_boundary_mutate(
                            individual[feature],
                            target_min,
                            target_max,
                            orig_min,
                            orig_max,
                            escape_dir,
                        )
                else:
                    # Feature not constrained by actionability - apply dual-boundary mutation
                    individual[feature] = self._dual_boundary_mutate(
                        individual[feature],
                        target_min,
                        target_max,
                        orig_min,
                        orig_max,
                        escape_dir,
                    )

                # Clip to target constraint boundaries if they exist
                if target_min is not None:
                    individual[feature] = max(target_min, individual[feature])
                if target_max is not None:
                    individual[feature] = min(target_max, individual[feature])

                # Ensure non-negative values and round
                individual[feature] = np.round(
                    max(0, individual[feature]), FEATURE_VALUE_PRECISION
                )

        return (individual,)

    def _dual_boundary_mutate(
        self,
        current_value,
        target_min,
        target_max,
        orig_min,
        orig_max,
        escape_dir="both",
    ):
        """
        Apply mutation that balances escaping original bounds and approaching target bounds.

        Uses escape_pressure parameter to control the balance:
        - escape_pressure=1.0: Fully focused on escaping original bounds
        - escape_pressure=0.0: Fully focused on approaching target bounds
        - escape_pressure=0.5 (default): Balanced approach

        Enhanced to handle boundary cases where target and origin constraints meet:
        - When origin_min=5.45 and target_max=5.45, mutate toward values just below origin_min
        - When origin_max=X and target_min>X, mutate toward values just above origin_max

        Args:
            current_value: Current feature value
            target_min, target_max: Target class bounds
            orig_min, orig_max: Original class bounds
            escape_dir: Preferred escape direction ('increase', 'decrease', 'both')

        Returns:
            float: Mutated value
        """
        escape_pressure = self.escape_pressure
        epsilon = MUTATION_EPSILON  # Small step for boundary crossing

        # Calculate mutation based on escape direction and pressure
        if escape_dir == "increase":
            # Must increase to escape original and reach target
            # Target point is the min of target range (the threshold we need to cross)
            if (
                orig_max is not None
                and target_min is not None
                and target_min >= orig_max
            ):
                # Clear boundary case: origin_max <= target_min
                # We need to cross from below orig_max to above target_min
                # Mutate toward just above the origin's max (entering target's valid range)
                if current_value <= orig_max:
                    # Still in/below original bounds - push toward the boundary
                    target_point = orig_max + epsilon
                    range_to_target = max(MUTATION_RANGE_MIN, target_point - current_value)
                    mutation_range = max(
                        MUTATION_RANGE_MIN, range_to_target * MUTATION_RANGE_SCALE_BOUNDARY_PUSH
                    )
                    return current_value + np.random.uniform(0, mutation_range)
                else:
                    # Already past origin's max - continue into target range
                    if target_max is not None:
                        mutation_range = max(
                            MUTATION_RANGE_MIN,
                            (target_max - current_value) * MUTATION_RANGE_SCALE_GENERAL,
                        )
                    else:
                        mutation_range = MUTATION_RANGE_DEFAULT
                    return current_value + np.random.uniform(0, mutation_range)
            elif target_min is not None:
                target_point = target_min
                range_to_target = max(MUTATION_RANGE_MIN, target_point - current_value)
                mutation_range = max(MUTATION_RANGE_MIN, range_to_target * MUTATION_RANGE_SCALE_GENERAL)
                # Bias toward increase
                return current_value + np.random.uniform(0, mutation_range)
            elif target_max is not None:
                # Only upper bound - move toward middle of range below it
                mutation_range = max(
                    MUTATION_RANGE_MIN, (target_max - current_value) * MUTATION_RANGE_SCALE_GENERAL
                )
                return current_value + np.random.uniform(0, mutation_range)
            else:
                return current_value + np.random.uniform(0, MUTATION_RANGE_DEFAULT)

        elif escape_dir == "decrease":
            # Must decrease to escape original and reach target
            # Target point is the max of target range (the threshold we need to cross below)
            if (
                orig_min is not None
                and target_max is not None
                and target_max <= orig_min
            ):
                # Clear boundary case: target_max <= origin_min
                # We need to cross from above orig_min to below target_max
                # Mutate toward just below the origin's min (entering target's valid range)
                if current_value >= orig_min:
                    # Still in/above original bounds - push toward the boundary
                    target_point = orig_min - epsilon
                    range_to_target = max(MUTATION_RANGE_MIN, current_value - target_point)
                    mutation_range = max(
                        MUTATION_RANGE_MIN, range_to_target * MUTATION_RANGE_SCALE_BOUNDARY_PUSH
                    )
                    return current_value - np.random.uniform(0, mutation_range)
                else:
                    # Already past origin's min - continue into target range
                    if target_min is not None:
                        mutation_range = max(
                            MUTATION_RANGE_MIN,
                            (current_value - target_min) * MUTATION_RANGE_SCALE_GENERAL,
                        )
                    else:
                        mutation_range = MUTATION_RANGE_DEFAULT
                    return current_value - np.random.uniform(0, mutation_range)
            elif target_max is not None:
                target_point = target_max
                range_to_target = max(MUTATION_RANGE_MIN, current_value - target_point)
                mutation_range = max(MUTATION_RANGE_MIN, range_to_target * MUTATION_RANGE_SCALE_GENERAL)
                # Bias toward decrease
                return current_value - np.random.uniform(0, mutation_range)
            elif target_min is not None:
                # Only lower bound - move toward middle of range above it
                mutation_range = max(
                    MUTATION_RANGE_MIN, (current_value - target_min) * MUTATION_RANGE_SCALE_GENERAL
                )
                return current_value - np.random.uniform(0, mutation_range)
            else:
                return current_value - np.random.uniform(0, MUTATION_RANGE_DEFAULT)

        else:  # 'both' - no clear escape direction
            # Use standard bounded mutation with bias based on escape_pressure
            if target_min is not None and target_max is not None:
                range_size = target_max - target_min
                mutation_range = range_size * MUTATION_RANGE_SCALE_ACTIONABLE

                # Calculate center of target range
                target_center = (target_min + target_max) / 2

                # Bias mutation toward target center based on escape_pressure
                bias = (target_center - current_value) * escape_pressure * MUTATION_RANGE_SCALE_ACTIONABLE
                mutation = np.random.uniform(-mutation_range, mutation_range) + bias
                return current_value + mutation

            elif target_min is not None:
                # Only min bound - prefer staying above it
                mutation_range = max(
                    MUTATION_RANGE_DEFAULT, (current_value - target_min) * MUTATION_RANGE_SCALE_ACTIONABLE
                )
                return current_value + np.random.uniform(
                    -mutation_range * 0.3, mutation_range
                )

            elif target_max is not None:
                # Only max bound - prefer staying below it
                mutation_range = max(
                    MUTATION_RANGE_DEFAULT, (target_max - current_value) * MUTATION_RANGE_SCALE_ACTIONABLE
                )
                return current_value + np.random.uniform(
                    -mutation_range, mutation_range * 0.3
                )

            else:
                # No constraints - use default mutation
                return current_value + np.random.uniform(
                    -MUTATION_RANGE_DEFAULT, MUTATION_RANGE_DEFAULT
                )

    def crossover_dict(self, ind1, ind2, indpb, sample=None):
        """Custom crossover operator for dict-based individuals.

        Args:
            ind1, ind2: Parent individuals (dicts)
            indpb: Probability of swapping each feature
            sample: Original sample dict for actionability enforcement (optional)

        Returns:
            Tuple of two offspring individuals
        """
        for key in ind1.keys():
            if np.random.rand() < indpb:
                ind1[key], ind2[key] = ind2[key], ind1[key]

        # Enforce actionability constraints after crossover
        if sample is not None and self.dict_non_actionable:
            for ind in [ind1, ind2]:
                for feature, constraint in self.dict_non_actionable.items():
                    if feature in ind and feature in sample:
                        original_value = sample[feature]
                        if constraint == "non_decreasing":
                            ind[feature] = max(ind[feature], original_value)
                        elif constraint == "non_increasing":
                            ind[feature] = min(ind[feature], original_value)
                        elif constraint == "no_change":
                            ind[feature] = original_value

        return ind1, ind2
