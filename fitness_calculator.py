"""
FitnessCalculator: Calculates fitness scores for counterfactual candidates.

Extracted from CounterFactualModel.py to provide focused fitness calculation
functionality for counterfactual generation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine

from constants import (
    INVALID_FITNESS,
    CLASS_PENALTY_SOFT_BASE,
    CLASS_PENALTY_HARD_BOOST,
    BOUNDARY_PENALTY_THRESHOLD,
    BOUNDARY_PENALTY_VALUE,
    CONSTRAINT_VIOLATION_MULTIPLIER,
    DEFAULT_BOUNDARY_DISTANCE,
    FITNESS_SHARING_BASE_SIGMA,
    UNCONSTRAINED_CHANGE_PENALTY_FACTOR,
)


class FitnessCalculator:
    """
    Calculates fitness scores and related metrics for counterfactual candidates.
    """

    def __init__(
        self,
        model,
        feature_names=None,
        diversity_weight=0.1,
        repulsion_weight=0.1,
        boundary_weight=15.0,
        distance_factor=5.0,
        sparsity_factor=1.0,
        constraints_factor=3.0,
        original_escape_weight=2.0,
        max_bonus_cap=10.0,
        unconstrained_penalty_factor=UNCONSTRAINED_CHANGE_PENALTY_FACTOR,
        constraint_validator=None,
        boundary_analyzer=None,
        verbose=False,
    ):
        """
        Initialize the FitnessCalculator.

        Args:
            model: The machine learning model used for predictions.
            feature_names: List of feature names from the model.
            diversity_weight (float): Weight for diversity bonus in fitness calculation.
            repulsion_weight (float): Weight for repulsion bonus in fitness calculation.
            boundary_weight (float): Weight for boundary proximity in fitness calculation.
            distance_factor (float): Weight for distance component in fitness calculation.
            sparsity_factor (float): Weight for sparsity component in fitness calculation.
            constraints_factor (float): Weight for constraint violation component.
            original_escape_weight (float): Weight for penalizing staying within original class bounds.
            max_bonus_cap (float): Maximum cap for diversity/repulsion bonuses.
            unconstrained_penalty_factor (float): Penalty multiplier for changing features without target constraints.
            constraint_validator: ConstraintValidator instance for validation.
            boundary_analyzer: BoundaryAnalyzer instance for escape penalty calculation.
            verbose (bool): Whether to print detailed logging.
        """
        self.model = model
        self.feature_names = feature_names
        self.diversity_weight = diversity_weight
        self.repulsion_weight = repulsion_weight
        self.boundary_weight = boundary_weight
        self.distance_factor = distance_factor
        self.sparsity_factor = sparsity_factor
        self.constraints_factor = constraints_factor
        self.original_escape_weight = original_escape_weight
        self.max_bonus_cap = max_bonus_cap
        self.unconstrained_penalty_factor = unconstrained_penalty_factor
        self.constraint_validator = constraint_validator
        self.boundary_analyzer = boundary_analyzer
        self.verbose = verbose

    def calculate_distance(
        self, original_sample, counterfactual_sample, metric="euclidean"
    ):
        """
        Calculates the distance between the original sample and the counterfactual sample.

        Parameters:
        - original_sample: Array-like, shape (n_features,), the original input sample.
        - counterfactual_sample: Array-like, shape (n_features,), the counterfactual sample.
        - metric: String, the distance metric to use. Options are "euclidean", "manhattan", or "cosine".

        Returns:
        - Distance between the original sample and the counterfactual sample.
        """
        # Ensure inputs are numpy arrays
        original_sample = np.array(original_sample, dtype=float)
        counterfactual_sample = np.array(counterfactual_sample, dtype=float)

        # Check for NaN/Inf values and replace with 0 to avoid errors
        if not np.all(np.isfinite(original_sample)):
            original_sample = np.nan_to_num(
                original_sample, nan=0.0, posinf=0.0, neginf=0.0
            )
        if not np.all(np.isfinite(counterfactual_sample)):
            counterfactual_sample = np.nan_to_num(
                counterfactual_sample, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Validate metric and compute distance
        if metric == "euclidean":
            distance = euclidean(original_sample, counterfactual_sample)
        elif metric == "manhattan":
            distance = cityblock(original_sample, counterfactual_sample)
        elif metric == "cosine":
            # Avoid division by zero in cosine similarity
            if np.all(original_sample == 0) or np.all(counterfactual_sample == 0):
                distance = 1  # Max cosine distance if one vector is zero
            else:
                distance = cosine(original_sample, counterfactual_sample)
        else:
            raise ValueError(
                "Invalid metric. Choose from 'euclidean', 'manhattan', or 'cosine'."
            )

        return distance

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        """
        Calculate sparsity as the ratio of changed features.

        Args:
            original_sample: dict of feature values
            counterfactual_sample: dict of feature values

        Returns:
            float: Ratio of changed features (0=no changes, 1=all changed)
        """
        # Convert dicts to arrays in consistent order
        feature_names = list(original_sample.keys())
        original_array = np.array([original_sample[f] for f in feature_names])
        counterfactual_array = np.array(
            [counterfactual_sample[f] for f in feature_names]
        )

        # Count how many features differ
        changed_features = np.sum(original_array != counterfactual_array)

        # Return ratio of changed features
        return changed_features / len(feature_names)

    def _get_target_constrained_features(self, target_class, constraints):
        """
        Get set of normalized feature names that have constraints in the target class.

        Args:
            target_class (int): The target class.
            constraints (dict): Constraints dictionary.

        Returns:
            set: Set of normalized feature names with target constraints.
        """
        if not constraints or target_class is None:
            return set()

        constrained_features = set()
        target_constraints = constraints.get(f"Class {target_class}", [])

        for constraint in target_constraints:
            feature_name = constraint.get("feature", "")
            if feature_name and self.boundary_analyzer:
                norm_name = self.boundary_analyzer._normalize_feature_name(feature_name)
                constrained_features.add(norm_name)

        return constrained_features

    def calculate_unconstrained_change_penalty(
        self, individual, sample, target_class, constraints
    ):
        """
        Calculate penalty for changing features that don't have constraints in the target class.
        Features without target constraints should be changed as a last resort.

        Args:
            individual (dict): The counterfactual candidate.
            sample (dict): The original sample.
            target_class (int): The target class.
            constraints (dict): Constraints dictionary.

        Returns:
            float: Penalty score based on unconstrained feature changes.
        """
        if not constraints or target_class is None:
            return 0.0

        # Get features that have constraints in target class
        constrained_features = self._get_target_constrained_features(
            target_class, constraints
        )

        if not constrained_features:
            # No constraints defined - no penalty
            return 0.0

        # Calculate weighted penalty for feature changes
        penalty = 0.0
        total_features = 0

        for feature, cf_value in individual.items():
            orig_value = sample.get(feature, cf_value)
            total_features += 1

            # Check if feature changed
            if cf_value != orig_value:
                # Normalize feature name for comparison
                norm_feature = (
                    self.boundary_analyzer._normalize_feature_name(feature)
                    if self.boundary_analyzer
                    else feature.lower()
                )

                if norm_feature not in constrained_features:
                    # Feature has no target constraint - apply higher penalty
                    # Multiply by factor here to avoid cancellation in fitness calc
                    change_magnitude = abs(cf_value - orig_value)
                    penalty += change_magnitude * self.unconstrained_penalty_factor
                else:
                    # Feature has target constraint - standard penalty (no extra weight)
                    change_magnitude = abs(cf_value - orig_value)
                    penalty += change_magnitude

        # Normalize by number of features to keep penalty scale consistent
        return penalty / total_features if total_features > 0 else 0.0

    def individual_diversity(self, individual, population):
        """
        Calculate the average distance from this individual to all others in the population.

        Args:
            individual (dict): The individual to calculate diversity for.
            population (list): List of all individuals in the population.

        Returns:
            float: Average distance to other individuals.
        """
        if len(population) <= 1:
            return 0.0

        ind_array = np.array([individual[key] for key in sorted(individual.keys())])
        distances = []

        for other in population:
            other_array = np.array([other[key] for key in sorted(other.keys())])
            if not np.array_equal(ind_array, other_array):
                distances.append(np.linalg.norm(ind_array - other_array))

        return np.mean(distances) if distances else 0.0

    def min_distance_to_others(self, individual, population):
        """
        Calculate the minimum distance from this individual to any other in the population.

        Args:
            individual (dict): The individual to calculate distance for.
            population (list): List of all individuals in the population.

        Returns:
            float: Minimum distance to nearest neighbor.
        """
        if len(population) <= 1:
            return 0.0

        ind_array = np.array([individual[key] for key in sorted(individual.keys())])
        distances = []

        for other in population:
            other_array = np.array([other[key] for key in sorted(other.keys())])
            if not np.array_equal(ind_array, other_array):
                distances.append(np.linalg.norm(ind_array - other_array))

        return min(distances) if distances else 0.0

    def distance_to_boundary_line(self, individual, target_class):
        """
        Calculate distance to decision boundary based on class probabilities.

        Args:
            individual (dict): The individual to calculate boundary distance for.
            target_class (int): The target class.

        Returns:
            float: Distance to decision boundary.
        """
        features = np.array(
            [individual[key] for key in sorted(individual.keys())]
        ).reshape(1, -1)

        try:
            # Convert to DataFrame with feature names if available for model compatibility
            if self.feature_names is not None:
                features_df = pd.DataFrame(features, columns=self.feature_names)
                probs = self.model.predict_proba(features_df)[0]
            else:
                probs = self.model.predict_proba(features)[0]
            target_prob = probs[target_class]
            other_probs = [p for i, p in enumerate(probs) if i != target_class]
            max_other_prob = max(other_probs) if other_probs else 0.0

            # Distance to boundary is the difference between target and highest other class
            boundary_distance = abs(target_prob - max_other_prob)
            return boundary_distance
        except:
            # Fallback if model doesn't support predict_proba
            return DEFAULT_BOUNDARY_DISTANCE

    def calculate_fitness(
        self,
        individual,
        original_features,
        sample,
        target_class,
        metric="euclidean",
        population=None,
        original_class=None,
        return_components=False,
    ):
        """
        Calculate the fitness score for an individual sample using weighted components.
        Based on the total_fitness logic from dpg_aug.ipynb.

        Enhanced with dual-boundary support: penalizes staying within original class bounds
        while rewarding movement toward target class bounds.

        Uses soft class penalty based on prediction probabilities to provide gradient
        information even when the sample doesn't yet predict as target class.

        Args:
            individual (dict): The individual sample with feature values.
            original_features (np.array): The original feature values.
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.
            metric (str): The distance metric to use for calculating distance.
            population (list): The current population for diversity calculations.
            original_class (int): The original class of the sample for escape penalty.
            return_components (bool): If True, return (fitness, components_dict) instead of just fitness.

        Returns:
            float or tuple: The fitness score (lower is better), or if return_components=True,
                a tuple of (fitness, components_dict) where components_dict contains all individual
                fitness component values for debugging/analysis.
        """
        # Convert individual feature values to a numpy array
        features = np.array([individual[feature] for feature in sample.keys()]).reshape(
            1, -1
        )

        # Check if the change is actionable (requires constraint_validator)
        if self.constraint_validator and not self.constraint_validator.is_actionable_change(
            individual, sample
        ):
            if self.verbose:
                print(f"[VERBOSE-DPG] Fitness: INVALID (non-actionable change)")
            return INVALID_FITNESS

        # Check if sample is identical to original
        if np.array_equal(features.flatten(), original_features.flatten()):
            if self.verbose:
                print(f"[VERBOSE-DPG] Fitness: INVALID (identical to original)")
            return INVALID_FITNESS

        # Check the constraints (pass original_class for smart overlap handling)
        is_valid_constraint = True
        penalty_constraints = 0.0
        if self.constraint_validator:
            is_valid_constraint, penalty_constraints = self.constraint_validator.validate_constraints(
                individual, sample, target_class, original_class=original_class
            )

        # Calculate class prediction probability for soft penalty
        # This provides gradient information even when not yet in target class
        try:
            if self.feature_names is not None:
                features_df = pd.DataFrame(features, columns=self.feature_names)
                probs = self.model.predict_proba(features_df)[0]
            else:
                probs = self.model.predict_proba(features)[0]

            target_prob = probs[target_class]
            predicted_class = np.argmax(probs)

            # Soft class penalty: penalize low probability for target class
            # Range: 0 (target_prob=1) to large value (target_prob=0)
            # Use exponential to strongly penalize low probabilities
            class_penalty = CLASS_PENALTY_SOFT_BASE * (1.0 - target_prob) ** 2

            # Additional hard penalty if not predicting target class
            if predicted_class != target_class:
                class_penalty += CLASS_PENALTY_HARD_BOOST

        except Exception:
            # Fallback: use hard prediction
            if self.constraint_validator:
                is_valid_class = self.constraint_validator.check_validity(
                    features.flatten(), original_features.flatten(), target_class
                )
                if not is_valid_class:
                    return INVALID_FITNESS
            class_penalty = 0.0

        # Calculate core components
        distance_score = self.calculate_distance(
            original_features, features.flatten(), metric
        )
        sparsity_score = self.calculate_sparsity(sample, individual)

        # Calculate penalty for changing unconstrained features
        unconstrained_penalty = 0.0
        if (
            self.unconstrained_penalty_factor > 0
            and self.constraint_validator
            and target_class is not None
        ):
            unconstrained_penalty = self.calculate_unconstrained_change_penalty(
                individual,
                sample,
                target_class,
                self.constraint_validator.constraints,
            )

        # Base fitness (minimize distance and sparsity, penalize constraint violations and wrong class)
        # Distance is the PRIMARY component - scale others relative to a reference distance
        # This ensures distance_factor actually controls the importance of distance
        reference_distance = max(distance_score, 0.01)  # Avoid division by zero
        
        # Normalize penalties relative to distance scale so distance_factor is meaningful
        # When distance_factor=500, other components should be secondary
        normalized_sparsity = self.sparsity_factor * sparsity_score * reference_distance
        normalized_constraint_penalty = self.constraints_factor * penalty_constraints * reference_distance
        normalized_class_penalty = class_penalty * reference_distance / 10.0  # Scale down class penalty
        
        base_fitness = (
            self.distance_factor * distance_score  # Primary component
            + normalized_sparsity
            + normalized_constraint_penalty
            + unconstrained_penalty * reference_distance
            + normalized_class_penalty
        )

        # DUAL-BOUNDARY: Add original class escape penalty
        # This penalizes individuals that haven't escaped the original class boundaries
        # Only for non-overlapping features where escaping is meaningful
        escape_penalty = 0.0
        if (
            original_class is not None
            and self.original_escape_weight > 0
            and self.boundary_analyzer
        ):
            escape_penalty = self.boundary_analyzer.calculate_original_escape_penalty(
                individual, sample, original_class, target_class=target_class
            )
            base_fitness += self.original_escape_weight * escape_penalty

        # Initialize bonus/penalty variables for component tracking
        div_bonus = 0.0
        rep_bonus = 0.0
        line_bonus = 0.0
        boundary_penalty = 0.0
        niche_count = 1.0

        # If population is provided, add diversity and repulsion bonuses
        if population is not None and len(population) > 1:
            # Diversity bonus: reward being different from others
            div = self.individual_diversity(individual, population)
            div_bonus = self.diversity_weight * div

            # Repulsion bonus: reward having minimum distance to nearest neighbor
            min_d = self.min_distance_to_others(individual, population)
            rep_bonus = self.repulsion_weight * min_d

            # Boundary bonus: reward proximity to decision boundary
            dist_line = self.distance_to_boundary_line(individual, target_class)
            line_bonus = 1.0 / (1.0 + dist_line) * self.boundary_weight

            # Cap bonuses to prevent unbounded negative fitness
            # This is critical for high-dimensional datasets (e.g., German Credit with 20+ features)
            total_bonus = div_bonus + rep_bonus + line_bonus
            if total_bonus > self.max_bonus_cap:
                scale_factor = self.max_bonus_cap / total_bonus
                div_bonus *= scale_factor
                rep_bonus *= scale_factor
                line_bonus *= scale_factor

            # Penalty for being too far from boundary (only if not yet predicting target)
            boundary_penalty = (
                BOUNDARY_PENALTY_VALUE
                if dist_line > BOUNDARY_PENALTY_THRESHOLD and class_penalty > 0
                else 0.0
            )

            # Total fitness (lower is better, so we subtract bonuses)
            fitness = (
                base_fitness - div_bonus - rep_bonus - line_bonus + boundary_penalty
            )

            # FITNESS SHARING: Penalize individuals in crowded regions to maintain diversity
            # This prevents population collapse to identical clones
            # Dynamic sigma_share: scale with sqrt(n_features) to account for dimensionality
            n_features = len(individual)
            sigma_share = max(
                FITNESS_SHARING_BASE_SIGMA, np.sqrt(n_features) * 1.5
            )  # Scale sharing radius with dimensionality (increased multiplier)
            niche_count = 1.0  # Start at 1 (counting self)

            ind_array = np.array([individual[key] for key in sorted(individual.keys())])
            for other in population:
                if other is not individual:
                    other_array = np.array([other[key] for key in sorted(other.keys())])
                    dist = np.linalg.norm(ind_array - other_array)

                    # Triangular sharing function: nearby individuals increase niche count
                    # Use softer contribution (0.5 multiplier) to reduce aggressiveness
                    if dist < sigma_share:
                        niche_count += 0.5 * (1.0 - (dist / sigma_share))

            # Apply fitness sharing: use square root for softer penalty
            # This allows some clustering near the decision boundary
            # sqrt(niche_count) is gentler than niche_count directly
            fitness *= np.sqrt(niche_count)
        else:
            # Without population, just use base fitness
            fitness = base_fitness

        # Additional penalty for constraint violations
        constraint_violation_multiplier = 1.0
        if not is_valid_constraint:
            fitness *= CONSTRAINT_VIOLATION_MULTIPLIER
            constraint_violation_multiplier = CONSTRAINT_VIOLATION_MULTIPLIER

        # If component breakdown requested, build and return it
        if return_components:
            components = {
                'distance_score': float(distance_score),
                'sparsity_score': float(sparsity_score),
                'penalty_constraints': float(penalty_constraints),
                'unconstrained_penalty': float(unconstrained_penalty),
                'class_penalty': float(class_penalty),
                'escape_penalty': float(escape_penalty),
                'div_bonus': float(div_bonus),
                'rep_bonus': float(rep_bonus),
                'line_bonus': float(line_bonus),
                'boundary_penalty': float(boundary_penalty),
                'niche_count': float(niche_count),
                'constraint_violation_multiplier': float(constraint_violation_multiplier),
                'base_fitness': float(base_fitness),
                'total_fitness': float(fitness),
            }
            return fitness, components

        return fitness
