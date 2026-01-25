import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean, cityblock, cosine

from deap import base, creator, tools

class CounterFactualModel:
    def __init__(self, model, constraints, dict_non_actionable=None, verbose=False, 
                 diversity_weight=0.5, repulsion_weight=4.0, boundary_weight=15.0, 
                 distance_factor=2.0, sparsity_factor=1.0, constraints_factor=3.0,
                 original_escape_weight=2.0, escape_pressure=0.5, prioritize_non_overlapping=True,
                 max_bonus_cap=50.0, X_train=None, y_train=None, min_probability_margin=0.001):
        """
        Initialize the CounterFactualDPG object.

        Args:
            model: The machine learning model used for predictions.
            constraints (dict): Nested dictionary containing constraints for features.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
              non_decreasing: feature cannot decrease
              non_increasing: feature cannot increase
              no_change: feature cannot change
            diversity_weight (float): Weight for diversity bonus in fitness calculation.
            repulsion_weight (float): Weight for repulsion bonus in fitness calculation.
            boundary_weight (float): Weight for boundary proximity in fitness calculation.
            distance_factor (float): Weight for distance component in fitness calculation.
            sparsity_factor (float): Weight for sparsity component in fitness calculation.
            constraints_factor (float): Weight for constraint violation component in fitness calculation.
            original_escape_weight (float): Weight for penalizing staying within original class bounds.
            escape_pressure (float): Balance between escaping original (1.0) vs approaching target (0.0).
            prioritize_non_overlapping (bool): Prioritize mutating features with non-overlapping boundaries.
            max_bonus_cap (float): Maximum cap for diversity/repulsion bonuses to prevent unbounded negative fitness.
            X_train (DataFrame): Training data features for nearest neighbor fallback.
            y_train (Series): Training data labels for nearest neighbor fallback.
            min_probability_margin (float): Minimum margin the target class probability must exceed the 
                second-highest class probability by. Prevents accepting weak counterfactuals where
                the prediction is essentially a tie. Default 0.001 (0.1% margin).
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable #non_decreasing, non_increasing, no_change
        self.average_fitness_list = []
        self.best_fitness_list = []
        self.evolution_history = []  # Store best individual per generation for visualization
        self.verbose = verbose
        self.diversity_weight = diversity_weight
        self.repulsion_weight = repulsion_weight
        self.boundary_weight = boundary_weight
        self.distance_factor = distance_factor
        self.sparsity_factor = sparsity_factor
        self.constraints_factor = constraints_factor
        # Dual-boundary parameters
        self.original_escape_weight = original_escape_weight
        self.escape_pressure = escape_pressure
        self.prioritize_non_overlapping = prioritize_non_overlapping
        # Fitness calculation parameters
        self.max_bonus_cap = max_bonus_cap
        # Store feature names from the model if available
        self.feature_names = getattr(model, 'feature_names_in_', None)
        # Cache for boundary analysis results
        self._boundary_analysis_cache = {}
        # Store training data for nearest neighbor fallback
        self.X_train = X_train
        self.y_train = y_train
        # Minimum probability margin for accepting counterfactuals
        self.min_probability_margin = min_probability_margin

    def _analyze_boundary_overlap(self, original_class, target_class):
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
            'non_overlapping': [],  # Features with clear escape path
            'overlapping': [],      # Features with overlapping bounds
            'escape_direction': {}, # Direction to escape: 'increase', 'decrease', or 'both'
            'feature_bounds': {}    # Store both bounds for each feature
        }
        
        # Build lookup dict for original constraints
        orig_bounds = {}
        for c in original_constraints:
            feature = c.get("feature", "")
            norm_feature = self._normalize_feature_name(feature)
            orig_bounds[norm_feature] = {
                'min': c.get('min'),
                'max': c.get('max'),
                'original_name': feature
            }
        
        # Analyze each target constraint
        for tc in target_constraints:
            feature = tc.get("feature", "")
            norm_feature = self._normalize_feature_name(feature)
            target_min = tc.get('min')
            target_max = tc.get('max')
            
            # Store bounds info
            analysis['feature_bounds'][norm_feature] = {
                'target_min': target_min,
                'target_max': target_max,
                'original_min': None,
                'original_max': None,
                'feature_name': feature
            }
            
            if norm_feature in orig_bounds:
                orig_min = orig_bounds[norm_feature].get('min')
                orig_max = orig_bounds[norm_feature].get('max')
                
                analysis['feature_bounds'][norm_feature]['original_min'] = orig_min
                analysis['feature_bounds'][norm_feature]['original_max'] = orig_max
                
                # Determine escape direction based on constraint comparison
                # Key insight: We need to move FROM original bounds TO target bounds
                non_overlapping = False
                escape_dir = 'both'
                
                # First check if constraints are identical (100% overlap, no discrimination)
                if (target_min == orig_min and target_max == orig_max):
                    # Identical constraints - maximally overlapping, not useful for discrimination
                    non_overlapping = False
                    escape_dir = 'both'
                else:
                    # Case 1: Target has upper bound, Original has lower bound
                    # Example: target_max=2.45, orig_min=2.50 -> must DECREASE to escape
                    # Use strict inequality to avoid false positives when bounds touch
                    if target_max is not None and orig_min is not None:
                        if target_max < orig_min:
                            non_overlapping = True
                            escape_dir = 'decrease'  # Must decrease to get below target_max
                        elif target_max < orig_min + (orig_max - orig_min if orig_max else 1):
                            escape_dir = 'decrease'  # Prefer decreasing
                    
                    # Case 2: Target has lower bound, Original has upper bound
                    # Example: target_min=5, orig_max=4 -> must INCREASE to escape
                    # Use strict inequality
                    if target_min is not None and orig_max is not None:
                        if target_min > orig_max:
                            non_overlapping = True
                            escape_dir = 'increase'  # Must increase to get above target_min
                        elif target_min > orig_min if orig_min else 0:
                            escape_dir = 'increase'  # Prefer increasing
                    
                    # Case 3: Both have same type of bound - compare values
                    if target_min is not None and orig_min is not None and target_max is None and orig_max is None:
                        if target_min > orig_min:
                            escape_dir = 'increase'  # Target requires higher minimum
                        elif target_min < orig_min:
                            escape_dir = 'decrease'  # Target allows lower values
                            
                    if target_max is not None and orig_max is not None and target_min is None and orig_min is None:
                        if target_max < orig_max:
                            escape_dir = 'decrease'  # Target requires lower maximum
                        elif target_max > orig_max:
                            escape_dir = 'increase'  # Target allows higher values
                
                if non_overlapping:
                    analysis['non_overlapping'].append(feature)
                else:
                    analysis['overlapping'].append(feature)
                
                analysis['escape_direction'][norm_feature] = escape_dir
            else:
                # No original constraint for this feature - it's in overlapping (no restriction)
                analysis['overlapping'].append(feature)
                analysis['escape_direction'][norm_feature] = 'both'
        
        # Warn if no non-overlapping features found (constraints are non-discriminative)
        if len(analysis['non_overlapping']) == 0 and len(target_constraints) > 0:
            if self.verbose:
                print(f"WARNING: No non-overlapping boundaries found between Class {original_class} and Class {target_class}.")
                print(f"  The DPG constraints are nearly identical for both classes.")
                print(f"  This may indicate the dataset lacks clear class-separating features in the constraint space.")
                print(f"  Counterfactual generation may be difficult or produce poor results.")
        
        self._boundary_analysis_cache[cache_key] = analysis
        return analysis

    def _calculate_original_escape_penalty(self, individual, sample, original_class, target_class=None):
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
            boundary_analysis = self._analyze_boundary_overlap(original_class, target_class)
            non_overlapping_features = set(
                self._normalize_feature_name(f) for f in boundary_analysis.get('non_overlapping', [])
            )
        
        penalty = 0.0
        features_checked = 0
        
        for feature, value in individual.items():
            norm_feature = self._normalize_feature_name(feature)
            
            # Only apply escape penalty for non-overlapping features
            # For overlapping features, being within original bounds is OK if also in target bounds
            if target_class is not None and norm_feature not in non_overlapping_features:
                continue
            
            original_value = sample.get(feature, value)
            
            # Find matching original constraint
            matching_constraint = next(
                (c for c in original_constraints if self._features_match(c.get("feature", ""), feature)),
                None
            )
            
            if matching_constraint:
                orig_min = matching_constraint.get('min')
                orig_max = matching_constraint.get('max')
                
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
                            dist_from_boundary = 1.0 - abs(value - center) / (range_size / 2)
                            penalty += max(0, dist_from_boundary)
                    elif orig_min is not None:
                        # Single min bound - penalize being above it (deeper inside = worse)
                        # Use distance from the boundary relative to original value
                        if original_value > orig_min:
                            range_estimate = original_value - orig_min
                            dist_inside = (value - orig_min) / range_estimate if range_estimate > 0 else 0.5
                            penalty += max(0, min(1, dist_inside))
                        else:
                            penalty += 0.5
                    elif orig_max is not None:
                        # Single max bound - penalize being below it (deeper inside = worse)
                        if original_value < orig_max:
                            range_estimate = orig_max - original_value
                            dist_inside = (orig_max - value) / range_estimate if range_estimate > 0 else 0.5
                            penalty += max(0, min(1, dist_inside))
                        else:
                            penalty += 0.5
        
        return penalty

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
              return False
          if constraint == "non_increasing" and new_value > original_value:
              return False
          if constraint == "no_change" and new_value != original_value:
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
        - 0 if the predicted class matches the desired class and the sample is different from the original.
        - np.inf if the predicted class does not match the desired class or the sample is identical to the original.
        """
        # Ensure the input samples are numpy arrays
        counterfactual_sample = np.array(counterfactual_sample).reshape(1, -1)
        original_sample = np.array(original_sample).reshape(1, -1)

        # Check if the counterfactual sample is different from the original sample
        if np.array_equal(counterfactual_sample, original_sample):
            return False  # Return np.inf if the samples are identical

        # Predict the class for the counterfactual sample
        # Convert to DataFrame with feature names if available for model compatibility
        if self.feature_names is not None:
            counterfactual_df = pd.DataFrame(counterfactual_sample, columns=self.feature_names)
            predicted_class = self.model.predict(counterfactual_df)[0]
        else:
            predicted_class = self.model.predict(counterfactual_sample)[0]

        # Check if the predicted class matches the desired class
        if predicted_class == desired_class:
            return True
        else:
            return False

    def plot_fitness(self):
        """
        Delegate plotting to CounterFactualVisualizer.plot_fitness to keep visualizations centralized.
        """
        try:
            from CounterFactualVisualizer import plot_fitness as _plot_fitness
            return _plot_fitness(self)
        except Exception:
            # Fallback: minimal inline plot in case visualizer import fails
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.best_fitness_list, label='Best Fitness', color='blue')
            ax.plot(self.average_fitness_list, label='Average Fitness', color='green')
            ax.set_title('Fitness Over Generations')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend()
            plt.tight_layout()
            plt.close(fig)
            return fig

    def calculate_distance(self,original_sample, counterfactual_sample, metric="euclidean"):
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
            original_sample = np.nan_to_num(original_sample, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(counterfactual_sample)):
            counterfactual_sample = np.nan_to_num(counterfactual_sample, nan=0.0, posinf=0.0, neginf=0.0)

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
            raise ValueError("Invalid metric. Choose from 'euclidean', 'manhattan', or 'cosine'.")

        return distance

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
        import re
        # Remove anything in parentheses (like units)
        feature = re.sub(r'\s*\([^)]*\)', '', feature)
        # Replace underscores with spaces
        feature = feature.replace('_', ' ')
        # Normalize multiple spaces to single space
        feature = re.sub(r'\s+', ' ', feature)
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
        return self._normalize_feature_name(feature1) == self._normalize_feature_name(feature2)

    def validate_constraints(self, S_prime, sample, target_class, original_class=None, strict_mode=True):
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
        class_constraints = self.constraints.get(str("Class "+str(target_class)), [])

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Validate numerical constraints specific to the target class
                matching_constraint = next(
                    (condition for condition in class_constraints if self._features_match(condition["feature"], feature)),
                    None
                )
                
                if matching_constraint:
                    min_val = matching_constraint.get("min")
                    max_val = matching_constraint.get("max")
                    
                    # Check if the new value violates min constraint
                    if min_val is not None and new_value < min_val:
                        valid_change = False
                        penalty += abs(new_value - min_val)
                    
                    # Check if the new value violates max constraint
                    if max_val is not None and new_value > max_val:
                        valid_change = False
                        penalty += abs(new_value - max_val)

        # In relaxed mode, skip non-target class penalty (used when constraints overlap significantly)
        if not strict_mode:
            return valid_change, penalty

        # Get boundary overlap analysis to identify non-overlapping features
        # Only penalize non-target class violations for NON-OVERLAPPING features
        non_overlapping_features = set()
        if original_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(original_class, target_class)
            non_overlapping_features = set(
                self._normalize_feature_name(f) for f in boundary_analysis.get('non_overlapping', [])
            )

        # Collect all constraints that are NOT related to the target class
        non_target_class_constraints = [
            condition
            for class_name, conditions in self.constraints.items()
            if class_name != "Class " + str(target_class)  # Exclude the target class constraints
            for condition in conditions
        ]

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)
            norm_feature = self._normalize_feature_name(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Only apply non-target penalty for NON-OVERLAPPING features
                # For overlapping features, being within both class constraints is acceptable
                if original_class is not None and norm_feature not in non_overlapping_features:
                    # This feature has overlapping constraints - skip non-target penalty
                    continue
                
                # Validate numerical constraints NOT related to the target class
                matching_constraint = next(
                    (condition for condition in non_target_class_constraints if self._features_match(condition["feature"], feature)),
                    None
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


        #print('Total Penalty:', penalty)
        return valid_change, penalty

    def get_valid_sample(self, sample, target_class, original_class=None):
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
        adjusted_sample = sample.copy()  # Start with the original values
        # Filter the constraints for the specified target class
        class_constraints = self.constraints.get(f"Class {target_class}", [])
        original_constraints = self.constraints.get(f"Class {original_class}", []) if original_class is not None else []
        
        # Get boundary analysis for escape direction if original class provided
        boundary_analysis = None
        if original_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(original_class, target_class)

        for feature, original_value in sample.items():
            min_value = -np.inf
            max_value = np.inf
            escape_dir = 'both'
            orig_min, orig_max = None, None

            # Find the constraints for this feature using direct lookup
            matching_constraint = next(
                (condition for condition in class_constraints if self._features_match(condition["feature"], feature)),
                None
            )
            
            if matching_constraint:
                min_value = matching_constraint.get("min") if matching_constraint.get("min") is not None else -np.inf
                max_value = matching_constraint.get("max") if matching_constraint.get("max") is not None else np.inf
            
            # Get original class bounds for escape direction
            if original_constraints:
                matching_orig = next(
                    (c for c in original_constraints if self._features_match(c.get("feature", ""), feature)),
                    None
                )
                if matching_orig:
                    orig_min = matching_orig.get('min')
                    orig_max = matching_orig.get('max')
            
            # Get escape direction from boundary analysis
            if boundary_analysis:
                norm_feature = self._normalize_feature_name(feature)
                escape_dir = boundary_analysis.get('escape_direction', {}).get(norm_feature, 'both')

            # Incorporate non-actionable constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                
                if actionability == "non_decreasing":
                    min_value = max(min_value, original_value)
                    if min_value > max_value:
                        max_value = min_value + min_value * 0.1  # Adjust to ensure valid range
                elif actionability == "non_increasing":
                    max_value = min(max_value, original_value)
                    if max_value < min_value:
                        min_value = max_value + max_value * 0.1  # Adjust to ensure valid range
                elif actionability == "no_change":
                    adjusted_sample[feature] = original_value
                    continue

            # If no explicit min/max constraints, use range around original value
            if min_value == -np.inf:
                min_value = original_value - 0.5 * (abs(original_value) + 1.0)
            if max_value == np.inf:
                max_value = original_value + 0.5 * (abs(original_value) + 1.0)

            # Determine target value based on escape direction and dual-boundary awareness
            if escape_dir == 'increase' and max_value != np.inf:
                # Bias toward upper bound to escape original class
                target_value = min_value + (max_value - min_value) * (0.5 + 0.3 * self.escape_pressure)
            elif escape_dir == 'decrease' and min_value != -np.inf:
                # Bias toward lower bound to escape original class
                target_value = min_value + (max_value - min_value) * (0.5 - 0.3 * self.escape_pressure)
            else:
                # Default: keep original value if within bounds, otherwise use midpoint
                if min_value <= original_value <= max_value:
                    target_value = original_value
                else:
                    target_value = (min_value + max_value) / 2

            # Clip to bounds and set
            adjusted_sample[feature] = np.clip(target_value, min_value, max_value)
            
        return adjusted_sample

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
        counterfactual_array = np.array([counterfactual_sample[f] for f in feature_names])
        
        # Count how many features differ
        changed_features = np.sum(original_array != counterfactual_array)
        
        # Return ratio of changed features
        return changed_features / len(feature_names)

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
        features = np.array([individual[key] for key in sorted(individual.keys())]).reshape(1, -1)
        
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
            return 0.05

    def calculate_fitness(self, individual, original_features, sample, target_class, metric="cosine", population=None, original_class=None):
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

            Returns:
                float: The fitness score for the individual (lower is better).
            """
            INVALID_FITNESS = 1e6  # Large penalty for invalid samples
            
            # Convert individual feature values to a numpy array
            features = np.array([individual[feature] for feature in sample.keys()]).reshape(1, -1)

            # Check if the change is actionable
            if not self.is_actionable_change(individual, sample):
                return INVALID_FITNESS

            # Check if sample is identical to original
            if np.array_equal(features.flatten(), original_features.flatten()):
                return INVALID_FITNESS

            # Check the constraints (pass original_class for smart overlap handling)
            is_valid_constraint, penalty_constraints = self.validate_constraints(
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
                class_penalty = 100.0 * (1.0 - target_prob) ** 2
                
                # Additional hard penalty if not predicting target class
                if predicted_class != target_class:
                    class_penalty += 50.0  # Smaller than before, combined with soft penalty
                    
            except Exception:
                # Fallback: use hard prediction
                is_valid_class = self.check_validity(features.flatten(), original_features.flatten(), target_class)
                if not is_valid_class:
                    return INVALID_FITNESS
                class_penalty = 0.0
            
            # Calculate core components
            distance_score = self.calculate_distance(original_features, features.flatten(), metric)
            sparsity_score = self.calculate_sparsity(sample, individual)
            
            # Base fitness (minimize distance and sparsity, penalize constraint violations and wrong class)
            base_fitness = (self.distance_factor * distance_score + 
                          self.sparsity_factor * sparsity_score + 
                          self.constraints_factor * penalty_constraints +
                          class_penalty)
            
            # DUAL-BOUNDARY: Add original class escape penalty
            # This penalizes individuals that haven't escaped the original class boundaries
            # Only for non-overlapping features where escaping is meaningful
            if original_class is not None and self.original_escape_weight > 0:
                escape_penalty = self._calculate_original_escape_penalty(
                    individual, sample, original_class, target_class=target_class
                )
                base_fitness += self.original_escape_weight * escape_penalty
            
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
                boundary_penalty = 30.0 if dist_line > 0.1 and class_penalty > 0 else 0.0
                
                # Total fitness (lower is better, so we subtract bonuses)
                fitness = base_fitness - div_bonus - rep_bonus - line_bonus + boundary_penalty
                
                # FITNESS SHARING: Penalize individuals in crowded regions to maintain diversity
                # This prevents population collapse to identical clones
                # Dynamic sigma_share: scale with sqrt(n_features) to account for dimensionality
                n_features = len(individual)
                sigma_share = max(1.0, np.sqrt(n_features))  # Scale sharing radius with dimensionality
                niche_count = 1.0  # Start at 1 (counting self)
                
                ind_array = np.array([individual[key] for key in sorted(individual.keys())])
                for other in population:
                    if other is not individual:
                        other_array = np.array([other[key] for key in sorted(other.keys())])
                        dist = np.linalg.norm(ind_array - other_array)
                        
                        # Triangular sharing function: nearby individuals increase niche count
                        if dist < sigma_share:
                            niche_count += 1.0 - (dist / sigma_share)
                
                # Apply fitness sharing: multiply fitness by niche count
                # This makes crowded regions less attractive (higher fitness = worse for minimization)
                fitness *= niche_count
            else:
                # Without population, just use base fitness
                fitness = base_fitness
            
            # Additional penalty for constraint violations
            if not is_valid_constraint:
                fitness *= 2.0  # Reduced from 5.0 - constraints are already penalized in base

            return fitness

    def _create_deap_individual(self, sample_dict, feature_names):
        """Create a DEAP individual from a dictionary."""
        individual = creator.Individual(sample_dict)
        return individual

    def _mutate_individual(self, individual, sample, feature_names, mutation_rate, target_class=None, original_class=None, boundary_analysis=None):
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
        if boundary_analysis is None and original_class is not None and target_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(original_class, target_class)
        
        # Determine feature mutation priority based on non-overlapping boundaries
        non_overlapping_features = set()
        escape_directions = {}
        if boundary_analysis and self.prioritize_non_overlapping:
            non_overlapping_features = set(
                self._normalize_feature_name(f) for f in boundary_analysis.get('non_overlapping', [])
            )
            escape_directions = boundary_analysis.get('escape_direction', {})
        
        for feature in feature_names:
            norm_feature = self._normalize_feature_name(feature)
            
            # Adjust mutation rate: higher for non-overlapping features
            effective_mutation_rate = mutation_rate
            if self.prioritize_non_overlapping and norm_feature in non_overlapping_features:
                # Boost mutation rate for features with clear escape paths
                effective_mutation_rate = min(1.0, mutation_rate * 1.5)
            
            if np.random.rand() < effective_mutation_rate:
                # Get target constraint boundaries for this feature
                target_min, target_max = None, None
                if target_constraints:
                    matching_constraint = next(
                        (c for c in target_constraints if self._features_match(c.get("feature", ""), feature)),
                        None
                    )
                    if matching_constraint:
                        target_min = matching_constraint.get('min')
                        target_max = matching_constraint.get('max')
                
                # Get original constraint boundaries for escape direction
                orig_min, orig_max = None, None
                if original_constraints:
                    matching_orig = next(
                        (c for c in original_constraints if self._features_match(c.get("feature", ""), feature)),
                        None
                    )
                    if matching_orig:
                        orig_min = matching_orig.get('min')
                        orig_max = matching_orig.get('max')
                
                # Determine escape direction based on analysis
                escape_dir = escape_directions.get(norm_feature, 'both')
                
                # Apply mutation based on actionability constraints
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionability = self.dict_non_actionable[feature]
                    original_value = sample[feature]
                    
                    if actionability == "non_decreasing":
                        # Only allow increase - use escape pressure to bias toward target upper bound
                        if target_max is not None:
                            mutation_range = min(0.5, (target_max - individual[feature]) * 0.1)
                        else:
                            mutation_range = 0.5
                        individual[feature] += np.random.uniform(0, mutation_range)
                        
                    elif actionability == "non_increasing":
                        # Only allow decrease - use escape pressure to bias toward target lower bound
                        if target_min is not None:
                            mutation_range = min(0.5, (individual[feature] - target_min) * 0.1)
                        else:
                            mutation_range = 0.5
                        individual[feature] += np.random.uniform(-mutation_range, 0)
                        
                    elif actionability == "no_change":
                        individual[feature] = original_value  # Do not change
                    else:
                        # Apply dual-boundary mutation
                        individual[feature] = self._dual_boundary_mutate(
                            individual[feature], target_min, target_max, orig_min, orig_max, escape_dir
                        )
                else:
                    # Feature not constrained by actionability - apply dual-boundary mutation
                    individual[feature] = self._dual_boundary_mutate(
                        individual[feature], target_min, target_max, orig_min, orig_max, escape_dir
                    )
                
                # Clip to target constraint boundaries if they exist
                if target_min is not None:
                    individual[feature] = max(target_min, individual[feature])
                if target_max is not None:
                    individual[feature] = min(target_max, individual[feature])
                
                # Ensure non-negative values and round
                individual[feature] = np.round(max(0, individual[feature]), 2)
                
        return individual,

    def _dual_boundary_mutate(self, current_value, target_min, target_max, orig_min, orig_max, escape_dir='both'):
        """
        Apply mutation that balances escaping original bounds and approaching target bounds.
        
        Uses escape_pressure parameter to control the balance:
        - escape_pressure=1.0: Fully focused on escaping original bounds
        - escape_pressure=0.0: Fully focused on approaching target bounds
        - escape_pressure=0.5 (default): Balanced approach
        
        Args:
            current_value: Current feature value
            target_min, target_max: Target class bounds
            orig_min, orig_max: Original class bounds
            escape_dir: Preferred escape direction ('increase', 'decrease', 'both')
            
        Returns:
            float: Mutated value
        """
        escape_pressure = self.escape_pressure
        
        # Calculate mutation based on escape direction and pressure
        if escape_dir == 'increase':
            # Must increase to escape original and reach target
            # Target point is the min of target range (the threshold we need to cross)
            if target_min is not None:
                target_point = target_min
                range_to_target = max(0.1, target_point - current_value)
                mutation_range = max(0.1, range_to_target * 0.15)
                # Bias toward increase
                return current_value + np.random.uniform(0, mutation_range)
            elif target_max is not None:
                # Only upper bound - move toward middle of range below it
                mutation_range = max(0.1, (target_max - current_value) * 0.15)
                return current_value + np.random.uniform(0, mutation_range)
            else:
                return current_value + np.random.uniform(0, 0.5)
                
        elif escape_dir == 'decrease':
            # Must decrease to escape original and reach target
            # Target point is the max of target range (the threshold we need to cross below)
            if target_max is not None:
                target_point = target_max
                range_to_target = max(0.1, current_value - target_point)
                mutation_range = max(0.1, range_to_target * 0.15)
                # Bias toward decrease
                return current_value - np.random.uniform(0, mutation_range)
            elif target_min is not None:
                # Only lower bound - move toward middle of range above it
                mutation_range = max(0.1, (current_value - target_min) * 0.15)
                return current_value - np.random.uniform(0, mutation_range)
            else:
                return current_value - np.random.uniform(0, 0.5)
                
        else:  # 'both' - no clear escape direction
            # Use standard bounded mutation with bias based on escape_pressure
            if target_min is not None and target_max is not None:
                range_size = target_max - target_min
                mutation_range = range_size * 0.1
                
                # Calculate center of target range
                target_center = (target_min + target_max) / 2
                
                # Bias mutation toward target center based on escape_pressure
                bias = (target_center - current_value) * escape_pressure * 0.1
                mutation = np.random.uniform(-mutation_range, mutation_range) + bias
                return current_value + mutation
                
            elif target_min is not None:
                # Only min bound - prefer staying above it
                mutation_range = max(0.5, (current_value - target_min) * 0.1)
                return current_value + np.random.uniform(-mutation_range * 0.3, mutation_range)
                
            elif target_max is not None:
                # Only max bound - prefer staying below it
                mutation_range = max(0.5, (target_max - current_value) * 0.1)
                return current_value + np.random.uniform(-mutation_range, mutation_range * 0.3)
                
            else:
                # No constraints - use default mutation
                return current_value + np.random.uniform(-0.5, 0.5)

    def _crossover_dict(self, ind1, ind2, indpb, sample=None):
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

    def genetic_algorithm(self, sample, target_class, population_size=100, generations=100, mutation_rate=0.8, metric="euclidean", delta_threshold=0.01, patience=10, n_jobs=-1, original_class=None, num_best_results=1):
        """Genetic algorithm implementation using DEAP framework
        
        Enhanced with dual-boundary support: uses both original and target class constraints
        to guide evolution. Mutations escape original class bounds while approaching target bounds.
        
        Args:
            sample (dict): Original sample features.
            target_class (int): Target class for counterfactual.
            population_size (int): Population size for GA.
            generations (int): Maximum generations to run.
            mutation_rate (float): Base mutation rate.
            metric (str): Distance metric for fitness calculation.
            delta_threshold (float): Convergence threshold.
            patience (int): Generations without improvement before early stopping.
            n_jobs (int): Number of parallel jobs for fitness evaluation.
                         -1 = use all CPUs (default), 1 = sequential.
            original_class (int): Original class for escape-aware mutation (dual-boundary).
            num_best_results (int): Number of top individuals to return from single GA run.
        """
        feature_names = list(sample.keys())
        original_features = np.array([sample[feature] for feature in feature_names])
        
        # Pre-compute boundary analysis for dual-boundary operations
        boundary_analysis = None
        if original_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(original_class, target_class)
            if self.verbose:
                non_overlapping = boundary_analysis.get('non_overlapping', [])
                print(f"[Dual-Boundary] Non-overlapping features: {non_overlapping}")
                print(f"[Dual-Boundary] Escape directions: {boundary_analysis.get('escape_direction', {})}")
        
        # Reset DEAP classes if they already exist
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create DEAP types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("Individual", dict, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        toolbox = base.Toolbox()
        
        # Enable parallel processing (default behavior with n_jobs=-1)
        if n_jobs != 1:
            from multiprocessing import Pool
            import os
            if n_jobs == -1:
                n_jobs = os.cpu_count()
            pool = Pool(processes=n_jobs)
            toolbox.register("map", pool.map)
        
        # Register individual creation (now with original_class for escape-aware initialization)
        toolbox.register("individual", self._create_deap_individual, 
                        sample_dict=self.get_valid_sample(sample, target_class, original_class),
                        feature_names=feature_names)
        
        # Register population creation
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register mating with custom dict crossover (pass original sample for constraint enforcement)
        toolbox.register("mate", self._crossover_dict, indpb=0.6, sample=sample)

        # Define diversity-aware selection function
        def select_diverse(individuals, k):
            """
            Select k individuals balancing fitness quality and diversity.
            This prevents selection from collapsing to clones of the best individual.
            """
            selected = []
            remaining = list(individuals)
            
            if not remaining:
                return selected
            
            # Always include the best individual first
            remaining_sorted = sorted(remaining, key=lambda x: x.fitness.values[0] if x.fitness.valid else 1e9)
            best = remaining_sorted[0]
            selected.append(best)
            remaining.remove(best)
            
            # Select remaining individuals balancing fitness and diversity
            while len(selected) < k and remaining:
                best_candidate = None
                best_score = float('inf')
                
                for candidate in remaining:
                    # Get fitness (lower is better)
                    fitness_val = candidate.fitness.values[0] if candidate.fitness.valid else 1e9
                    
                    # Calculate minimum distance to already selected individuals
                    cand_array = np.array([candidate[key] for key in sorted(candidate.keys())])
                    min_dist = float('inf')
                    for sel in selected:
                        sel_array = np.array([sel[key] for key in sorted(sel.keys())])
                        dist = np.linalg.norm(cand_array - sel_array)
                        min_dist = min(min_dist, dist)
                    
                    # Score combines fitness and diversity (lower is better)
                    # Give 30% weight to diversity bonus
                    diversity_bonus = -0.3 * min_dist  # Negative because we want to minimize
                    score = fitness_val + diversity_bonus
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    # If no candidate found, fill with random from remaining
                    if remaining:
                        selected.append(remaining.pop(0))
            
            return selected

        # Register diversity-aware selection instead of tournament selection
        toolbox.register("select", select_diverse)
        toolbox.register("mutate", self._mutate_individual, 
                        sample=sample, 
                        feature_names=feature_names,
                        mutation_rate=mutation_rate,
                        target_class=target_class,
                        original_class=original_class,
                        boundary_analysis=boundary_analysis)
        
        # Create initial population starting near the original sample
        # First individual is the original sample adjusted to constraint boundaries (escape-aware)
        base_individual = self.get_valid_sample(sample, target_class, original_class)
        population = [self._create_deap_individual(base_individual.copy(), feature_names)]
        
        # Remaining individuals are perturbations biased by escape direction
        target_constraints = self.constraints.get(f"Class {target_class}", [])
        original_constraints = self.constraints.get(f"Class {original_class}", []) if original_class else []
        escape_directions = boundary_analysis.get('escape_direction', {}) if boundary_analysis else {}
        
        for _ in range(population_size - 1):
            perturbed = sample.copy()
            # Add perturbations biased by escape direction for each feature
            for feature in feature_names:
                norm_feature = self._normalize_feature_name(feature)
                escape_dir = escape_directions.get(norm_feature, 'both')
                
                # Base perturbation
                if escape_dir == 'increase':
                    perturbation = np.random.uniform(0, 0.4)  # Bias toward increase
                elif escape_dir == 'decrease':
                    perturbation = np.random.uniform(-0.4, 0)  # Bias toward decrease
                else:
                    perturbation = np.random.uniform(-0.2, 0.2)  # Symmetric
                
                perturbed[feature] = sample[feature] + perturbation
                
                # Clip to target constraint boundaries if they exist
                matching_constraint = next(
                    (c for c in target_constraints if self._features_match(c.get("feature", ""), feature)),
                    None
                )
                if matching_constraint:
                    feature_min = matching_constraint.get('min')
                    feature_max = matching_constraint.get('max')
                    if feature_min is not None:
                        perturbed[feature] = max(feature_min, perturbed[feature])
                    if feature_max is not None:
                        perturbed[feature] = min(feature_max, perturbed[feature])
                
                # Ensure non-negative and round
                perturbed[feature] = np.round(max(0, perturbed[feature]), 2)
            
            population.append(self._create_deap_individual(perturbed, feature_names))
        
        # Register evaluate operator after population creation so it can capture population in closure
        # Now includes original_class for escape penalty calculation
        toolbox.register("evaluate", lambda ind: (self.calculate_fitness(
            ind, original_features, sample, target_class, metric, population, original_class),))
        
        # Setup statistics
        # Define INVALID_FITNESS threshold for filtering statistics
        INVALID_FITNESS = 1e6
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.nanmean([val[0] for val in x if not np.isinf(val[0]) and val[0] < INVALID_FITNESS]) if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x) else np.nan)
        stats.register("min", lambda x: np.nanmin([val[0] for val in x if not np.isinf(val[0]) and val[0] < INVALID_FITNESS]) if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x) else np.inf)
        
        # Setup hall of fame to keep best individuals
        hof = tools.HallOfFame(num_best_results)
        
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.evolution_history = []  # Reset evolution history for this run
        previous_best_fitness = float('inf')
        stable_generations = 0
        current_mutation_rate = mutation_rate
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Update statistics and hall of fame
            record = stats.compile(population)
            hof.update(population)
            
            # Store best individual for this generation (deep copy to preserve state)
            if hof[0].fitness.values[0] != np.inf:
                self.evolution_history.append(dict(hof[0]))
            
            best_fitness = record["min"]
            average_fitness = record["avg"]
            
            self.best_fitness_list.append(best_fitness)
            self.average_fitness_list.append(average_fitness)
            
            # Check for convergence
            fitness_improvement = previous_best_fitness - best_fitness
            if fitness_improvement < delta_threshold:
                stable_generations += 1
            else:
                stable_generations = 0
            
            if self.verbose:
                print(f"****** Generation {generation + 1}: Average Fitness = {average_fitness:.4f}, Best Fitness = {best_fitness:.4f}, fitness improvement = {fitness_improvement:.4f}")
            
            previous_best_fitness = best_fitness
            
            # Stop if convergence reached
            if stable_generations >= patience:
                if self.verbose:
                    print(f"Convergence reached at generation {generation + 1}")
                break
            
            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.7:  # crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            
            # Inject random immigrants to maintain genetic diversity (10% of population)
            # This prevents premature convergence and helps escape local optima
            # Enhanced: immigrants are now escape-aware, biased toward target class bounds
            num_immigrants = max(1, int(0.1 * population_size))
            target_constraints = self.constraints.get(f"Class {target_class}", [])
            
            for i in range(num_immigrants):
                # Create a new random individual within constraint boundaries, biased by escape direction
                immigrant = {}
                for feature in feature_names:
                    norm_feature = self._normalize_feature_name(feature)
                    escape_dir = escape_directions.get(norm_feature, 'both')
                    
                    # Find constraint for this feature
                    matching_constraint = next(
                        (c for c in target_constraints if self._features_match(c.get("feature", ""), feature)),
                        None
                    )
                    
                    if matching_constraint:
                        feature_min = matching_constraint.get('min')
                        feature_max = matching_constraint.get('max')
                        
                        # Generate random value within constraints, biased by escape direction
                        if feature_min is not None and feature_max is not None:
                            if escape_dir == 'increase':
                                # Bias toward upper half of target range
                                mid = (feature_min + feature_max) / 2
                                immigrant[feature] = np.random.uniform(mid, feature_max)
                            elif escape_dir == 'decrease':
                                # Bias toward lower half of target range
                                mid = (feature_min + feature_max) / 2
                                immigrant[feature] = np.random.uniform(feature_min, mid)
                            else:
                                immigrant[feature] = np.random.uniform(feature_min, feature_max)
                        elif feature_min is not None:
                            immigrant[feature] = np.random.uniform(feature_min, feature_min + 2.0)
                        elif feature_max is not None:
                            immigrant[feature] = np.random.uniform(max(0, feature_max - 2.0), feature_max)
                        else:
                            # No constraints - use original sample ± random offset
                            immigrant[feature] = sample[feature] + np.random.uniform(-1.0, 1.0)
                    else:
                        # No constraint found - use original sample ± random offset
                        immigrant[feature] = sample[feature] + np.random.uniform(-1.0, 1.0)
                    
                    # Ensure non-negative and round
                    immigrant[feature] = np.round(max(0, immigrant[feature]), 2)
                
                # Replace one of the worst offspring with this immigrant
                immigrant_ind = creator.Individual(immigrant)
                offspring[-(i + 1)] = immigrant_ind
            
            # Reduce mutation rate over generations (adaptive mutation)
            current_mutation_rate *= 0.99
            toolbox.unregister("mutate")
            toolbox.register("mutate", self._mutate_individual, 
                           sample=sample, 
                           feature_names=feature_names,
                           mutation_rate=current_mutation_rate,
                           target_class=target_class,
                           original_class=original_class,
                           boundary_analysis=boundary_analysis)
            
            # Elitism: Preserve best individuals from current population
            # Keep top 10% of current population (minimum 1, maximum 5)
            elite_size = max(1, min(5, int(0.1 * population_size)))
            
            # Sort current population by fitness (best first for minimization)
            sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])
            elites = sorted_population[:elite_size]
            
            # Replace population with offspring, but keep elites
            # Replace worst individuals in offspring with elites
            population[:] = offspring[:-elite_size] + elites
        
        # Clean up multiprocessing pool if used
        if n_jobs != 1:
            pool.close()
            pool.join()
        
        # Return the best individuals found
        # Check for both np.inf and INVALID_FITNESS (1e6) to detect failed counterfactuals
        INVALID_FITNESS = 1e6
        valid_counterfactuals = []
        
        for i in range(len(hof)):
            best_fitness = hof[i].fitness.values[0]
            if best_fitness == np.inf or best_fitness >= INVALID_FITNESS:
                if self.verbose:
                    print(f"Counterfactual #{i+1} generation failed: fitness = {best_fitness}")
                continue
            
            # Final validation: verify the individual actually predicts the target class
            # AND has sufficient probability margin over other classes
            best_individual = dict(hof[i])
            features = np.array([best_individual[f] for f in sample.keys()]).reshape(1, -1)
            
            try:
                if self.feature_names is not None:
                    features_df = pd.DataFrame(features, columns=self.feature_names)
                    predicted_class = self.model.predict(features_df)[0]
                    proba = self.model.predict_proba(features_df)[0]
                else:
                    predicted_class = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                
                if predicted_class != target_class:
                    if self.verbose:
                        print(f"Counterfactual #{i+1} failed: predicts class {predicted_class}, not target {target_class}")
                    continue
                
                # Check probability margin - target class should be clearly higher than second-best
                # NOTE: proba is indexed by position, not by class label.
                # Use model.classes_ to find the correct index for target_class
                if hasattr(self.model, 'classes_'):
                    class_list = list(self.model.classes_)
                    if target_class in class_list:
                        target_idx = class_list.index(target_class)
                    else:
                        target_idx = target_class  # Fallback to direct indexing
                else:
                    target_idx = target_class
                target_prob = proba[target_idx]
                sorted_probs = np.sort(proba)[::-1]  # Descending order
                second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                margin = target_prob - second_best_prob
                
                if margin < self.min_probability_margin:
                    if self.verbose:
                        print(f"Counterfactual #{i+1} rejected: target class probability ({target_prob:.3f}) "
                              f"not sufficiently higher than second-best ({second_best_prob:.3f}). "
                              f"Margin {margin:.3f} < required {self.min_probability_margin:.3f}")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"Counterfactual #{i+1} validation failed with error: {e}")
                continue
            
            valid_counterfactuals.append(best_individual)
            if len(valid_counterfactuals) >= num_best_results:
                break
        
        # Return None if no valid counterfactuals found, otherwise return the list
        if not valid_counterfactuals:
            return None
        return valid_counterfactuals

    def generate_counterfactual(self, sample, target_class, population_size=100, generations=100, 
                                  mutation_rate=0.8, n_jobs=-1, allow_relaxation=True, relaxation_factor=2.0, num_best_results=1):
        """
        Generate a counterfactual for the given sample and target class using a genetic algorithm.
        
        Enhanced with dual-boundary support: the GA uses both original and target class
        constraints to guide evolution, escaping original bounds while approaching target bounds.
        
        If allow_relaxation=True and strict constraints fail, automatically retries with
        progressively relaxed constraints to ensure a valid counterfactual is found.

        Args:
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.
            population_size (int): Size of the population for the genetic algorithm.
            generations (int): Number of generations to run.
            mutation_rate (float): Per-feature mutation probability.
            n_jobs (int): Number of parallel jobs. -1=all CPUs (default), 1=sequential.
            allow_relaxation (bool): If True, retry with relaxed constraints on failure.
            relaxation_factor (float): Factor to expand constraint bounds by (2.0 = double range).
            num_best_results (int): Number of top individuals to return from single GA run.

        Returns:
            list or None: A list of modified samples representing counterfactuals, or None if not found.
        """
        sample_class = self.model.predict(pd.DataFrame([sample]))[0]

        if sample_class == target_class:
            raise ValueError("Target class need to be different from the predicted class label.")

        # Pass original_class to enable dual-boundary GA
        counterfactuals = self.genetic_algorithm(
            sample, target_class, population_size, generations,
            mutation_rate=mutation_rate, n_jobs=n_jobs, original_class=sample_class, num_best_results=num_best_results
        )
        
        # If strict constraints failed and relaxation is allowed, try with relaxed constraints
        if (counterfactuals is None or len(counterfactuals) == 0) and allow_relaxation and self.constraints:
            if self.verbose:
                print("\nStrict constraints failed. Attempting with relaxed constraints...")
            
            # Store original constraints
            original_constraints = self.constraints
            
            # Try progressively relaxed constraints
            for relax_level in [relaxation_factor, relaxation_factor * 2, None]:
                if relax_level is None:
                    # Final attempt: no constraints (pure model optimization)
                    if self.verbose:
                        print("  Attempting without DPG constraints (pure classification optimization)...")
                    self.constraints = {}
                else:
                    if self.verbose:
                        print(f"  Attempting with {relax_level}x relaxed constraints...")
                    self.constraints = self._relax_constraints(original_constraints, relax_level)
                
                counterfactuals = self.genetic_algorithm(
                    sample, target_class, population_size, generations,
                    mutation_rate=mutation_rate, n_jobs=n_jobs, original_class=sample_class, num_best_results=num_best_results
                )
                
                if counterfactuals is not None and len(counterfactuals) > 0:
                    if self.verbose:
                        # Check constraint validity with original constraints
                        is_valid, penalty = self.validate_constraints(
                            counterfactuals[0], sample, target_class, 
                            original_class=sample_class, strict_mode=True
                        )
                        print(f"  Found {len(counterfactuals)} counterfactual(s) (original constraint valid: {is_valid}, penalty: {penalty:.2f}")
                    break
            
            # Restore original constraints
            self.constraints = original_constraints
        
        # Ultimate fallback: use nearest neighbor from training data
        if (counterfactuals is None or len(counterfactuals) == 0) and allow_relaxation:
            if self.verbose:
                print("\nGA methods failed. Attempting nearest neighbor fallback...")
            neighbor_cf = self.find_nearest_counterfactual(
                sample, target_class, validate_prediction=True
            )
            if neighbor_cf is not None:
                if self.verbose:
                    print("  Nearest neighbor fallback succeeded!")
                counterfactuals = [neighbor_cf]
            else:
                counterfactuals = None
        
        return counterfactuals
    
    def _relax_constraints(self, constraints, factor=2.0):
        """
        Relax constraints by expanding their bounds by a factor.
        
        Args:
            constraints: Original constraint dictionary in format:
                {class_key: {feature_name: {'min': ..., 'max': ...}, ...}, ...}
            factor: Factor to expand bounds by (e.g., 2.0 doubles the range)
            
        Returns:
            dict: Relaxed constraints in same format
        """
        relaxed = {}
        for class_key, class_constraints in constraints.items():
            relaxed[class_key] = {}
            
            # Handle both dict and list formats
            if isinstance(class_constraints, dict):
                # Format: {feature_name: {'min': ..., 'max': ...}, ...}
                for feature, bounds in class_constraints.items():
                    if isinstance(bounds, dict):
                        min_val = bounds.get('min')
                        max_val = bounds.get('max')
                        
                        if min_val is not None and max_val is not None:
                            # Expand the range by factor
                            center = (min_val + max_val) / 2
                            half_range = (max_val - min_val) / 2
                            relaxed[class_key][feature] = {
                                'min': center - half_range * factor,
                                'max': center + half_range * factor
                            }
                        else:
                            relaxed[class_key][feature] = bounds.copy()
                    else:
                        relaxed[class_key][feature] = bounds
            elif isinstance(class_constraints, list):
                # Legacy format: [{'feature': ..., 'min': ..., 'max': ...}, ...]
                relaxed[class_key] = []
                for c in class_constraints:
                    feature = c.get('feature', '')
                    min_val = c.get('min')
                    max_val = c.get('max')
                    
                    if min_val is not None and max_val is not None:
                        center = (min_val + max_val) / 2
                        half_range = (max_val - min_val) / 2
                        relaxed[class_key].append({
                            'feature': feature,
                            'min': center - half_range * factor,
                            'max': center + half_range * factor
                        })
                    else:
                        relaxed[class_key].append(c.copy())
        return relaxed

    def find_nearest_counterfactual(self, sample, target_class, X_train=None, y_train=None, 
                                    metric='euclidean', validate_prediction=True):
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
        from scipy.spatial.distance import cdist
        
        feature_names = list(sample.keys())
        sample_array = np.array([sample[f] for f in feature_names]).reshape(1, -1)
        
        # Try to get training data from the model if not provided
        if X_train is None:
            # Try to access training data if stored
            X_train = getattr(self, 'X_train', None)
        if y_train is None:
            y_train = getattr(self, 'y_train', None)
            
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
                        candidate_df = pd.DataFrame(candidate, columns=self.feature_names)
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
                    if hasattr(self.model, 'classes_'):
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
                            print(f"  Skipping candidate with weak margin: {margin:.3f} < {self.min_probability_margin}")
                        continue  # Skip samples with weak probability margin
                except Exception as e:
                    if self.verbose:
                        print(f"Prediction failed: {e}")
                    continue
            
            # Convert to dict
            candidate_dict = {feature_names[i]: candidate[0][i] for i in range(len(feature_names))}
            
            if self.verbose:
                print(f"Found nearest counterfactual at distance {distances[idx]:.2f}")
                # Check constraint validity
                if self.constraints:
                    original_class = self.model.predict(pd.DataFrame([sample]))[0]
                    is_valid, penalty = self.validate_constraints(
                        candidate_dict, sample, target_class, original_class=original_class
                    )
                    print(f"  DPG constraint valid: {is_valid}, penalty: {penalty:.2f}")
            
            return candidate_dict
        
        if self.verbose:
            print(f"No valid counterfactual found among {len(target_samples)} target class samples")
        return None

