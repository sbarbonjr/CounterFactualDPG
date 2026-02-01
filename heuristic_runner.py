"""
HeuristicRunner: Generates counterfactual candidates using heuristic approach.

Generates a pool of candidate counterfactuals using escape-aware perturbations,
evaluates fitness, and returns the best candidates using greedy diverse selection.
No iterative evolution - single-pass candidate generation and evaluation.
"""

import numpy as np
import pandas as pd

from constants import INVALID_FITNESS


class Fitness:
    """Simple fitness container for compatibility with HOF."""
    def __init__(self, value):
        self.values = (value,)
        self.valid = True


class Individual(dict):
    """
    Dict subclass that can hold a fitness attribute.
    
    Replaces DEAP's Individual for simple dict-based individuals
    that need fitness tracking.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = Fitness(float('inf'))


class DistanceBasedHOF:
    """
    Hall of Fame that maintains diverse candidates while considering distance.
    
    Uses a diversity-aware selection strategy:
    1. Always keeps the closest candidate (best proximity)
    2. For remaining slots, uses greedy diverse selection balancing proximity and diversity
    
    This ensures the HOF contains candidates that are:
    - Close to the original (for good counterfactuals)
    - Diverse from each other (for meaningful selection later)
    """
    
    def __init__(self, maxsize, original_features, feature_names, diversity_weight=0.5):
        self.maxsize = maxsize
        self.original_features = original_features
        self.feature_names = feature_names
        self.diversity_weight = diversity_weight  # Balance between diversity (1.0) and proximity (0.0)
        self.items = []
    
    def update(self, population):
        """Update HOF with individuals, maintaining diversity while keeping good proximity."""
        # Collect all valid candidates
        new_candidates = []
        for ind in population:
            # Skip invalid fitness individuals
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                continue
            if ind.fitness.values[0] >= INVALID_FITNESS:
                continue
            
            # Calculate distance to original
            ind_array = np.array([ind[f] for f in self.feature_names])
            distance = np.linalg.norm(ind_array - self.original_features)
            
            # Check for duplicates in new_candidates
            is_dup = False
            for _, _, existing_array in new_candidates:
                if np.allclose(ind_array, existing_array, atol=0.01):
                    is_dup = True
                    break
            
            # Also check existing items
            if not is_dup:
                for _, existing_ind in self.items:
                    existing_array = np.array([existing_ind[f] for f in self.feature_names])
                    if np.allclose(ind_array, existing_array, atol=0.01):
                        is_dup = True
                        break
            
            if not is_dup:
                new_candidates.append((distance, dict(ind), ind_array))
        
        # Merge with existing items
        all_candidates = []
        for dist, ind in self.items:
            ind_array = np.array([ind[f] for f in self.feature_names])
            all_candidates.append((dist, ind, ind_array))
        all_candidates.extend(new_candidates)
        
        if not all_candidates:
            return
        
        # Sort by distance to get proximity ranking
        all_candidates.sort(key=lambda x: x[0])
        
        # Calculate distance range for normalization
        distances = [c[0] for c in all_candidates]
        min_dist, max_dist = min(distances), max(distances)
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        # Greedy diverse selection for HOF
        selected = []
        selected_arrays = []
        
        # Always include the closest candidate first
        if all_candidates:
            best = all_candidates[0]
            selected.append((best[0], best[1]))
            selected_arrays.append(best[2])
            remaining = all_candidates[1:]
        else:
            remaining = []
        
        # For remaining slots, use diversity-aware selection
        while len(selected) < self.maxsize and remaining:
            best_score = float('-inf')
            best_idx = 0
            
            for i, (dist, ind, ind_array) in enumerate(remaining):
                # Normalize distance (0 = closest, 1 = farthest)
                norm_dist = (dist - min_dist) / dist_range if dist_range > 0 else 0
                
                # Calculate minimum diversity from already selected
                min_diversity = float('inf')
                for sel_array in selected_arrays:
                    div = np.linalg.norm(ind_array - sel_array)
                    min_diversity = min(min_diversity, div)
                
                # Normalize diversity (estimate max possible diversity from data)
                # Use distance range as rough estimate of feature space spread
                max_possible_div = dist_range * 2 if dist_range > 0 else 1.0
                norm_diversity = min(min_diversity / max_possible_div, 1.0)
                
                # Amplify diversity with sqrt for better discrimination
                amplified_diversity = np.sqrt(norm_diversity)
                
                # Score: high diversity, low distance
                # diversity_weight controls the trade-off
                proximity_score = 1.0 - norm_dist  # Higher is better (closer)
                score = self.diversity_weight * amplified_diversity + (1 - self.diversity_weight) * proximity_score
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Add best candidate
            best = remaining[best_idx]
            selected.append((best[0], best[1]))
            selected_arrays.append(best[2])
            remaining.pop(best_idx)
        
        self.items = selected
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx][1]
    
    def __iter__(self):
        return iter([item[1] for item in self.items])


class HeuristicRunner:
    """
    Generates counterfactual candidates using heuristic approach.
    
    Uses escape-aware perturbations to generate a pool of candidates,
    evaluates fitness for validity checking, and returns best candidates
    using greedy diverse selection based on distance to original.
    """

    def __init__(
        self,
        model,
        constraints,
        dict_non_actionable=None,  
        feature_names=None,
        verbose=False,
        min_probability_margin=0.001,
        diversity_lambda=0.5,
    ):
        """
        Initialize the heuristic runner.

        Args:
            model: Trained ML model with predict() and predict_proba() methods.
            constraints (dict): Feature constraints per class.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints
                (non_decreasing, non_increasing, no_change).
            feature_names (list): List of feature names (for DataFrame model input).
            verbose (bool): Whether to print progress messages.
            min_probability_margin (float): Minimum probability difference between
                target class and second-best class for valid counterfactuals.
            diversity_lambda (float): Weight for diversity vs proximity trade-off (0-1).
                Higher values prioritize diversity, lower values prioritize fitness/proximity.
                Default 0.5 for balanced selection.
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable 
        self.feature_names = feature_names
        self.verbose = verbose
        self.min_probability_margin = min_probability_margin
        self.diversity_lambda = diversity_lambda

        # Tracking attributes
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.std_fitness_list = []
        self.evolution_history = []
        self.per_cf_evolution_histories = []
        self.cf_generation_found = []
        self._original_features = None

    def run(
        self,
        sample,
        target_class,
        original_class,
        population_size,
        metric,
        num_best_results,
        boundary_analysis,
        # Callbacks for CounterFactualModel methods
        create_individual_func,
        calculate_fitness_func,
        get_valid_sample_func,
        normalize_feature_func,
        features_match_func,
        overgeneration_factor=5,
        weak_constraints=True,
    ):
        """
        Generate counterfactual candidates using heuristic approach.

        Args:
            sample (dict): Original sample features.
            target_class (int): Target class for counterfactual.
            original_class (int): Original class for escape-aware generation.
            population_size (int): Number of candidates to generate.
            metric (str): Distance metric for fitness calculation.
            num_best_results (int): Number of top individuals to return.
            boundary_analysis (dict): Pre-computed boundary analysis between classes.
            create_individual_func: Function to create individual dict.
            calculate_fitness_func: Function to calculate fitness.
            get_valid_sample_func: Function to generate valid sample.
            normalize_feature_func: Function to normalize feature names.
            features_match_func: Function to check if features match.
            overgeneration_factor (int): Generate this many times the requested CFs.
            weak_constraints (bool): Extend DPG bounds to include original value.

        Returns:
            list: Valid counterfactuals (or None if none found).
        """
        feature_names = list(sample.keys())
        original_features = np.array([sample[feature] for feature in feature_names])
        
        # Store for selection
        self._original_features = original_features
        
        # Calculate internal num_results for overgeneration (generate 5x, return top N)
        internal_num_results = num_best_results * overgeneration_factor

        # Log dual-boundary info
        if boundary_analysis and self.verbose:
            non_overlapping = boundary_analysis.get("non_overlapping", [])
            print(f"[Dual-Boundary] Non-overlapping features: {non_overlapping}")
            print(
                f"[Dual-Boundary] Escape directions: {boundary_analysis.get('escape_direction', {})}"
            )

        # Initialize population of candidates
        population = self._initialize_population(
            sample,
            target_class,
            original_class,
            population_size,
            feature_names,
            boundary_analysis,
            create_individual_func,
            get_valid_sample_func,
            normalize_feature_func,
            features_match_func,
            weak_constraints,
        )

        # Reset tracking
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.std_fitness_list = []
        self.evolution_history = []
        self.per_cf_evolution_histories = []
        self.cf_generation_found = []

        # Evaluate fitness for all candidates
        fitnesses = []
        for ind in population:
            fitness = calculate_fitness_func(
                ind,
                original_features,
                sample,
                target_class,
                metric,
                population,
                original_class,
            )
            ind.fitness = Fitness(fitness)
            fitnesses.append(fitness)

        # Compute statistics for tracking (single generation)
        valid_fitnesses = [f for f in fitnesses if f < INVALID_FITNESS and not np.isinf(f)]
        if valid_fitnesses:
            best_fitness = min(valid_fitnesses)
            avg_fitness = np.mean(valid_fitnesses)
            std_fitness = np.std(valid_fitnesses)
        else:
            best_fitness = float('inf')
            avg_fitness = float('nan')
            std_fitness = float('nan')

        self.best_fitness_list = [best_fitness]
        self.average_fitness_list = [avg_fitness]
        self.std_fitness_list = [std_fitness]

        # Track best individual for evolution history
        best_ind = min(population, key=lambda x: x.fitness.values[0])
        if best_ind.fitness.values[0] < INVALID_FITNESS:
            entry = dict(best_ind)
            entry["_fitness"] = best_ind.fitness.values[0]
            self.evolution_history = [entry]

        if self.verbose:
            print(f"Candidates generated: {len(population)}, Best fitness: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")
            print(f"Overgeneration: requesting {internal_num_results} CFs to select best {num_best_results}")

        # Create HOF using diversity-aware ranking
        # Use larger capacity to preserve diverse candidates, pass diversity_lambda for balancing
        hof = DistanceBasedHOF(
            maxsize=internal_num_results * 6,  # Larger pool for diversity
            original_features=original_features,
            feature_names=feature_names,
            diversity_weight=self.diversity_lambda  # Use same lambda for HOF diversity
        )
        hof.update(population)

        # Validate and return results
        return self._validate_counterfactuals(
            hof,
            sample,
            target_class,
            num_best_results,
            internal_num_results,
            calculate_fitness_func,
            original_features,
            metric,
            population,
            original_class,
        )

    def _initialize_population(
        self,
        sample,
        target_class,
        original_class,
        population_size,
        feature_names,
        boundary_analysis,
        create_individual_func,
        get_valid_sample_func,
        normalize_feature_func,
        features_match_func,
        weak_constraints,
    ):
        """
        Create initial population with escape-aware perturbations.
        
        Returns list of Individual objects that can hold fitness.
        """
        # First individual is the original sample adjusted to constraint boundaries
        base_counterfactual = get_valid_sample_func(sample, target_class, original_class, weak_constraints)
        if self.verbose:
            print("[HeuristicRunner] Base counterfactual (constraint-adjusted):", base_counterfactual)
        
        
        
        population = [Individual(create_individual_func(base_counterfactual.copy(), feature_names))]

        # Get constraints and escape directions
        target_constraints = self.constraints.get(f"Class {target_class}", [])
        raw_escape_directions = (
            boundary_analysis.get("escape_direction", {}) if boundary_analysis else {}
        )
        
        # Pre-compute feature bounds for all features (for stratified sampling)
        # AND apply escape direction overrides based on actual sample position
        feature_bounds = {}
        escape_directions = {}  # Corrected escape directions
        
        for feature in feature_names:
            matching_constraint = next(
                (c for c in target_constraints if features_match_func(c.get("feature", ""), feature)),
                None,
            )
            if matching_constraint:
                # Raw target bounds (before weak_constraints extension)
                raw_target_min = matching_constraint.get("min")
                raw_target_max = matching_constraint.get("max")
                original_value = sample[feature]
                
                # Apply weak_constraints extension
                feature_min = raw_target_min
                feature_max = raw_target_max
                if weak_constraints:
                    if feature_min is not None:
                        feature_min = min(feature_min, original_value)
                    if feature_max is not None:
                        feature_max = max(feature_max, original_value)
                
                feature_bounds[feature] = {
                    'min': feature_min,
                    'max': feature_max,
                    'base': base_counterfactual[feature],
                    'original': sample[feature],  # Store original for diversity exploration
                    'range': (feature_max - feature_min) if (feature_min is not None and feature_max is not None) else 1.0
                }
                
                # CRITICAL: Override escape direction based on actual sample position
                # (same logic as in sample_generator.py)
                norm_feature = normalize_feature_func(feature)
                escape_dir = raw_escape_directions.get(norm_feature, "both")
                
                if raw_target_min is not None and raw_target_max is not None:
                    if original_value < raw_target_min and escape_dir == "decrease":
                        # Sample is BELOW target range, must INCREASE not decrease
                        if self.verbose:
                            print(f"[HeuristicRunner] OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                        escape_dir = "increase"
                    elif original_value > raw_target_max and escape_dir == "increase":
                        # Sample is ABOVE target range, must DECREASE not increase
                        if self.verbose:
                            print(f"[HeuristicRunner] OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                        escape_dir = "decrease"
                elif raw_target_min is not None and original_value < raw_target_min and escape_dir == "decrease":
                    if self.verbose:
                        print(f"[HeuristicRunner] OVERRIDE {feature}: escape=decrease→increase (value {original_value:.2f} < target_min {raw_target_min:.2f})")
                    escape_dir = "increase"
                elif raw_target_max is not None and original_value > raw_target_max and escape_dir == "increase":
                    if self.verbose:
                        print(f"[HeuristicRunner] OVERRIDE {feature}: escape=increase→decrease (value {original_value:.2f} > target_max {raw_target_max:.2f})")
                    escape_dir = "decrease"
                
                escape_directions[norm_feature] = escape_dir
            else:
                feature_bounds[feature] = None  # No constraint, keep original

        # Identify critical features - those that actually changed in base counterfactual
        # These are the features we should prioritize for perturbation
        critical_features = []
        for feature in feature_names:
            if abs(base_counterfactual[feature] - sample[feature]) > 0.01:
                critical_features.append(feature)
        
        if self.verbose:
            print(f"[HeuristicRunner] Critical features (changed in base CF): {critical_features}")
            print(f"[HeuristicRunner] Non-critical features (keep near base): {[f for f in feature_names if f not in critical_features]}")

        # Create remaining individuals using STRATIFIED sampling for diversity
        # Strategy: divide the remaining slots into tiers that explore different regions
        # KEY INSIGHT: To get diversity, we need to explore at different "depths" from original to boundary
        # NEW: Focus perturbations on critical features, keep non-critical near base values
        remaining_slots = population_size - 1
        
        for individual in range(remaining_slots):
            perturbed = base_counterfactual.copy()
            
            # Use stratified tiers with different exploration strategies
            # t goes from 0 to 1 across the population
            t = (individual + 1) / remaining_slots
            
            for feature in feature_names:
                bounds = feature_bounds.get(feature)
                if bounds is None:
                    perturbed[feature] = sample[feature]
                    continue
                
                feature_min = bounds['min']
                feature_max = bounds['max']
                feature_range = bounds['range']
                base_val = bounds['base']
                original_val = bounds['original']
                
                norm_feature = normalize_feature_func(feature)
                escape_dir = escape_directions.get(norm_feature, "both")
                
                # Determine the target extreme based on escape direction
                if escape_dir == "increase":
                    target_extreme = feature_max
                elif escape_dir == "decrease":
                    target_extreme = feature_min
                else:  # "both"
                    # Alternate between min and max based on individual index
                    target_extreme = feature_max if individual % 2 == 0 else feature_min
                
                # Handle None target_extreme - fallback to base value perturbation
                if target_extreme is None:
                    # No valid target extreme, just perturb around base
                    perturbation = np.random.uniform(-0.25, 0.25) * feature_range
                    perturbed[feature] = base_val + perturbation
                elif t < 0.15:
                    # Tier 1 (15%): Near base CF - small perturbations for proximity
                    perturbation = np.random.uniform(-0.2, 0.2) * feature_range
                    perturbed[feature] = base_val + perturbation
                elif t < 0.4:
                    # Tier 2 (25%): Medium exploration around base
                    # Determine if this is a critical or non-critical feature
                    is_critical = feature in critical_features
                    
                    # CRITICAL: Use different perturbation scales for critical vs non-critical
                    # Critical features get full perturbation range for diversity
                    # Non-critical features get smaller perturbations to stay valid
                    if is_critical:
                        perturbation_scale = 0.3 + (t - 0.15) * 0.8  # 0.3 to 0.5 (full range)
                    else:
                        # Non-critical: only small perturbations to maintain validity
                        perturbation_scale = 0.05 + (t - 0.15) * 0.15  # 0.05 to 0.2 (much smaller)
                    
                    if escape_dir == "increase":
                        if is_critical:
                            perturbation = np.random.uniform(-0.15 * feature_range, perturbation_scale * feature_range)
                        else:
                            perturbation = np.random.uniform(-0.05 * feature_range, perturbation_scale * feature_range)
                    elif escape_dir == "decrease":
                        if is_critical:
                            perturbation = np.random.uniform(-perturbation_scale * feature_range, 0.15 * feature_range)
                        else:
                            perturbation = np.random.uniform(-perturbation_scale * feature_range, 0.05 * feature_range)
                    else:
                        if is_critical:
                            perturbation = np.random.uniform(-perturbation_scale * feature_range / 2, 
                                                              perturbation_scale * feature_range / 2)
                        else:
                            perturbation = np.random.uniform(-perturbation_scale * feature_range / 3,
                                                              perturbation_scale * feature_range / 3)
                    perturbed[feature] = base_val + perturbation
                else:
                    # Tier 3-4 (60%): WIDE exploration for diversity - SIGNIFICANTLY INCREASED
                    # Create feature-wise variation: randomly decide if this feature pushes toward
                    # extreme or stays conservative. This creates diverse combinations.
                    # Use individual index + feature index for deterministic but varied patterns
                    feature_idx = list(feature_names).index(feature) if feature in feature_names else 0
                    is_critical = feature in critical_features
                    
                    # CRITICAL: Different patterns for critical vs non-critical features
                    # Non-critical features get reduced push factors to maintain validity
                    if is_critical:
                        pattern = (individual + feature_idx) % 5  # 5 patterns for variety
                        # Pattern 0: Strong push, 1: Medium-high, 2: Medium, 3: Conservative, 4: Alternative
                        # Corresponds to push_factors: [0.7-1.0, 0.5-0.8, 0.25-0.6, 0.0±0.25, alternative]
                    else:
                        # Use different pattern set for non-critical to reduce extreme changes
                        pattern = (individual + feature_idx) % 3  # 3 patterns (more conservative)
                        # Pattern 0: Small push (0.1), 1: Conservative (0.0±0.15), 2: Very small push (0.05)
                    
                    if is_critical:
                        if pattern == 0:
                            # Push strongly toward extreme (max for increase, min for decrease)
                            push_factor = 0.7 + np.random.uniform(0, 0.3)  # 0.7 to 1.0 toward extreme
                            perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                        elif pattern == 1:
                            # Medium-high push toward extreme
                            push_factor = 0.5 + np.random.uniform(0, 0.3)  # 0.5 to 0.8 toward extreme
                            perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                        elif pattern == 2:
                            # Medium push
                            push_factor = 0.25 + np.random.uniform(0, 0.35)  # 0.25 to 0.6 toward extreme
                            perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                        elif pattern == 3:
                            # Moderate perturbation around base
                            perturbed[feature] = base_val + np.random.uniform(-0.25, 0.25) * feature_range
                        else:
                            # Counter-direction exploration if escape is "both", else medium push
                            if escape_dir == "both":
                                counter_extreme = feature_min if target_extreme == feature_max else feature_max
                                if counter_extreme is not None:
                                    perturbed[feature] = base_val + np.random.uniform(0.1, 0.4) * (counter_extreme - base_val)
                                else:
                                    perturbed[feature] = base_val + np.random.uniform(-0.3, 0.3) * feature_range
                            else:
                                push_factor = 0.3 + np.random.uniform(0, 0.4)  # 0.3 to 0.7 toward extreme
                                perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                    else:
                        # Non-critical features: apply much more conservative perturbations
                        if pattern == 0:
                            # Small push toward extreme (0.1)
                            push_factor = 0.05 + np.random.uniform(0, 0.1)  # 0.05 to 0.15 toward extreme
                            perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                        elif pattern == 1:
                            # Conservative perturbation around base
                            perturbed[feature] = base_val + np.random.uniform(-0.15, 0.15) * feature_range
                        else:
                            # Very small push toward extreme (0.05)
                            push_factor = 0.03 + np.random.uniform(0, 0.05)  # 0.03 to 0.08 toward extreme
                            perturbed[feature] = base_val + push_factor * (target_extreme - base_val)
                    
                    # CRITICAL ADDITION: For non-critical features, blend heavily toward base
                    # This ensures they stay close to the working base counterfactual
                    if not is_critical:
                        # Blend base_val with perturbed value (80% base, 20% perturbed)
                        # This strongly constrains non-critical features to stay near working values
                        perturbed[feature] = 0.8 * base_val + 0.2 * perturbed[feature]
                
                # Clip to bounds
                if feature_min is not None:
                    perturbed[feature] = max(feature_min, perturbed[feature])
                if feature_max is not None:
                    perturbed[feature] = min(feature_max, perturbed[feature])

                # ENFORCE ACTIONABILITY CONSTRAINTS AFTER clipping (actionability wins over constraints)
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionability = self.dict_non_actionable[feature]
                    original_value = sample[feature]
                    
                    if actionability == "no_change":
                        # Feature cannot change at all
                        perturbed[feature] = original_value
                    elif actionability == "non_decreasing":
                        # Feature can only increase or stay the same
                        perturbed[feature] = max(perturbed[feature], original_value)
                    elif actionability == "non_increasing":
                        # Feature can only decrease or stay the same
                        perturbed[feature] = min(perturbed[feature], original_value)

                perturbed[feature] = np.round(perturbed[feature], 5)

            if self.verbose:
                print(f"[HeuristicRunner] Added individual with perturbed features: {perturbed}")

            population.append(Individual(create_individual_func(perturbed, feature_names)))

        return population

    def _validate_counterfactuals(
        self, hof, sample, target_class, num_best_results, internal_num_results,
        calculate_fitness_func, original_features, metric, population, original_class
    ):
        """
        Validate counterfactuals and build evolution histories.
        
        NEW APPROACH: Fitness-first with diverse selection
        1. Build pool of valid CFs from HOF
        2. Calculate fitness for ALL valid CFs
        3. Sort by fitness to get quality-ranked list
        4. Apply greedy diverse selection on top fitness candidates to pick diverse CFs
        
        This ensures we pick from high-quality CFs while maintaining diversity.
        """
        feature_names = list(sample.keys())
        original_features = self._original_features
        
        # Use instance-level diversity_lambda (configurable)
        # Higher values prioritize diversity, lower values prioritize fitness/proximity
        diversity_lambda = self.diversity_lambda
        
        if self.verbose:
            print("\n=== CF Selection (Fitness -> Diverse Selection) ===")
            print(f"Final requested: {num_best_results} CFs")
            print(f"Overgeneration pool: {internal_num_results} CFs")
            print(f"Available in HOF: {len(hof)}")
        
        # STEP 1: Build candidate pool from HOF with fitness calculation
        candidates = []
        for ind in hof:
            cf_dict = dict(ind) if not isinstance(ind, dict) else ind
            cf_array = np.array([cf_dict[f] for f in feature_names])
            distance = np.linalg.norm(cf_array - original_features)
            
            # Verify prediction
            try:
                if self.feature_names is not None:
                    features_df = pd.DataFrame([cf_dict])
                    predicted_class = self.model.predict(features_df)[0]
                    proba = self.model.predict_proba(features_df)[0]
                else:
                    features = np.array([cf_dict[f] for f in feature_names]).reshape(1, -1)
                    predicted_class = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                
                # Skip if doesn't predict target class
                if predicted_class != target_class:
                    if self.verbose:
                        print(f"[VERBOSE-DPG] Candidate rejected: predicted class {predicted_class} != target {target_class}")
                    continue
                
                # Check probability margin
                if hasattr(self.model, "classes_"):
                    class_list = list(self.model.classes_)
                    target_idx = class_list.index(target_class) if target_class in class_list else target_class
                else:
                    target_idx = target_class
                
                target_prob = proba[target_idx]
                sorted_probs = np.sort(proba)[::-1]
                second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                margin = target_prob - second_best_prob
                
                if margin < self.min_probability_margin:
                    if self.verbose:
                        print(f"[VERBOSE-DPG] Candidate rejected: margin {margin:.6f} < {self.min_probability_margin:.6f} (target_prob={target_prob:.4f}, second={second_best_prob:.4f})")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"[VERBOSE-DPG] Candidate rejected: prediction error {e}")
                continue
            
            # Calculate fitness for this candidate
            fitness = calculate_fitness_func(
                cf_dict,
                original_features,
                sample,
                target_class,
                metric,
                population,
                original_class,
            )
            
            candidates.append({
                'cf': cf_dict,
                'array': cf_array,
                'distance': distance,
                'fitness': fitness,
            })
        
        if self.verbose:
            print(f"[VERBOSE-DPG] Valid candidates after filtering: {len(candidates)}/{len(hof)} from HOF")
            if len(hof) > len(candidates):
                rejected = len(hof) - len(candidates)
                print(f"[VERBOSE-DPG] Rejected {rejected} candidates (wrong class or insufficient margin)")
        
        if not candidates:
            if self.verbose:
                print(f"[VERBOSE-DPG] No valid candidates found - all {len(hof)} HOF entries rejected")
            return None
        
        # STEP 2: Calculate normalization ranges from ALL candidates BEFORE filtering
        # This ensures diversity_lambda has real effect by using full fitness range
        final_counterfactuals = []
        selected_arrays = []
        
        if len(candidates) > 1:
            all_fitnesses = [c['fitness'] for c in candidates]
            fitness_range = max(all_fitnesses) - min(all_fitnesses)
            fitness_min = min(all_fitnesses)
            
            # Estimate max possible diversity from ALL candidates
            all_arrays = [c['array'] for c in candidates]
            max_diversity = 0
            for i in range(len(all_arrays)):
                for j in range(i+1, len(all_arrays)):
                    d = np.linalg.norm(all_arrays[i] - all_arrays[j])
                    max_diversity = max(max_diversity, d)
            max_diversity = max(max_diversity, 0.01)  # Avoid division by zero
        else:
            fitness_range = 1.0
            fitness_min = 0.0
            max_diversity = 1.0
        
        # Now sort and filter to quality pool
        candidates.sort(key=lambda x: x['fitness'])
        quality_pool = candidates[:internal_num_results]
        
        if self.verbose:
            print(f"\nQuality pool (top {len(quality_pool)} by fitness from {len(candidates)} total)")
            if quality_pool:
                pool_fitnesses = [c['fitness'] for c in quality_pool]
                distances = [c['distance'] for c in quality_pool]
                print(f"  Pool fitness range: {min(pool_fitnesses):.4f} to {max(pool_fitnesses):.4f}")
                print(f"  Full fitness range for normalization: {fitness_min:.4f} to {fitness_min + fitness_range:.4f}")
                print(f"  Distance range: {min(distances):.4f} to {max(distances):.4f}")
                print(f"  Max diversity for normalization: {max_diversity:.4f}")
                print(f"  Diversity lambda: {diversity_lambda} (lower = more proximity priority)")
        
        # Always pick the best fitness CF first (which is also typically closest)
        if quality_pool:
            best = quality_pool[0]
            final_counterfactuals.append(best['cf'])
            selected_arrays.append(best['array'])
            quality_pool.remove(best)
            
            if self.verbose:
                print(f"  [CF #1] Best fitness: {best['fitness']:.4f}, distance={best['distance']:.4f}")
        
        # Greedy diverse selection for remaining CFs
        while len(final_counterfactuals) < num_best_results and quality_pool:
            best_candidate = None
            best_score = float('-inf')
            
            for cand in quality_pool:
                # Skip near-duplicates
                is_dup = False
                for sel_arr in selected_arrays:
                    if np.allclose(cand['array'], sel_arr, atol=0.01):
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Calculate minimum distance to already selected CFs
                min_dist_to_selected = float('inf')
                for sel_arr in selected_arrays:
                    dist = np.linalg.norm(cand['array'] - sel_arr)
                    min_dist_to_selected = min(min_dist_to_selected, dist)
                
                # Normalize terms to [0, 1] range for fair comparison
                # Diversity: higher is better (normalize to 0-1)
                normalized_diversity = min(min_dist_to_selected / max_diversity, 1.0)
                
                # DIVERSITY AMPLIFICATION: Use sqrt to amplify small diversity differences
                # This makes the diversity term more impactful when candidates are similar
                # sqrt(0.1) = 0.316, sqrt(0.5) = 0.707, sqrt(1.0) = 1.0
                # This spreads out small values more, giving diversity more influence
                amplified_diversity = np.sqrt(normalized_diversity)
                
                # Fitness: lower is better, so normalize and invert (0=best, 1=worst)
                if fitness_range > 0:
                    normalized_fitness = (cand['fitness'] - fitness_min) / fitness_range
                else:
                    normalized_fitness = 0.0
                
                # MMR-style score: high diversity (amplified), low fitness (normalized)
                # Both terms now in [0, 1] range, diversity amplified for more impact
                score = diversity_lambda * amplified_diversity - (1 - diversity_lambda) * normalized_fitness
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate:
                final_counterfactuals.append(best_candidate['cf'])
                selected_arrays.append(best_candidate['array'])
                quality_pool.remove(best_candidate)
                
                if self.verbose:
                    min_div = float('inf')
                    for sel_arr in selected_arrays[:-1]:
                        d = np.linalg.norm(best_candidate['array'] - sel_arr)
                        min_div = min(min_div, d)
                    print(f"  [CF #{len(final_counterfactuals)}] Fitness={best_candidate['fitness']:.4f}, "
                          f"Distance={best_candidate['distance']:.4f}, MinDivFromSelected={min_div:.4f}, "
                          f"Score={best_score:.4f}")
            else:
                if self.verbose:
                    print(f"  No more diverse candidates available (remaining pool: {len(quality_pool)})")
                break
        
        # Log summary
        if self.verbose:
            print(f"\n=== CF Selection Summary ===")
            print(f"Total CFs: {len(final_counterfactuals)}/{num_best_results} requested")
            
            if final_counterfactuals:
                distances = []
                fitnesses = []
                diversities = []
                
                for i, cf in enumerate(final_counterfactuals):
                    cf_array = np.array([cf[f] for f in feature_names])
                    dist = np.linalg.norm(cf_array - original_features)
                    distances.append(dist)
                    
                    # Calculate fitness
                    fitness = calculate_fitness_func(
                        cf, original_features, sample, target_class, 
                        metric, population, original_class
                    )
                    fitnesses.append(fitness)
                    
                    # Calculate min diversity from other selected CFs
                    if i > 0:
                        min_div = float('inf')
                        for j, other_cf in enumerate(final_counterfactuals):
                            if i != j:
                                other_array = np.array([other_cf[f] for f in feature_names])
                                div = np.linalg.norm(cf_array - other_array)
                                min_div = min(min_div, div)
                        diversities.append(min_div)
                
                print(f"Distance to original: min={min(distances):.4f}, max={max(distances):.4f}, avg={np.mean(distances):.4f}")
                print(f"Fitness scores: min={min(fitnesses):.4f}, max={max(fitnesses):.4f}, avg={np.mean(fitnesses):.4f}")
                if diversities:
                    print(f"Min diversity between CFs: min={min(diversities):.4f}, max={max(diversities):.4f}, avg={np.mean(diversities):.4f}")
        
        # Build per-CF evolution histories (simplified - single entry)
        self.per_cf_evolution_histories = [list(self.evolution_history) for _ in final_counterfactuals]
        self.cf_generation_found = [0] * len(final_counterfactuals)  # All found in "generation 0"
        
        if not final_counterfactuals:
            return None
        
        return final_counterfactuals
