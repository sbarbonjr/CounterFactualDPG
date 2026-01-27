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
    Hall of Fame that ranks individuals by distance to original sample.
    
    Unlike GA's HOF which uses fitness (with diversity/repulsion bonuses),
    this ranks purely by Euclidean distance to the original sample.
    This prevents the drift-away problem where bonuses cause CFs to move far from original.
    """
    
    def __init__(self, maxsize, original_features, feature_names):
        self.maxsize = maxsize
        self.original_features = original_features
        self.feature_names = feature_names
        self.items = []
    
    def update(self, population):
        """Update HOF with individuals from population, ranked by distance to original."""
        for ind in population:
            # Skip invalid fitness individuals
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                continue
            if ind.fitness.values[0] >= INVALID_FITNESS:
                continue
            
            # Calculate distance to original
            ind_array = np.array([ind[f] for f in self.feature_names])
            distance = np.linalg.norm(ind_array - self.original_features)
            
            # Check for duplicates
            is_dup = False
            for existing_dist, existing_ind in self.items:
                existing_array = np.array([existing_ind[f] for f in self.feature_names])
                if np.allclose(ind_array, existing_array, atol=0.01):
                    is_dup = True
                    break
            
            if not is_dup:
                self.items.append((distance, dict(ind)))
                # Sort by distance and keep top maxsize
                self.items.sort(key=lambda x: x[0])
                self.items = self.items[:self.maxsize]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx][1]
    
    def __iter__(self):
        return iter([item[1] for item in self.items])


class GeneticAlgorithmRunner:
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
        feature_names,
        verbose=False,
        min_probability_margin=0.001,
        generation_debugging=False,
    ):
        """
        Initialize the heuristic runner.

        Args:
            model: Trained ML model with predict() and predict_proba() methods.
            constraints (dict): Feature constraints per class.
            feature_names (list): List of feature names (for DataFrame model input).
            verbose (bool): Whether to print progress messages.
            min_probability_margin (float): Minimum probability difference between
                target class and second-best class for valid counterfactuals.
            generation_debugging (bool): Unused, kept for API compatibility.
        """
        self.model = model
        self.constraints = constraints
        self.feature_names = feature_names
        self.verbose = verbose
        self.min_probability_margin = min_probability_margin
        self.generation_debugging = generation_debugging

        # Tracking attributes (kept for API compatibility, simplified)
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.std_fitness_list = []
        self.evolution_history = []
        self.hof_evolution_histories = {}
        self.per_cf_evolution_histories = []
        self.generation_debug_table = []
        self.cf_generation_found = []
        self._original_features = None

    def run(
        self,
        sample,
        target_class,
        original_class,
        population_size,
        generations,  # Unused, kept for API compatibility
        mutation_rate,  # Unused, kept for API compatibility
        metric,
        delta_threshold,  # Unused, kept for API compatibility
        patience,  # Unused, kept for API compatibility
        n_jobs,  # Unused, kept for API compatibility
        num_best_results,
        boundary_analysis,
        # Callbacks for CounterFactualModel methods
        create_individual_func,
        crossover_func,  # Unused, kept for API compatibility
        mutate_func,
        calculate_fitness_func,
        get_valid_sample_func,
        normalize_feature_func,
        features_match_func,
    ):
        """
        Generate counterfactual candidates using heuristic approach.

        Args:
            sample (dict): Original sample features.
            target_class (int): Target class for counterfactual.
            original_class (int): Original class for escape-aware generation.
            population_size (int): Number of candidates to generate.
            generations: Unused, kept for API compatibility.
            mutation_rate: Unused, kept for API compatibility.
            metric (str): Distance metric for fitness calculation.
            delta_threshold: Unused, kept for API compatibility.
            patience: Unused, kept for API compatibility.
            n_jobs: Unused, kept for API compatibility.
            num_best_results (int): Number of top individuals to return.
            boundary_analysis (dict): Pre-computed boundary analysis between classes.
            create_individual_func: Function to create individual dict.
            crossover_func: Unused, kept for API compatibility.
            mutate_func: Function for perturbation.
            calculate_fitness_func: Function to calculate fitness.
            get_valid_sample_func: Function to generate valid sample.
            normalize_feature_func: Function to normalize feature names.
            features_match_func: Function to check if features match.

        Returns:
            list: Valid counterfactuals (or None if none found).
        """
        feature_names = list(sample.keys())
        original_features = np.array([sample[feature] for feature in feature_names])
        
        # Store for selection
        self._original_features = original_features

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
        )

        # Reset tracking
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.std_fitness_list = []
        self.evolution_history = []
        self.hof_evolution_histories = {}
        self.per_cf_evolution_histories = []
        self.generation_debug_table = []
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

        # Create HOF using distance-based ranking
        hof = DistanceBasedHOF(num_best_results * 4, original_features, feature_names)
        hof.update(population)

        # Validate and return results
        return self._validate_counterfactuals(
            hof,
            sample,
            target_class,
            num_best_results,
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
    ):
        """
        Create initial population with escape-aware perturbations.
        
        Returns list of Individual objects that can hold fitness.
        """
        # First individual is the original sample adjusted to constraint boundaries
        base_individual = get_valid_sample_func(sample, target_class, original_class)
        population = [Individual(create_individual_func(base_individual.copy(), feature_names))]

        # Get constraints and escape directions
        target_constraints = self.constraints.get(f"Class {target_class}", [])
        escape_directions = (
            boundary_analysis.get("escape_direction", {}) if boundary_analysis else {}
        )

        # Create remaining individuals with escape-aware perturbations
        for _ in range(population_size - 1):
            perturbed = sample.copy()

            for feature in feature_names:
                norm_feature = normalize_feature_func(feature)
                escape_dir = escape_directions.get(norm_feature, "both")

                # Bias perturbation by escape direction
                if escape_dir == "increase":
                    perturbation = np.random.uniform(0, 0.4)
                elif escape_dir == "decrease":
                    perturbation = np.random.uniform(-0.4, 0)
                else:
                    perturbation = np.random.uniform(-0.2, 0.2)

                perturbed[feature] = sample[feature] + perturbation

                # Clip to target constraint boundaries
                matching_constraint = next(
                    (
                        c
                        for c in target_constraints
                        if features_match_func(c.get("feature", ""), feature)
                    ),
                    None,
                )
                if matching_constraint:
                    feature_min = matching_constraint.get("min")
                    feature_max = matching_constraint.get("max")
                    if feature_min is not None:
                        perturbed[feature] = max(feature_min, perturbed[feature])
                    if feature_max is not None:
                        perturbed[feature] = min(feature_max, perturbed[feature])

                # Ensure non-negative and round
                perturbed[feature] = np.round(max(0, perturbed[feature]), 2)

            population.append(Individual(create_individual_func(perturbed, feature_names)))

        return population

    def _validate_counterfactuals(
        self, hof, sample, target_class, num_best_results
    ):
        """
        Validate counterfactuals and build evolution histories.
        
        Uses GREEDY DIVERSE SELECTION to balance:
        - Proximity: CFs should be close to original sample
        - Diversity: CFs should be different from each other
        """
        feature_names = list(sample.keys())
        original_features = self._original_features
        
        # Tunable parameter for diversity vs proximity
        diversity_lambda = 0.6
        
        if self.verbose:
            print("\n=== CF Selection (Greedy Diverse) ===")
            print(f"Requested: {num_best_results} CFs")
            print(f"Available in HOF: {len(hof)}")
        
        # Build candidate pool from HOF
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
                    continue
                    
            except Exception:
                continue
            
            candidates.append({
                'cf': cf_dict,
                'array': cf_array,
                'distance': distance,
            })
        
        if self.verbose:
            print(f"Valid candidates: {len(candidates)}")
        
        final_counterfactuals = []
        selected_arrays = []
        
        # STEP 1: Always pick the closest CF first
        if candidates:
            candidates.sort(key=lambda x: x['distance'])
            closest = candidates[0]
            final_counterfactuals.append(closest['cf'])
            selected_arrays.append(closest['array'])
            candidates.remove(closest)
            
            if self.verbose:
                print(f"  [CF #1] Closest: distance={closest['distance']:.4f}")
        
        # STEP 2: Greedy diverse selection for remaining CFs
        while len(final_counterfactuals) < num_best_results and candidates:
            best_candidate = None
            best_score = float('-inf')
            
            for cand in candidates:
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
                
                # MMR-style score
                diversity_term = min_dist_to_selected
                proximity_term = cand['distance']
                score = diversity_lambda * diversity_term - (1 - diversity_lambda) * proximity_term
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate:
                final_counterfactuals.append(best_candidate['cf'])
                selected_arrays.append(best_candidate['array'])
                candidates.remove(best_candidate)
                
                if self.verbose:
                    min_div = float('inf')
                    for sel_arr in selected_arrays[:-1]:
                        d = np.linalg.norm(best_candidate['array'] - sel_arr)
                        min_div = min(min_div, d)
                    print(f"  [CF #{len(final_counterfactuals)}] Distance={best_candidate['distance']:.4f}, "
                          f"MinDivFromSelected={min_div:.4f}")
            else:
                break
        
        # Log summary
        if self.verbose:
            print(f"\n=== CF Selection Summary ===")
            print(f"Total CFs: {len(final_counterfactuals)}/{num_best_results} requested")
            
            if final_counterfactuals:
                distances = []
                for cf in final_counterfactuals:
                    cf_array = np.array([cf[f] for f in feature_names])
                    dist = np.linalg.norm(cf_array - original_features)
                    distances.append(dist)
                print(f"Distance to original: min={min(distances):.4f}, max={max(distances):.4f}, avg={np.mean(distances):.4f}")
        
        # Build per-CF evolution histories (simplified - single entry)
        self.per_cf_evolution_histories = [list(self.evolution_history) for _ in final_counterfactuals]
        self.cf_generation_found = [0] * len(final_counterfactuals)  # All found in "generation 0"
        
        if not final_counterfactuals:
            return None
        
        return final_counterfactuals
