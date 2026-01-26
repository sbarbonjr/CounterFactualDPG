"""
GeneticAlgorithmRunner: Orchestrates the genetic algorithm for counterfactual generation.

Extracted from CounterFactualModel.py to handle DEAP setup, population initialization,
evolution loop, and result validation in a focused, testable class.
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools

from constants import INVALID_FITNESS


class DistanceBasedHOF:
    """
    Custom Hall of Fame that ranks individuals by distance to original sample,
    NOT by total fitness (which includes diversity/repulsion bonuses).
    
    This ensures the HOF contains the truly closest valid counterfactuals,
    preventing the drift-away problem where bonuses make far CFs appear "better".
    """
    
    def __init__(self, maxsize, original_features, feature_names, model, 
                 target_class, min_probability_margin=0.001, model_feature_names=None):
        """
        Args:
            maxsize: Maximum number of individuals to keep.
            original_features: numpy array of original sample feature values.
            feature_names: List of feature names (dict keys).
            model: ML model for prediction validation.
            target_class: Target class to validate against.
            min_probability_margin: Minimum margin for valid predictions.
            model_feature_names: Feature names expected by model (for DataFrame creation).
        """
        self.maxsize = maxsize
        self.original_features = original_features
        self.feature_names = feature_names
        self.model = model
        self.target_class = target_class
        self.min_probability_margin = min_probability_margin
        self.model_feature_names = model_feature_names
        self.items = []  # List of (distance, fitness, individual) tuples
    
    def update(self, population):
        """
        Update HOF with individuals from population, ranked by distance to original.
        Uses fitness as secondary criteria when distances are similar.
        Only includes individuals that predict target class with sufficient margin.
        """
        for ind in population:
            # Skip if invalid fitness
            if not ind.fitness.valid or ind.fitness.values[0] >= INVALID_FITNESS:
                continue
            
            # Calculate distance to original
            ind_array = np.array([ind[f] for f in self.feature_names])
            distance = np.linalg.norm(ind_array - self.original_features)
            
            # Get fitness for secondary ranking
            fitness = ind.fitness.values[0]
            
            # Validate prediction
            features = ind_array.reshape(1, -1)
            try:
                if self.model_feature_names is not None:
                    features_df = pd.DataFrame(features, columns=self.model_feature_names)
                    predicted_class = self.model.predict(features_df)[0]
                    proba = self.model.predict_proba(features_df)[0]
                else:
                    predicted_class = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                
                if predicted_class != self.target_class:
                    continue
                
                # Check probability margin
                if hasattr(self.model, "classes_"):
                    class_list = list(self.model.classes_)
                    target_idx = class_list.index(self.target_class) if self.target_class in class_list else self.target_class
                else:
                    target_idx = self.target_class
                
                target_prob = proba[target_idx]
                sorted_probs = np.sort(proba)[::-1]
                second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                margin = target_prob - second_best_prob
                
                if margin < self.min_probability_margin:
                    continue
                    
            except Exception:
                continue
            
            # Check if this individual is essentially a duplicate
            is_duplicate = False
            for existing_dist, existing_fitness, existing_ind in self.items:
                existing_array = np.array([existing_ind[f] for f in self.feature_names])
                if np.allclose(ind_array, existing_array, atol=0.01):
                    # Keep the closer one
                    if distance < existing_dist:
                        self.items.remove((existing_dist, existing_fitness, existing_ind))
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                # Clone the individual to preserve it
                ind_copy = creator.Individual(dict(ind))
                ind_copy.fitness = ind.fitness
                # Store (distance, fitness, individual) - fitness is tiebreaker
                self.items.append((distance, fitness, ind_copy))
                # Sort by distance first, then fitness as tiebreaker
                self.items.sort(key=lambda x: (x[0], x[1]))
                self.items = self.items[:self.maxsize * 2]  # Keep extra for diversity
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx][2]  # Return individual (index 2 in 3-tuple)
    
    def __iter__(self):
        return (item[2] for item in self.items)  # Return individual (index 2 in 3-tuple)


class GeneticAlgorithmRunner:
    """
    Runs genetic algorithm using DEAP framework for counterfactual generation.
    
    Enhanced with dual-boundary support: uses both original and target class constraints
    to guide evolution. Mutations escape original class bounds while approaching target bounds.
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
        Initialize the genetic algorithm runner.

        Args:
            model: Trained ML model with predict() and predict_proba() methods.
            constraints (dict): Feature constraints per class.
            feature_names (list): List of feature names (for DataFrame model input).
            verbose (bool): Whether to print progress messages.
            min_probability_margin (float): Minimum probability difference between
                target class and second-best class for valid counterfactuals.
            generation_debugging (bool): Enable detailed per-generation fitness tracking.
        """
        self.model = model
        self.constraints = constraints
        self.feature_names = feature_names
        self.verbose = verbose
        self.min_probability_margin = min_probability_margin
        self.generation_debugging = generation_debugging

        # Evolution tracking
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.evolution_history = []
        self.hof_evolution_histories = {}
        self.per_cf_evolution_histories = []
        self.generation_debug_table = []  # Per-generation fitness component breakdown
        self.best_minimal_cfs = []  # Track historically closest valid CFs
        self._original_features = None  # Stored for proximity-weighted selection

    def run(
        self,
        sample,
        target_class,
        original_class,
        population_size,
        generations,
        mutation_rate,
        metric,
        delta_threshold,
        patience,
        n_jobs,
        num_best_results,
        boundary_analysis,
        # Callbacks for CounterFactualModel methods
        create_individual_func,
        crossover_func,
        mutate_func,
        calculate_fitness_func,
        get_valid_sample_func,
        normalize_feature_func,
        features_match_func,
    ):
        """
        Run the genetic algorithm to find counterfactuals.

        Args:
            sample (dict): Original sample features.
            target_class (int): Target class for counterfactual.
            original_class (int): Original class for escape-aware mutation.
            population_size (int): Population size for GA.
            generations (int): Maximum generations to run.
            mutation_rate (float): Base mutation rate.
            metric (str): Distance metric for fitness calculation.
            delta_threshold (float): Convergence threshold.
            patience (int): Generations without improvement before early stopping.
            n_jobs (int): Number of parallel jobs (-1=all CPUs, 1=sequential).
            num_best_results (int): Number of top individuals to return.
            boundary_analysis (dict): Pre-computed boundary analysis between classes.
            create_individual_func: Function to create DEAP individual.
            crossover_func: Function for crossover operation.
            mutate_func: Function for mutation operation.
            calculate_fitness_func: Function to calculate fitness.
            get_valid_sample_func: Function to generate valid sample.
            normalize_feature_func: Function to normalize feature names.
            features_match_func: Function to check if features match.

        Returns:
            list: Valid counterfactuals (or None if none found).
        """
        feature_names = list(sample.keys())
        original_features = np.array([sample[feature] for feature in feature_names])
        
        # Store for proximity-weighted selection
        self._original_features = original_features

        # Log dual-boundary info
        if boundary_analysis and self.verbose:
            non_overlapping = boundary_analysis.get("non_overlapping", [])
            print(f"[Dual-Boundary] Non-overlapping features: {non_overlapping}")
            print(
                f"[Dual-Boundary] Escape directions: {boundary_analysis.get('escape_direction', {})}"
            )

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

        # Enable parallel processing
        pool = None
        if n_jobs != 1:
            from multiprocessing import Pool
            import os

            if n_jobs == -1:
                n_jobs = os.cpu_count()
            pool = Pool(processes=n_jobs)
            toolbox.register("map", pool.map)

        # Register genetic operators
        toolbox.register(
            "individual",
            create_individual_func,
            sample_dict=get_valid_sample_func(sample, target_class, original_class),
            feature_names=feature_names,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover_func, indpb=0.6, sample=sample)
        toolbox.register("select", self._select_diverse)
        toolbox.register(
            "mutate",
            mutate_func,
            sample=sample,
            feature_names=feature_names,
            mutation_rate=mutation_rate,
            target_class=target_class,
            original_class=original_class,
            boundary_analysis=boundary_analysis,
        )

        # Initialize population
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

        # Register evaluate operator (captures population in closure)
        toolbox.register(
            "evaluate",
            lambda ind: (
                calculate_fitness_func(
                    ind,
                    original_features,
                    sample,
                    target_class,
                    metric,
                    population,
                    original_class,
                ),
            ),
        )

        # Setup statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register(
            "avg",
            lambda x: np.nanmean(
                [
                    val[0]
                    for val in x
                    if not np.isinf(val[0]) and val[0] < INVALID_FITNESS
                ]
            )
            if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x)
            else np.nan,
        )
        stats.register(
            "min",
            lambda x: np.nanmin(
                [
                    val[0]
                    for val in x
                    if not np.isinf(val[0]) and val[0] < INVALID_FITNESS
                ]
            )
            if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x)
            else np.inf,
        )
        stats.register(
            "std",
            lambda x: np.nanstd(
                [
                    val[0]
                    for val in x
                    if not np.isinf(val[0]) and val[0] < INVALID_FITNESS
                ]
            )
            if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x)
            else np.nan,
        )

        # Setup hall of fame - use DistanceBasedHOF for minimal distance tracking
        # Standard HOF is still used internally for evolution, but DistanceBasedHOF
        # ensures final selection prioritizes closest valid CFs
        hof = tools.HallOfFame(num_best_results)
        distance_hof = DistanceBasedHOF(
            maxsize=num_best_results * 4,  # Keep extra candidates for diversity
            original_features=original_features,
            feature_names=feature_names,
            model=self.model,
            target_class=target_class,
            min_probability_margin=self.min_probability_margin,
            model_feature_names=self.feature_names,
        )

        # Reset tracking
        self.best_fitness_list = []
        self.average_fitness_list = []
        self.std_fitness_list = []
        self.evolution_history = []
        self.hof_evolution_histories = {i: [] for i in range(num_best_results)}
        self.generation_debug_table = []
        self.best_minimal_cfs = []  # Track historically closest valid CFs

        # Run evolution
        best_individuals = self._evolve(
            toolbox,
            population,
            stats,
            hof,
            generations,
            delta_threshold,
            patience,
            mutation_rate,
            mutate_func,
            sample,
            feature_names,
            target_class,
            original_class,
            boundary_analysis,
            population_size,
            num_best_results,
            normalize_feature_func,
            features_match_func,
            calculate_fitness_func,
            original_features,
            metric,
            distance_hof,  # Pass distance-based HOF
        )

        # Clean up multiprocessing pool
        if pool is not None:
            pool.close()
            pool.join()

        # Validate and return results - use distance_hof for final selection
        return self._validate_counterfactuals(
            best_individuals,
            distance_hof,  # Use distance-based HOF instead of standard HOF
            sample,
            target_class,
            num_best_results,
        )

    def _select_diverse(self, individuals, k):
        """
        Select k individuals balancing fitness quality, diversity, and proximity to original.
        
        Scoring components (lower is better):
        - fitness_val: raw fitness from GA
        - diversity_bonus: negative reward for being different from selected (encourages spread)
        - proximity_penalty: positive penalty for being far from original (encourages closeness)
        
        This prevents population from drifting too far from the original sample.
        """
        selected = []
        remaining = list(individuals)

        if not remaining:
            return selected

        # Always include the best individual first
        remaining_sorted = sorted(
            remaining, key=lambda x: x.fitness.values[0] if x.fitness.valid else 1e9
        )
        best = remaining_sorted[0]
        selected.append(best)
        remaining.remove(best)

        # Select remaining individuals balancing fitness, diversity, and proximity
        while len(selected) < k and remaining:
            best_candidate = None
            best_score = float("inf")

            for candidate in remaining:
                # Get fitness (lower is better)
                fitness_val = (
                    candidate.fitness.values[0] if candidate.fitness.valid else 1e9
                )

                # Calculate minimum distance to already selected individuals
                cand_array = np.array(
                    [candidate[key] for key in sorted(candidate.keys())]
                )
                min_dist = float("inf")
                for sel in selected:
                    sel_array = np.array([sel[key] for key in sorted(sel.keys())])
                    dist = np.linalg.norm(cand_array - sel_array)
                    min_dist = min(min_dist, dist)

                # Diversity bonus: reward being different from selected (negative = better)
                diversity_bonus = -0.2 * min_dist  # Reduced from 0.3 to balance with proximity
                
                # Proximity penalty: penalize being far from original (positive = worse)
                proximity_penalty = 0.0
                if self._original_features is not None:
                    dist_to_original = np.linalg.norm(cand_array - self._original_features)
                    proximity_penalty = 0.4 * dist_to_original  # 40% weight on proximity
                
                # Combined score (lower is better)
                score = fitness_val + diversity_bonus + proximity_penalty

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
        """
        # First individual is the original sample adjusted to constraint boundaries
        base_individual = get_valid_sample_func(sample, target_class, original_class)
        population = [create_individual_func(base_individual.copy(), feature_names)]

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

            population.append(create_individual_func(perturbed, feature_names))

        return population

    def _evolve(
        self,
        toolbox,
        population,
        stats,
        hof,
        generations,
        delta_threshold,
        patience,
        mutation_rate,
        mutate_func,
        sample,
        feature_names,
        target_class,
        original_class,
        boundary_analysis,
        population_size,
        num_best_results,
        normalize_feature_func,
        features_match_func,
        calculate_fitness_func,
        original_features,
        metric,
        distance_hof=None,
    ):
        """
        Run the evolution loop for the specified number of generations.
        
        Args:
            distance_hof: DistanceBasedHOF instance for tracking closest valid CFs by distance.
        """
        previous_hof_items = [None] * num_best_results
        previous_best_fitness = float("inf")
        stable_generations = 0
        current_mutation_rate = mutation_rate

        target_constraints = self.constraints.get(f"Class {target_class}", [])
        escape_directions = (
            boundary_analysis.get("escape_direction", {}) if boundary_analysis else {}
        )

        for generation in range(generations):
            # Evaluate population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Track best-minimal valid CFs (closest to original that predict target class)
            self._update_best_minimal_cfs(
                population, sample, target_class, original_features, num_best_results, generation=generation
            )

            # Update statistics and hall of fame
            record = stats.compile(population)
            hof.update(population)
            
            # Update distance-based HOF (ranks by distance, not total fitness with bonuses)
            # This ensures we track the truly closest valid CFs
            if distance_hof is not None:
                distance_hof.update(population)

            # Track evolution history
            self._update_evolution_history(hof, num_best_results, previous_hof_items)

            best_fitness = record["min"]
            average_fitness = record["avg"]
            std_fitness = record["std"]

            self.best_fitness_list.append(best_fitness)
            self.average_fitness_list.append(average_fitness)
            self.std_fitness_list.append(std_fitness)

            # Collect detailed fitness component breakdown for generation debugging
            if self.generation_debugging and len(hof) > 0:
                best_ind = hof[0]
                if best_ind.fitness.values[0] != np.inf and best_ind.fitness.values[0] < INVALID_FITNESS:
                    try:
                        # Call fitness function with return_components=True to get breakdown
                        _, components = calculate_fitness_func(
                            best_ind,
                            original_features,
                            sample,
                            target_class,
                            metric,
                            population,
                            original_class,
                            return_components=True
                        )
                        
                        # Build debug row with generation, features, and all fitness components
                        debug_row = {
                            'generation': generation + 1,
                            'feature_values': {k: float(v) for k, v in best_ind.items()},
                            # Fitness components
                            **components  # Unpack all component values
                        }
                        self.generation_debug_table.append(debug_row)
                    except Exception as e:
                        # If component extraction fails, just store basics
                        # Always print this warning since generation_debugging is explicitly enabled
                        import traceback
                        print(f"Warning: Failed to extract fitness components for generation {generation + 1}: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        debug_row = {
                            'generation': generation + 1,
                            'total_fitness': float(best_ind.fitness.values[0]),
                            'feature_values': {k: float(v) for k, v in best_ind.items()},
                        }
                        self.generation_debug_table.append(debug_row)

            # Check for convergence
            fitness_improvement = previous_best_fitness - best_fitness
            if fitness_improvement < delta_threshold:
                stable_generations += 1
            else:
                stable_generations = 0

            if self.verbose:
                print(
                    f"****** Generation {generation + 1}: Average Fitness = {average_fitness:.4f}, Best Fitness = {best_fitness:.4f}, fitness improvement = {fitness_improvement:.4f}"
                )

            previous_best_fitness = best_fitness

            # Early stopping
            if stable_generations >= patience:
                if self.verbose:
                    print(f"Convergence reached at generation {generation + 1}")
                break

            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            # Inject random immigrants (10% of population)
            offspring = self._inject_immigrants(
                offspring,
                population_size,
                sample,
                feature_names,
                target_constraints,
                escape_directions,
                normalize_feature_func,
                features_match_func,
            )

            # Adaptive mutation rate
            current_mutation_rate *= 0.99
            toolbox.unregister("mutate")
            toolbox.register(
                "mutate",
                mutate_func,
                sample=sample,
                feature_names=feature_names,
                mutation_rate=current_mutation_rate,
                target_class=target_class,
                original_class=original_class,
                boundary_analysis=boundary_analysis,
            )

            # Elitism: preserve top individuals
            elite_size = max(1, min(5, int(0.1 * population_size)))
            sorted_population = sorted(
                population, key=lambda ind: ind.fitness.values[0]
            )
            elites = sorted_population[:elite_size]

            # Replace population with offspring, keeping elites
            population[:] = offspring[:-elite_size] + elites

        return hof

    def _update_best_minimal_cfs(self, population, sample, target_class, original_features, num_best_results, generation=0):
        """
        Track the historically closest valid counterfactuals found during evolution.
        A valid CF must predict the target class with sufficient probability margin.
        
        This preserves good candidates from early generations that might be lost
        as the population spreads due to diversity pressure.
        
        AGGRESSIVE TRACKING: Keeps 4x the requested CFs to ensure we have enough
        for the 80% minimal allocation in final selection.
        
        Args:
            generation: Current generation number (0-indexed)
        """
        feature_names = list(sample.keys())
        
        # Keep 4x candidates to ensure enough for 80% fill (some may be filtered as duplicates)
        max_candidates = num_best_results * 4
        
        for ind in population:
            # Skip invalid fitness
            if not ind.fitness.valid or ind.fitness.values[0] >= INVALID_FITNESS:
                continue
            
            # Check if this individual predicts the target class
            features = np.array([ind[f] for f in feature_names]).reshape(1, -1)
            try:
                if self.feature_names is not None:
                    features_df = pd.DataFrame(features, columns=self.feature_names)
                    predicted_class = self.model.predict(features_df)[0]
                    proba = self.model.predict_proba(features_df)[0]
                else:
                    predicted_class = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                
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
            
            # Calculate Euclidean distance to original (this is the TRUE base fitness metric)
            ind_array = np.array([ind[f] for f in feature_names])
            distance_to_original = np.linalg.norm(ind_array - original_features)
            
            # Create candidate entry - track by DISTANCE, not GA fitness (which includes bonuses)
            candidate = {
                'individual': dict(ind),
                'distance': distance_to_original,
                'ga_fitness': ind.fitness.values[0],  # Keep for reference
                'generation_found': generation,  # Track when this CF was found
            }
            
            # Check if this should be added to best_minimal_cfs
            # Keep top candidates by distance (not GA fitness)
            dominated = False
            to_remove = []
            
            for i, existing in enumerate(self.best_minimal_cfs):
                # Check if candidate is essentially the same individual
                existing_array = np.array([existing['individual'][f] for f in feature_names])
                if np.allclose(ind_array, existing_array, atol=0.01):
                    # Same individual - keep the one with better distance
                    if distance_to_original < existing['distance']:
                        to_remove.append(i)
                    else:
                        dominated = True
                    break
            
            # Remove dominated entries
            for i in reversed(to_remove):
                self.best_minimal_cfs.pop(i)
            
            if not dominated:
                self.best_minimal_cfs.append(candidate)
                # Sort by distance and keep only top candidates
                self.best_minimal_cfs.sort(key=lambda x: x['distance'])
                self.best_minimal_cfs = self.best_minimal_cfs[:max_candidates]

    def _update_evolution_history(self, hof, num_best_results, previous_hof_items):
        """
        Track evolution history for visualization.
        """
        # Store best individual for this generation
        if hof[0].fitness.values[0] != np.inf:
            entry = dict(hof[0])
            entry["_fitness"] = hof[0].fitness.values[0]
            self.evolution_history.append(entry)

        # Track evolution history per HOF entry
        for hof_idx in range(min(len(hof), num_best_results)):
            current_hof_item = hof[hof_idx]
            current_hof_dict = dict(current_hof_item)
            current_fitness = current_hof_item.fitness.values[0]

            # Skip invalid entries
            if current_fitness == np.inf or current_fitness >= INVALID_FITNESS:
                continue

            # Check if this HOF position has a new individual
            prev_item = previous_hof_items[hof_idx]
            is_new_entry = (
                prev_item is None
                or dict(prev_item) != current_hof_dict
                or prev_item.fitness.values[0] != current_fitness
            )

            if is_new_entry:
                # Copy shared evolution history as lineage
                if len(self.evolution_history) > 1:
                    self.hof_evolution_histories[hof_idx] = list(
                        self.evolution_history[:-1]
                    )
                else:
                    self.hof_evolution_histories[hof_idx] = []

                # Store reference to track changes
                from deap import base
                previous_hof_items[hof_idx] = base.Toolbox().clone(current_hof_item)

    def _inject_immigrants(
        self,
        offspring,
        population_size,
        sample,
        feature_names,
        target_constraints,
        escape_directions,
        normalize_feature_func,
        features_match_func,
    ):
        """
        Inject random immigrants to maintain genetic diversity.
        """
        num_immigrants = max(1, int(0.1 * population_size))

        for i in range(num_immigrants):
            immigrant = {}

            for feature in feature_names:
                norm_feature = normalize_feature_func(feature)
                escape_dir = escape_directions.get(norm_feature, "both")

                # Find constraint for this feature
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

                    # Generate random value biased by escape direction
                    if feature_min is not None and feature_max is not None:
                        if escape_dir == "increase":
                            mid = (feature_min + feature_max) / 2
                            immigrant[feature] = np.random.uniform(mid, feature_max)
                        elif escape_dir == "decrease":
                            mid = (feature_min + feature_max) / 2
                            immigrant[feature] = np.random.uniform(feature_min, mid)
                        else:
                            immigrant[feature] = np.random.uniform(
                                feature_min, feature_max
                            )
                    elif feature_min is not None:
                        immigrant[feature] = np.random.uniform(
                            feature_min, feature_min + 2.0
                        )
                    elif feature_max is not None:
                        immigrant[feature] = np.random.uniform(
                            max(0, feature_max - 2.0), feature_max
                        )
                    else:
                        immigrant[feature] = sample[feature] + np.random.uniform(
                            -1.0, 1.0
                        )
                else:
                    immigrant[feature] = sample[feature] + np.random.uniform(
                        -1.0, 1.0
                    )

                # Ensure non-negative and round
                immigrant[feature] = np.round(max(0, immigrant[feature]), 2)

            # Replace one of the worst offspring
            immigrant_ind = creator.Individual(immigrant)
            offspring[-(i + 1)] = immigrant_ind

        return offspring

    def _validate_counterfactuals(
        self, hof, hof_obj, sample, target_class, num_best_results
    ):
        """
        Validate counterfactuals and build evolution histories.
        
        Uses GREEDY DIVERSE SELECTION to balance:
        - Proximity: CFs should be close to original sample
        - Diversity: CFs should be different from each other
        
        Algorithm:
        1. Always pick the closest valid CF first
        2. For subsequent CFs, use MMR-style scoring:
           score = diversity_weight * min_dist_to_selected - proximity_weight * dist_to_original
        
        This ensures we get the closest CF plus diverse alternatives.
        """
        feature_names = list(sample.keys())
        original_features = self._original_features
        
        # Tunable parameter: how much to weight diversity vs proximity
        # Higher diversity_lambda = more spread out CFs
        # Lower diversity_lambda = CFs closer to original but more similar
        diversity_lambda = 0.6  # 60% diversity, 40% proximity for subsequent selections
        
        if self.verbose:
            print("\n=== CF Selection (Greedy Diverse) ===")
            print(f"Requested: {num_best_results} CFs")
            print(f"Available in DistanceBasedHOF: {len(hof_obj)}")
            print(f"Diversity lambda: {diversity_lambda}")
        
        # Build candidate pool from DistanceBasedHOF
        candidates = []
        for distance, fitness, ind in hof_obj.items:
            cf_dict = dict(ind)
            cf_array = np.array([cf_dict[f] for f in feature_names])
            candidates.append({
                'cf': cf_dict,
                'array': cf_array,
                'distance': distance,
                'fitness': fitness,
            })
        
        # Also add from best_minimal_cfs as backup
        for minimal_cf in self.best_minimal_cfs:
            cf_dict = minimal_cf['individual']
            cf_array = np.array([cf_dict[f] for f in feature_names])
            
            # Check if already in candidates (avoid duplicates)
            is_dup = False
            for cand in candidates:
                if np.allclose(cf_array, cand['array'], atol=0.01):
                    is_dup = True
                    break
            if not is_dup:
                candidates.append({
                    'cf': cf_dict,
                    'array': cf_array,
                    'distance': minimal_cf['distance'],
                    'generation_found': minimal_cf.get('generation_found', None),
                })
        
        if self.verbose:
            print(f"Total candidate pool: {len(candidates)}")
        
        final_counterfactuals = []
        selected_arrays = []
        
        # STEP 1: Always pick the closest CF first
        if candidates:
            # Sort by distance and pick closest
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
                # Skip near-duplicates of already selected
                is_dup = False
                for sel_arr in selected_arrays:
                    if np.allclose(cand['array'], sel_arr, atol=0.01):
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Calculate minimum distance to already selected CFs (diversity)
                min_dist_to_selected = float('inf')
                for sel_arr in selected_arrays:
                    dist = np.linalg.norm(cand['array'] - sel_arr)
                    min_dist_to_selected = min(min_dist_to_selected, dist)
                
                # MMR-style score: balance diversity and proximity
                # Normalize both terms to similar scales
                # diversity term: higher is better (more different from selected)
                # proximity term: lower distance is better (closer to original)
                diversity_term = min_dist_to_selected
                proximity_term = cand['distance']
                
                # Score: maximize diversity, minimize distance to original
                score = diversity_lambda * diversity_term - (1 - diversity_lambda) * proximity_term
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate:
                final_counterfactuals.append(best_candidate['cf'])
                selected_arrays.append(best_candidate['array'])
                candidates.remove(best_candidate)
                
                if self.verbose:
                    # Calculate diversity for logging
                    min_div = float('inf')
                    for sel_arr in selected_arrays[:-1]:
                        d = np.linalg.norm(best_candidate['array'] - sel_arr)
                        min_div = min(min_div, d)
                    print(f"  [CF #{len(final_counterfactuals)}] Distance={best_candidate['distance']:.4f}, "
                          f"MinDivFromSelected={min_div:.4f}, Score={best_score:.4f}")
            else:
                # No more valid candidates
                break
        
        # Log final summary
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
                
                # Calculate pairwise diversity
                if len(final_counterfactuals) > 1:
                    diversities = []
                    arrays = [np.array([cf[f] for f in feature_names]) for cf in final_counterfactuals]
                    for i in range(len(arrays)):
                        for j in range(i + 1, len(arrays)):
                            diversities.append(np.linalg.norm(arrays[i] - arrays[j]))
                    print(f"Pairwise diversity: min={min(diversities):.4f}, max={max(diversities):.4f}, avg={np.mean(diversities):.4f}")
        
        # Build per-CF evolution histories (simplified)
        self.per_cf_evolution_histories = [list(self.evolution_history) for _ in final_counterfactuals]
        
        # Build list of generation_found values corresponding to final_counterfactuals
        # Match generation info from original candidates (before they were removed)
        self.cf_generation_found = []
        for cf_dict in final_counterfactuals:
            cf_array = np.array([cf_dict[f] for f in feature_names])
            # Find matching candidate in original pool to get generation_found
            gen_found = None
            
            # Search in best_minimal_cfs directly (most reliable source)
            for minimal_cf in self.best_minimal_cfs:
                minimal_array = np.array([minimal_cf['individual'][f] for f in feature_names])
                if np.allclose(cf_array, minimal_array, atol=0.01):
                    gen_found = minimal_cf.get('generation_found')
                    break
            
            self.cf_generation_found.append(gen_found)
        
        # Return None if no valid counterfactuals found
        if not final_counterfactuals:
            return None
        
        return final_counterfactuals
