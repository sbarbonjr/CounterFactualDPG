import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean, cityblock, cosine

from deap import base, creator, tools

from boundary_analyzer import BoundaryAnalyzer
from constraint_validator import ConstraintValidator
from fitness_calculator import FitnessCalculator
from mutation_strategy import MutationStrategy
from sample_generator import SampleGenerator
from constants import INVALID_FITNESS


class CounterFactualModel:
    def __init__(
        self,
        model,
        constraints,
        dict_non_actionable=None,
        verbose=False,
        diversity_weight=0.5,
        repulsion_weight=4.0,
        boundary_weight=15.0,
        distance_factor=2.0,
        sparsity_factor=1.0,
        constraints_factor=3.0,
        original_escape_weight=2.0,
        escape_pressure=0.5,
        prioritize_non_overlapping=True,
        max_bonus_cap=50.0,
        X_train=None,
        y_train=None,
        min_probability_margin=0.001,
    ):
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
        self.dict_non_actionable = (
            dict_non_actionable  # non_decreasing, non_increasing, no_change
        )
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
        self.feature_names = getattr(model, "feature_names_in_", None)
        # Initialize BoundaryAnalyzer for constraint boundary analysis
        self.boundary_analyzer = BoundaryAnalyzer(constraints=constraints, verbose=verbose)
        # Initialize ConstraintValidator for constraint validation
        self.constraint_validator = ConstraintValidator(
            model=model,
            constraints=constraints,
            dict_non_actionable=dict_non_actionable,
            feature_names=self.feature_names,
            boundary_analyzer=self.boundary_analyzer,
        )
        # Initialize FitnessCalculator for fitness calculation
        self.fitness_calculator = FitnessCalculator(
            model=model,
            feature_names=self.feature_names,
            diversity_weight=diversity_weight,
            repulsion_weight=repulsion_weight,
            boundary_weight=boundary_weight,
            distance_factor=distance_factor,
            sparsity_factor=sparsity_factor,
            constraints_factor=constraints_factor,
            original_escape_weight=original_escape_weight,
            max_bonus_cap=max_bonus_cap,
            constraint_validator=self.constraint_validator,
            boundary_analyzer=self.boundary_analyzer,
        )
        # Initialize MutationStrategy for mutation and crossover operations
        self.mutation_strategy = MutationStrategy(
            constraints=constraints,
            dict_non_actionable=dict_non_actionable,
            escape_pressure=escape_pressure,
            prioritize_non_overlapping=prioritize_non_overlapping,
            boundary_analyzer=self.boundary_analyzer,
        )
        # Initialize SampleGenerator for sample generation
        self.sample_generator = SampleGenerator(
            model=model,
            constraints=constraints,
            dict_non_actionable=dict_non_actionable,
            feature_names=self.feature_names,
            escape_pressure=escape_pressure,
            X_train=X_train,
            y_train=y_train,
            min_probability_margin=min_probability_margin,
            verbose=verbose,
            boundary_analyzer=self.boundary_analyzer,
            constraint_validator=self.constraint_validator,
        )
        # Store training data for nearest neighbor fallback
        self.X_train = X_train
        self.y_train = y_train
        # Minimum probability margin for accepting counterfactuals
        self.min_probability_margin = min_probability_margin

    def _analyze_boundary_overlap(self, original_class, target_class):
        """
        Delegate to BoundaryAnalyzer for boundary overlap analysis.
        """
        return self.boundary_analyzer.analyze_boundary_overlap(original_class, target_class)

    def _calculate_original_escape_penalty(
        self, individual, sample, original_class, target_class=None
    ):
        """
        Delegate to BoundaryAnalyzer for escape penalty calculation.
        """
        return self.boundary_analyzer.calculate_original_escape_penalty(
            individual, sample, original_class, target_class
        )

    def is_actionable_change(self, counterfactual_sample, original_sample):
        """
        Delegate to ConstraintValidator for actionability check.
        """
        return self.constraint_validator.is_actionable_change(
            counterfactual_sample, original_sample
        )

    def check_validity(self, counterfactual_sample, original_sample, desired_class):
        """
        Delegate to ConstraintValidator for validity check.
        """
        return self.constraint_validator.check_validity(
            counterfactual_sample, original_sample, desired_class
        )

    def calculate_distance(
        self, original_sample, counterfactual_sample, metric="euclidean"
    ):
        """
        Delegate to FitnessCalculator for distance calculation.
        """
        return self.fitness_calculator.calculate_distance(
            original_sample, counterfactual_sample, metric
        )

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
        feature = re.sub(r"\s*\([^)]*\)", "", feature)
        # Replace underscores with spaces
        feature = feature.replace("_", " ")
        # Normalize multiple spaces to single space
        feature = re.sub(r"\s+", " ", feature)
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
        return self._normalize_feature_name(feature1) == self._normalize_feature_name(
            feature2
        )

    def validate_constraints(
        self, S_prime, sample, target_class, original_class=None, strict_mode=True
    ):
        """
        Delegate to ConstraintValidator for constraint validation.
        """
        return self.constraint_validator.validate_constraints(
            S_prime, sample, target_class, original_class, strict_mode
        )

    def get_valid_sample(self, sample, target_class, original_class=None):
        """
        Delegate to SampleGenerator for valid sample generation.
        """
        return self.sample_generator.get_valid_sample(sample, target_class, original_class)

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        """
        Delegate to FitnessCalculator for sparsity calculation.
        """
        return self.fitness_calculator.calculate_sparsity(
            original_sample, counterfactual_sample
        )

    def individual_diversity(self, individual, population):
        """
        Delegate to FitnessCalculator for diversity calculation.
        """
        return self.fitness_calculator.individual_diversity(individual, population)

    def min_distance_to_others(self, individual, population):
        """
        Delegate to FitnessCalculator for minimum distance calculation.
        """
        return self.fitness_calculator.min_distance_to_others(individual, population)

    def distance_to_boundary_line(self, individual, target_class):
        """
        Delegate to FitnessCalculator for boundary distance calculation.
        """
        return self.fitness_calculator.distance_to_boundary_line(
            individual, target_class
        )

    def calculate_fitness(
        self,
        individual,
        original_features,
        sample,
        target_class,
        metric="cosine",
        population=None,
        original_class=None,
    ):
        """
        Delegate to FitnessCalculator for fitness calculation.
        """
        return self.fitness_calculator.calculate_fitness(
            individual,
            original_features,
            sample,
            target_class,
            metric,
            population,
            original_class,
        )

    def _create_deap_individual(self, sample_dict, feature_names):
        """Delegate to MutationStrategy for individual creation."""
        return self.mutation_strategy.create_deap_individual(sample_dict, feature_names)

    def _mutate_individual(
        self,
        individual,
        sample,
        feature_names,
        mutation_rate,
        target_class=None,
        original_class=None,
        boundary_analysis=None,
    ):
        """Delegate to MutationStrategy for mutation."""
        return self.mutation_strategy.mutate_individual(
            individual,
            sample,
            feature_names,
            mutation_rate,
            target_class,
            original_class,
            boundary_analysis,
        )

    def _dual_boundary_mutate(
        self,
        current_value,
        target_min,
        target_max,
        orig_min,
        orig_max,
        escape_dir="both",
    ):
        """Delegate to MutationStrategy for dual boundary mutation."""
        return self.mutation_strategy._dual_boundary_mutate(
            current_value, target_min, target_max, orig_min, orig_max, escape_dir
        )

    def _crossover_dict(self, ind1, ind2, indpb, sample=None):
        """Delegate to MutationStrategy for crossover."""
        return self.mutation_strategy.crossover_dict(ind1, ind2, indpb, sample)

    def genetic_algorithm(
        self,
        sample,
        target_class,
        population_size=100,
        generations=100,
        mutation_rate=0.8,
        metric="euclidean",
        delta_threshold=0.01,
        patience=10,
        n_jobs=-1,
        original_class=None,
        num_best_results=1,
    ):
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
            boundary_analysis = self._analyze_boundary_overlap(
                original_class, target_class
            )
            if self.verbose:
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

        # Enable parallel processing (default behavior with n_jobs=-1)
        if n_jobs != 1:
            from multiprocessing import Pool
            import os

            if n_jobs == -1:
                n_jobs = os.cpu_count()
            pool = Pool(processes=n_jobs)
            toolbox.register("map", pool.map)

        # Register individual creation (now with original_class for escape-aware initialization)
        toolbox.register(
            "individual",
            self._create_deap_individual,
            sample_dict=self.get_valid_sample(sample, target_class, original_class),
            feature_names=feature_names,
        )

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
            remaining_sorted = sorted(
                remaining, key=lambda x: x.fitness.values[0] if x.fitness.valid else 1e9
            )
            best = remaining_sorted[0]
            selected.append(best)
            remaining.remove(best)

            # Select remaining individuals balancing fitness and diversity
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

                    # Score combines fitness and diversity (lower is better)
                    # Give 30% weight to diversity bonus
                    diversity_bonus = (
                        -0.3 * min_dist
                    )  # Negative because we want to minimize
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
        toolbox.register(
            "mutate",
            self._mutate_individual,
            sample=sample,
            feature_names=feature_names,
            mutation_rate=mutation_rate,
            target_class=target_class,
            original_class=original_class,
            boundary_analysis=boundary_analysis,
        )

        # Create initial population starting near the original sample
        # First individual is the original sample adjusted to constraint boundaries (escape-aware)
        base_individual = self.get_valid_sample(sample, target_class, original_class)
        population = [
            self._create_deap_individual(base_individual.copy(), feature_names)
        ]

        # Remaining individuals are perturbations biased by escape direction
        target_constraints = self.constraints.get(f"Class {target_class}", [])
        original_constraints = (
            self.constraints.get(f"Class {original_class}", [])
            if original_class
            else []
        )
        escape_directions = (
            boundary_analysis.get("escape_direction", {}) if boundary_analysis else {}
        )

        for _ in range(population_size - 1):
            perturbed = sample.copy()
            # Add perturbations biased by escape direction for each feature
            for feature in feature_names:
                norm_feature = self._normalize_feature_name(feature)
                escape_dir = escape_directions.get(norm_feature, "both")

                # Base perturbation
                if escape_dir == "increase":
                    perturbation = np.random.uniform(0, 0.4)  # Bias toward increase
                elif escape_dir == "decrease":
                    perturbation = np.random.uniform(-0.4, 0)  # Bias toward decrease
                else:
                    perturbation = np.random.uniform(-0.2, 0.2)  # Symmetric

                perturbed[feature] = sample[feature] + perturbation

                # Clip to target constraint boundaries if they exist
                matching_constraint = next(
                    (
                        c
                        for c in target_constraints
                        if self._features_match(c.get("feature", ""), feature)
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

            population.append(self._create_deap_individual(perturbed, feature_names))

        # Register evaluate operator after population creation so it can capture population in closure
        # Now includes original_class for escape penalty calculation
        toolbox.register(
            "evaluate",
            lambda ind: (
                self.calculate_fitness(
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
        # Use INVALID_FITNESS threshold for filtering statistics
        # (imported from constants module)
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

        # Setup hall of fame to keep best individuals
        hof = tools.HallOfFame(num_best_results)

        self.best_fitness_list = []
        self.average_fitness_list = []
        self.evolution_history = []  # Reset evolution history for this run (best individual per gen)

        # Track evolution history per HOF entry for visualization
        # Each HOF position gets its own history: snapshots of best individual at each generation
        # until that HOF entry was discovered/established
        hof_evolution_histories = {i: [] for i in range(num_best_results)}
        previous_hof_items = [
            None
        ] * num_best_results  # Track previous HOF entries to detect changes

        previous_best_fitness = float("inf")
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
            # This is the shared evolution_history (tracking hof[0])
            if hof[0].fitness.values[0] != np.inf:
                entry = dict(hof[0])
                entry["_fitness"] = hof[0].fitness.values[
                    0
                ]  # Store fitness for visualization
                self.evolution_history.append(entry)

            # Track evolution history per HOF entry
            # For each HOF position, append the current best individual's state
            # This creates a history path that each CF can use for visualization
            for hof_idx in range(min(len(hof), num_best_results)):
                current_hof_item = hof[hof_idx]
                current_hof_dict = dict(current_hof_item)
                current_fitness = current_hof_item.fitness.values[0]

                # Skip invalid entries
                if current_fitness == np.inf or current_fitness >= INVALID_FITNESS:
                    continue

                # Check if this HOF position has a new/different individual
                prev_item = previous_hof_items[hof_idx]
                is_new_entry = (
                    prev_item is None
                    or dict(prev_item) != current_hof_dict
                    or prev_item.fitness.values[0] != current_fitness
                )

                if is_new_entry:
                    # New individual entered this HOF position
                    # Copy the current shared evolution_history as its lineage up to this point
                    # (minus the last entry since that's the current generation)
                    if len(self.evolution_history) > 1:
                        hof_evolution_histories[hof_idx] = list(
                            self.evolution_history[:-1]
                        )
                    else:
                        hof_evolution_histories[hof_idx] = []

                    # Store reference to track changes
                    previous_hof_items[hof_idx] = toolbox.clone(current_hof_item)

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
                print(
                    f"****** Generation {generation + 1}: Average Fitness = {average_fitness:.4f}, Best Fitness = {best_fitness:.4f}, fitness improvement = {fitness_improvement:.4f}"
                )

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
                    escape_dir = escape_directions.get(norm_feature, "both")

                    # Find constraint for this feature
                    matching_constraint = next(
                        (
                            c
                            for c in target_constraints
                            if self._features_match(c.get("feature", ""), feature)
                        ),
                        None,
                    )

                    if matching_constraint:
                        feature_min = matching_constraint.get("min")
                        feature_max = matching_constraint.get("max")

                        # Generate random value within constraints, biased by escape direction
                        if feature_min is not None and feature_max is not None:
                            if escape_dir == "increase":
                                # Bias toward upper half of target range
                                mid = (feature_min + feature_max) / 2
                                immigrant[feature] = np.random.uniform(mid, feature_max)
                            elif escape_dir == "decrease":
                                # Bias toward lower half of target range
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
                            # No constraints - use original sample ± random offset
                            immigrant[feature] = sample[feature] + np.random.uniform(
                                -1.0, 1.0
                            )
                    else:
                        # No constraint found - use original sample ± random offset
                        immigrant[feature] = sample[feature] + np.random.uniform(
                            -1.0, 1.0
                        )

                    # Ensure non-negative and round
                    immigrant[feature] = np.round(max(0, immigrant[feature]), 2)

                # Replace one of the worst offspring with this immigrant
                immigrant_ind = creator.Individual(immigrant)
                offspring[-(i + 1)] = immigrant_ind

            # Reduce mutation rate over generations (adaptive mutation)
            current_mutation_rate *= 0.99
            toolbox.unregister("mutate")
            toolbox.register(
                "mutate",
                self._mutate_individual,
                sample=sample,
                feature_names=feature_names,
                mutation_rate=current_mutation_rate,
                target_class=target_class,
                original_class=original_class,
                boundary_analysis=boundary_analysis,
            )

            # Elitism: Preserve best individuals from current population
            # Keep top 10% of current population (minimum 1, maximum 5)
            elite_size = max(1, min(5, int(0.1 * population_size)))

            # Sort current population by fitness (best first for minimization)
            sorted_population = sorted(
                population, key=lambda ind: ind.fitness.values[0]
            )
            elites = sorted_population[:elite_size]

            # Replace population with offspring, but keep elites
            # Replace worst individuals in offspring with elites
            population[:] = offspring[:-elite_size] + elites

        # Clean up multiprocessing pool if used
        if n_jobs != 1:
            pool.close()
            pool.join()

        # Store per-HOF evolution histories on model instance for later access
        self.hof_evolution_histories = hof_evolution_histories

        # Return the best individuals found
        # Check for both np.inf and INVALID_FITNESS to detect failed counterfactuals
        # (INVALID_FITNESS imported from constants module)
        valid_counterfactuals = []
        valid_cf_hof_indices = []  # Track which HOF indices correspond to valid CFs

        for i in range(len(hof)):
            best_fitness = hof[i].fitness.values[0]
            if best_fitness == np.inf or best_fitness >= INVALID_FITNESS:
                if self.verbose:
                    print(
                        f"Counterfactual #{i + 1} generation failed: fitness = {best_fitness}"
                    )
                continue

            # Final validation: verify the individual actually predicts the target class
            # AND has sufficient probability margin over other classes
            best_individual = dict(hof[i])
            features = np.array([best_individual[f] for f in sample.keys()]).reshape(
                1, -1
            )

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
                        print(
                            f"Counterfactual #{i + 1} failed: predicts class {predicted_class}, not target {target_class}"
                        )
                    continue

                # Check probability margin - target class should be clearly higher than second-best
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
                sorted_probs = np.sort(proba)[::-1]  # Descending order
                second_best_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                margin = target_prob - second_best_prob

                if margin < self.min_probability_margin:
                    if self.verbose:
                        print(
                            f"Counterfactual #{i + 1} rejected: target class probability ({target_prob:.3f}) "
                            f"not sufficiently higher than second-best ({second_best_prob:.3f}). "
                            f"Margin {margin:.3f} < required {self.min_probability_margin:.3f}"
                        )
                    continue

            except Exception as e:
                if self.verbose:
                    print(f"Counterfactual #{i + 1} validation failed with error: {e}")
                continue

            valid_counterfactuals.append(best_individual)
            valid_cf_hof_indices.append(i)  # Track which HOF index this CF came from
            if len(valid_counterfactuals) >= num_best_results:
                break

        # Build per-CF evolution histories for valid counterfactuals
        # Each valid CF gets the evolution history from its corresponding HOF position
        self.per_cf_evolution_histories = []
        for hof_idx in valid_cf_hof_indices:
            # Get the evolution history for this HOF position
            cf_history = hof_evolution_histories.get(hof_idx, [])
            # If no specific history, fall back to shared evolution_history
            if not cf_history:
                cf_history = list(self.evolution_history)
            self.per_cf_evolution_histories.append(cf_history)

        # Return None if no valid counterfactuals found, otherwise return the list
        if not valid_counterfactuals:
            return None
        return valid_counterfactuals

    def generate_counterfactual(
        self,
        sample,
        target_class,
        population_size=100,
        generations=100,
        mutation_rate=0.8,
        n_jobs=-1,
        allow_relaxation=True,
        relaxation_factor=2.0,
        num_best_results=1,
    ):
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
            raise ValueError(
                "Target class need to be different from the predicted class label."
            )

        # Pass original_class to enable dual-boundary GA
        counterfactuals = self.genetic_algorithm(
            sample,
            target_class,
            population_size,
            generations,
            mutation_rate=mutation_rate,
            n_jobs=n_jobs,
            original_class=sample_class,
            num_best_results=num_best_results,
        )

        # If strict constraints failed and relaxation is allowed, try with relaxed constraints
        if (
            (counterfactuals is None or len(counterfactuals) == 0)
            and allow_relaxation
            and self.constraints
        ):
            if self.verbose:
                print(
                    "\nStrict constraints failed. Attempting with relaxed constraints..."
                )

            # Store original constraints
            original_constraints = self.constraints

            # Try progressively relaxed constraints
            for relax_level in [relaxation_factor, relaxation_factor * 2, None]:
                if relax_level is None:
                    # Final attempt: no constraints (pure model optimization)
                    if self.verbose:
                        print(
                            "  Attempting without DPG constraints (pure classification optimization)..."
                        )
                    self.constraints = {}
                else:
                    if self.verbose:
                        print(
                            f"  Attempting with {relax_level}x relaxed constraints..."
                        )
                    self.constraints = self._relax_constraints(
                        original_constraints, relax_level
                    )

                counterfactuals = self.genetic_algorithm(
                    sample,
                    target_class,
                    population_size,
                    generations,
                    mutation_rate=mutation_rate,
                    n_jobs=n_jobs,
                    original_class=sample_class,
                    num_best_results=num_best_results,
                )

                if counterfactuals is not None and len(counterfactuals) > 0:
                    if self.verbose:
                        # Check constraint validity with original constraints
                        is_valid, penalty = self.validate_constraints(
                            counterfactuals[0],
                            sample,
                            target_class,
                            original_class=sample_class,
                            strict_mode=True,
                        )
                        print(
                            f"  Found {len(counterfactuals)} counterfactual(s) (original constraint valid: {is_valid}, penalty: {penalty:.2f}"
                        )
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
                        min_val = bounds.get("min")
                        max_val = bounds.get("max")

                        if min_val is not None and max_val is not None:
                            # Expand the range by factor
                            center = (min_val + max_val) / 2
                            half_range = (max_val - min_val) / 2
                            relaxed[class_key][feature] = {
                                "min": center - half_range * factor,
                                "max": center + half_range * factor,
                            }
                        else:
                            relaxed[class_key][feature] = bounds.copy()
                    else:
                        relaxed[class_key][feature] = bounds
            elif isinstance(class_constraints, list):
                # Legacy format: [{'feature': ..., 'min': ..., 'max': ...}, ...]
                relaxed[class_key] = []
                for c in class_constraints:
                    feature = c.get("feature", "")
                    min_val = c.get("min")
                    max_val = c.get("max")

                    if min_val is not None and max_val is not None:
                        center = (min_val + max_val) / 2
                        half_range = (max_val - min_val) / 2
                        relaxed[class_key].append(
                            {
                                "feature": feature,
                                "min": center - half_range * factor,
                                "max": center + half_range * factor,
                            }
                        )
                    else:
                        relaxed[class_key].append(c.copy())
        return relaxed

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
        Delegate to SampleGenerator for nearest counterfactual search.
        """
        return self.sample_generator.find_nearest_counterfactual(
            sample, target_class, X_train, y_train, metric, validate_prediction
        )
