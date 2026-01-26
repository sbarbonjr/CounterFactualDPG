import pandas as pd

from boundary_analyzer import BoundaryAnalyzer
from constraint_validator import ConstraintValidator
from fitness_calculator import FitnessCalculator
from mutation_strategy import MutationStrategy
from sample_generator import SampleGenerator
from genetic_algorithm_runner import GeneticAlgorithmRunner
from constants import UNCONSTRAINED_CHANGE_PENALTY_FACTOR


class CounterFactualModel:
    def __init__(
        self,
        model,
        constraints,
        dict_non_actionable=None,
        verbose=False,
        diversity_weight=0.1,
        repulsion_weight=0.1,
        boundary_weight=15.0,
        distance_factor=5.0,
        sparsity_factor=1.0,
        constraints_factor=3.0,
        original_escape_weight=2.0,
        escape_pressure=0.5,
        prioritize_non_overlapping=True,
        max_bonus_cap=10.0,
        unconstrained_penalty_factor=UNCONSTRAINED_CHANGE_PENALTY_FACTOR,
        X_train=None,
        y_train=None,
        min_probability_margin=0.001,
        generation_debugging=False,
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
            unconstrained_penalty_factor (float): Penalty multiplier for changing features without target constraints.
                Higher values (e.g., 2.0-3.0) make unconstrained features change as last resort.
            X_train (DataFrame): Training data features for nearest neighbor fallback.
            y_train (Series): Training data labels for nearest neighbor fallback.
            min_probability_margin (float): Minimum margin the target class probability must exceed the
                second-highest class probability by. Prevents accepting weak counterfactuals where
                the prediction is essentially a tie. Default 0.001 (0.1% margin).
            generation_debugging (bool): Enable detailed per-generation fitness component tracking.
                When True, collects fitness breakdown (distance, sparsity, penalties, bonuses) for
                the best individual each generation, exported as a table for WandB logging.
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = (
            dict_non_actionable  # non_decreasing, non_increasing, no_change
        )
        self.average_fitness_list = []
        self.best_fitness_list = []
        self.std_fitness_list = []
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
            unconstrained_penalty_factor=unconstrained_penalty_factor,
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
        # Initialize GeneticAlgorithmRunner for GA execution
        self.ga_runner = GeneticAlgorithmRunner(
            model=model,
            constraints=constraints,
            feature_names=self.feature_names,
            verbose=verbose,
            min_probability_margin=min_probability_margin,
            generation_debugging=generation_debugging,
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
        metric="euclidean",
        population=None,
        original_class=None,
        return_components=False,
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
            return_components=return_components,
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
        # Pre-compute boundary analysis for dual-boundary operations
        boundary_analysis = None
        if original_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(
                original_class, target_class
            )

        # Delegate to GeneticAlgorithmRunner
        result = self.ga_runner.run(
            sample=sample,
            target_class=target_class,
            original_class=original_class,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            metric=metric,
            delta_threshold=delta_threshold,
            patience=patience,
            n_jobs=n_jobs,
            num_best_results=num_best_results,
            boundary_analysis=boundary_analysis,
            create_individual_func=self._create_deap_individual,
            crossover_func=self._crossover_dict,
            mutate_func=self._mutate_individual,
            calculate_fitness_func=self.calculate_fitness,
            get_valid_sample_func=self.get_valid_sample,
            normalize_feature_func=self._normalize_feature_name,
            features_match_func=self._features_match,
        )

        # Copy tracking attributes back from runner
        self.best_fitness_list = self.ga_runner.best_fitness_list
        self.average_fitness_list = self.ga_runner.average_fitness_list
        self.std_fitness_list = self.ga_runner.std_fitness_list
        self.evolution_history = self.ga_runner.evolution_history
        self.hof_evolution_histories = self.ga_runner.hof_evolution_histories
        self.per_cf_evolution_histories = self.ga_runner.per_cf_evolution_histories
        self.cf_generation_found = getattr(self.ga_runner, 'cf_generation_found', [])
        self.generation_debug_table = self.ga_runner.generation_debug_table

        return result

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
