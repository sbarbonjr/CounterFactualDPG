import pandas as pd

from boundary_analyzer import BoundaryAnalyzer
from utils.feature_utils import normalize_feature_name, features_match
from constraint_validator import ConstraintValidator
from fitness_calculator import FitnessCalculator
from sample_generator import SampleGenerator
from heuristic_runner import HeuristicRunner
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
        min_probability_margin=0.001,
        overgeneration_factor=20,
        requested_counterfactuals=5,
        diversity_lambda=0.5,
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
            min_probability_margin (float): Minimum margin the target class probability must exceed the
                second-highest class probability by. Prevents accepting weak counterfactuals where
                the prediction is essentially a tie. Default 0.001 (0.1% margin).
            overgeneration_factor (int): Multiplier for population size calculation. The population
                size is calculated as overgeneration_factor * requested_counterfactuals.
                Default 20 means if 5 CFs are requested, 100 candidates are generated initially.
                Higher values increase quality at the cost of computation.
            requested_counterfactuals (int): Number of counterfactuals to generate. Used with
                overgeneration_factor to calculate population_size internally.
                Default 5.
            diversity_lambda (float): Weight for diversity vs proximity trade-off in CF selection (0-1).
                Higher values (e.g., 0.7-0.9) prioritize selecting diverse counterfactuals.
                Lower values (e.g., 0.1-0.3) prioritize selecting the best-fitness/closest counterfactuals.
                Default 0.5 for balanced selection.
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
        self.unconstrained_penalty_factor = unconstrained_penalty_factor
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
            verbose=verbose,
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
            verbose=verbose,
        )
        # Initialize SampleGenerator for sample generation
        self.sample_generator = SampleGenerator(
            model=model,
            constraints=constraints,
            dict_non_actionable=dict_non_actionable,
            feature_names=self.feature_names,
            escape_pressure=escape_pressure,    
            min_probability_margin=min_probability_margin,
            verbose=verbose,
            boundary_analyzer=self.boundary_analyzer,
            constraint_validator=self.constraint_validator,
        )
        # Initialize HeuristicRunner for candidate generation
        self.heuristic_runner = HeuristicRunner(
            model=model,
            constraints=constraints,
            dict_non_actionable=dict_non_actionable,
            feature_names=self.feature_names,
            verbose=verbose,
            min_probability_margin=min_probability_margin,
            diversity_lambda=diversity_lambda,
        )
        # Minimum probability margin for accepting counterfactuals
        self.min_probability_margin = min_probability_margin
        # Overgeneration factor and requested counterfactuals for population size calculation
        self.overgeneration_factor = overgeneration_factor
        self.requested_counterfactuals = requested_counterfactuals
        # Calculate population_size internally: overgeneration_factor * requested_counterfactuals
        self.population_size = overgeneration_factor * requested_counterfactuals

        # Print configuration if verbose
        if self.verbose:
            print("=" * 80)
            print("CounterFactualModel Configuration")
            print("=" * 80)
            print(f"Model type: {type(self.model).__name__}")
            print(f"Verbose: {self.verbose}")
            print(f"Feature names: {self.feature_names}")
            print()
            print("Fitness Weights:")
            print(f"  - diversity_weight: {self.diversity_weight}")
            print(f"  - repulsion_weight: {self.repulsion_weight}")
            print(f"  - boundary_weight: {self.boundary_weight}")
            print(f"  - distance_factor: {self.distance_factor}")
            print(f"  - sparsity_factor: {self.sparsity_factor}")
            print(f"  - constraints_factor: {self.constraints_factor}")
            print(f"  - original_escape_weight: {self.original_escape_weight}")
            print(f"  - max_bonus_cap: {self.max_bonus_cap}")
            print()
            print("Generation Parameters:")
            print(f"  - escape_pressure: {self.escape_pressure}")
            print(f"  - prioritize_non_overlapping: {self.prioritize_non_overlapping}")
            print(f"  - unconstrained_penalty_factor: {self.unconstrained_penalty_factor}")
            print()
            print("Validation Parameters:")
            print(f"  - min_probability_margin: {self.min_probability_margin}")
            print()
            print("Population Parameters:")
            print(f"  - overgeneration_factor: {self.overgeneration_factor}")
            print(f"  - requested_counterfactuals: {self.requested_counterfactuals}")
            print(f"  - calculated population_size: {self.population_size}")
            print("=" * 80)

    def _analyze_boundary_overlap(self, original_class, target_class):
        """Delegate to BoundaryAnalyzer for boundary overlap analysis."""
        return self.boundary_analyzer.analyze_boundary_overlap(original_class, target_class)

    def is_actionable_change(self, counterfactual_sample, original_sample):
        """Delegate to ConstraintValidator for actionability check."""
        return self.constraint_validator.is_actionable_change(
            counterfactual_sample, original_sample
        )

    def check_validity(self, counterfactual_sample, original_sample, desired_class):
        """Delegate to ConstraintValidator for validity check."""
        return self.constraint_validator.check_validity(
            counterfactual_sample, original_sample, desired_class
        )

    def calculate_distance(
        self, original_sample, counterfactual_sample, metric="euclidean"
    ):
        """Delegate to FitnessCalculator for distance calculation."""
        return self.fitness_calculator.calculate_distance(
            original_sample, counterfactual_sample, metric
        )

    def _normalize_feature_name(self, feature):
        """Delegate to feature_utils for feature name normalization."""
        return normalize_feature_name(feature)

    def _features_match(self, feature1, feature2):
        """Delegate to feature_utils for feature matching."""
        return features_match(feature1, feature2)

    def validate_constraints(
        self, S_prime, sample, target_class, original_class=None, strict_mode=True
    ):
        """Delegate to ConstraintValidator for constraint validation."""
        return self.constraint_validator.validate_constraints(
            S_prime, sample, target_class, original_class, strict_mode
        )

    def get_valid_sample(self, sample, target_class, original_class=None, weak_constraints=True):
        """Delegate to SampleGenerator for valid sample generation."""
        return self.sample_generator.get_valid_sample(sample, target_class, original_class,weak_constraints)

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        """Delegate to FitnessCalculator for sparsity calculation."""
        return self.fitness_calculator.calculate_sparsity(
            original_sample, counterfactual_sample
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
        """Delegate to FitnessCalculator for fitness calculation."""
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

    def generate_candidates(
        self,
        sample,
        target_class,
        metric="euclidean",
        original_class=None,
        num_best_results=None,
    ):
        """Generate counterfactual candidates using heuristic approach.

        Uses escape-aware perturbations to generate candidates, evaluates fitness,
        and returns the best candidates using greedy diverse selection.

        Population size is calculated internally as:
        population_size = overgeneration_factor * requested_counterfactuals

        Args:
            sample (dict): Original sample features.
            target_class (int): Target class for counterfactual.
            metric (str): Distance metric for fitness calculation.
            original_class (int): Original class for escape-aware generation.
            num_best_results (int): Number of top individuals to return. If None,
                uses the instance's requested_counterfactuals.

        Returns:
            list: Valid counterfactuals (or None if none found).
        """
        # Pre-compute boundary analysis for dual-boundary operations
        boundary_analysis = None
        if original_class is not None:
            boundary_analysis = self._analyze_boundary_overlap(
                original_class, target_class
            )
        
        # Use instance's calculated population_size
        population_size = self.population_size
        
        # Use provided num_best_results or fall back to instance default
        if num_best_results is None:
            num_best_results = self.requested_counterfactuals

        # Delegate to HeuristicRunner
        result = self.heuristic_runner.run(
            sample=sample,
            target_class=target_class,
            original_class=original_class,
            population_size=population_size,
            metric=metric,
            num_best_results=num_best_results,
            boundary_analysis=boundary_analysis,
            create_individual_func=lambda d, _: dict(d),  # Simple dict copy
            calculate_fitness_func=self.calculate_fitness,
            get_valid_sample_func=self.get_valid_sample,
            normalize_feature_func=self._normalize_feature_name,
            features_match_func=self._features_match,
            overgeneration_factor=self.overgeneration_factor,
        )

        # Copy tracking attributes back from runner
        self.best_fitness_list = self.heuristic_runner.best_fitness_list
        self.average_fitness_list = self.heuristic_runner.average_fitness_list
        self.std_fitness_list = self.heuristic_runner.std_fitness_list
        self.evolution_history = self.heuristic_runner.evolution_history
        self.per_cf_evolution_histories = self.heuristic_runner.per_cf_evolution_histories
        self.cf_generation_found = getattr(self.heuristic_runner, 'cf_generation_found', [])

        return result

    def generate_counterfactual(
        self,
        sample,
        target_class,
        num_best_results=None,
    ):
        """
        Generate counterfactuals for the given sample and target class.

        Uses heuristic approach to generate candidate counterfactuals.

        Population size is calculated internally as:
        population_size = overgeneration_factor * requested_counterfactuals

        Args:
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.
            num_best_results (int): Number of top counterfactuals to return. If None,
                uses the instance's requested_counterfactuals.

        Returns:
            list or None: A list of counterfactuals, or None if not found.
        """
        sample_class = self.model.predict(pd.DataFrame([sample]))[0]

        if sample_class == target_class:
            raise ValueError(
                "Target class need to be different from the predicted class label."
            )

        # Use provided num_best_results or fall back to instance default
        if num_best_results is None:
            num_best_results = self.requested_counterfactuals

        # Generate counterfactuals using heuristic approach
        counterfactuals = self.generate_candidates(
            sample,
            target_class,
            original_class=sample_class,
            num_best_results=num_best_results,
        )

        return counterfactuals

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
