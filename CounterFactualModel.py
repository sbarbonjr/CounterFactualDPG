import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ast
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.spatial.distance import cdist

from deap import base, creator, tools, algorithms

class CounterFactualModel:
    def __init__(self, model, constraints, dict_non_actionable=None, verbose=False, 
                 diversity_weight=0.5, repulsion_weight=4.0, boundary_weight=15.0, 
                 distance_factor=2.0, sparsity_factor=1.0, constraints_factor=3.0):
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
        # Store feature names from the model if available
        self.feature_names = getattr(model, 'feature_names_in_', None)

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
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot best fitness and average fitness on the same graph
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
        original_sample = np.array(original_sample)
        counterfactual_sample = np.array(counterfactual_sample)

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

    def validate_constraints(self, S_prime, sample, target_class):
        """
        Validate if the modified sample S_prime meets all constraints for the specified target class.

        Args:
            S_prime (dict): Modified sample with feature values.
            sample (dict): The original sample with feature values.
            target_class (int): The target class for filtering constraints.

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

        # Collect all constraints that are NOT related to the target class
        non_target_class_constraints = [
            condition
            for class_name, conditions in self.constraints.items()
            if class_name != "Class " + str(target_class)  # Exclude the target class constraints
            for condition in conditions
        ]

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)

            # Check if the feature value has changed
            if new_value != original_value:
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


    def get_valid_sample(self, sample, target_class):
        """
        Generate a valid sample that meets all constraints for the specified target class
        while respecting actionable changes.

        Args:
            sample (dict): The sample with feature values.
            target_class (int): The target class for filtering constraints.

        Returns:
            dict: A valid sample that meets all constraints for the target class
                  and respects actionable changes.
        """
        adjusted_sample = sample.copy()  # Start with the original values
        # Filter the constraints for the specified target class
        class_constraints = self.constraints.get(f"Class {target_class}", [])

        for feature, original_value in sample.items():
            min_value = -np.inf
            max_value = np.inf


            # Find the constraints for this feature using direct lookup
            matching_constraint = next(
                (condition for condition in class_constraints if self._features_match(condition["feature"], feature)),
                None
            )
            
            if matching_constraint:
                min_value = matching_constraint.get("min") if matching_constraint.get("min") is not None else -np.inf
                max_value = matching_constraint.get("max") if matching_constraint.get("max") is not None else np.inf

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

            # If no explicit min/max constraints, generate values near the original value
            if min_value == -np.inf:
                min_value = original_value - 0.5 * (abs(original_value) + 1.0)
            if max_value == np.inf:
                max_value = original_value + 0.5 * (abs(original_value) + 1.0)

            adjusted_sample[feature] = np.random.uniform(min_value, max_value)
        return adjusted_sample

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        total_features = len(original_sample)
        unchanged_features = sum(
            original_sample[feature]*3 for feature in original_sample if original_sample[feature] != counterfactual_sample[feature]
        )
        sparsity = unchanged_features / total_features
        return sparsity

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

    def calculate_fitness(self, individual, original_features, sample, target_class, metric="cosine", population=None):
            """
            Calculate the fitness score for an individual sample using weighted components.
            Based on the total_fitness logic from dpg_aug.ipynb.

            Args:
                individual (dict): The individual sample with feature values.
                original_features (np.array): The original feature values.
                sample (dict): The original sample with feature values.
                target_class (int): The desired class for the counterfactual.
                metric (str): The distance metric to use for calculating distance.
                population (list): The current population for diversity calculations.

            Returns:
                float: The fitness score for the individual (lower is better).
            """
            INVALID_FITNESS = 1e6  # Large penalty for invalid samples
            
            # Convert individual feature values to a numpy array
            features = np.array([individual[feature] for feature in sample.keys()]).reshape(1, -1)

            # Check if the change is actionable
            if not self.is_actionable_change(individual, sample):
                return INVALID_FITNESS

            # Calculate validity score based on class
            is_valid_class = self.check_validity(features.flatten(), original_features.flatten(), target_class)

            # Check the constraints
            is_valid_constraint, penalty_constraints = self.validate_constraints(individual, sample, target_class)

            # Base fitness calculation
            if not is_valid_class:
                # If class is wrong, return high penalty
                return INVALID_FITNESS
            
            # Calculate core components
            distance_score = self.calculate_distance(original_features, features.flatten(), metric)
            sparsity_score = self.calculate_sparsity(sample, individual)
            
            # Base fitness (minimize distance and sparsity, penalize constraint violations)
            base_fitness = (self.distance_factor * distance_score + 
                          self.sparsity_factor * sparsity_score + 
                          self.constraints_factor * penalty_constraints)
            
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
                
                # Penalty for being too far from boundary
                boundary_penalty = 50.0 if dist_line > 0.1 else 0.0
                
                # Total fitness (lower is better, so we subtract bonuses)
                fitness = base_fitness - div_bonus - rep_bonus - line_bonus + boundary_penalty
            else:
                # Without population, just use base fitness
                fitness = base_fitness
            
            # Additional penalty for constraint violations
            if not is_valid_constraint:
                fitness *= 5.0  # Multiply penalty for constraint violations

            return fitness


    def _create_deap_individual(self, sample_dict, feature_names):
        """Create a DEAP individual from a dictionary."""
        individual = creator.Individual(sample_dict)
        return individual

    def _mutate_individual(self, individual, sample, feature_names, mutation_rate):
        """Custom mutation operator that respects actionability constraints."""
        for feature in feature_names:
            if np.random.rand() < mutation_rate:
                # Apply mutation only if the feature is actionable
                if self.dict_non_actionable and feature in self.dict_non_actionable:
                    actionability = self.dict_non_actionable[feature]
                    original_value = sample[feature]
                    if actionability == "non_decreasing":
                        mutation_value = np.random.uniform(0, 0.5)  # Only allow increase
                        individual[feature] += mutation_value
                    elif actionability == "non_increasing":
                        mutation_value = np.random.uniform(-0.5, 0)  # Only allow decrease
                        individual[feature] += mutation_value
                    elif actionability == "no_change":
                        individual[feature] = original_value  # Do not change
                    else:
                        # If no specific actionability rule, apply normal mutation
                        individual[feature] += np.random.uniform(-0.5, 0.5)
                else:
                    # If the feature is not in the non-actionable list, apply normal mutation
                    individual[feature] += np.random.uniform(-0.5, 0.5)

                # Ensure offspring values stay within valid domain constraints
                individual[feature] = np.round(max(0, individual[feature]), 2)
        return individual,

    def _crossover_dict(self, ind1, ind2, indpb):
        """Custom crossover operator for dict-based individuals.
        
        Args:
            ind1, ind2: Parent individuals (dicts)
            indpb: Probability of swapping each feature
            
        Returns:
            Tuple of two offspring individuals
        """
        for key in ind1.keys():
            if np.random.rand() < indpb:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def genetic_algorithm(self, sample, target_class, population_size=100, generations=100, mutation_rate=0.8, metric="euclidean", delta_threshold=0.01, patience=10, n_jobs=-1):
        """Genetic algorithm implementation using DEAP framework.
        
        Args:
            n_jobs (int): Number of parallel jobs for fitness evaluation. 
                         -1 = use all CPUs (default), 1 = sequential.
        """
        feature_names = list(sample.keys())
        original_features = np.array([sample[feature] for feature in feature_names])
        
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
        
        # Register individual creation
        toolbox.register("individual", self._create_deap_individual, 
                        sample_dict=self.get_valid_sample(sample, target_class),
                        feature_names=feature_names)
        
        # Register population creation
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register mating with custom dict crossover
        toolbox.register("mate", self._crossover_dict, indpb=0.6)

        # Register genetic operators
        toolbox.register("evaluate", lambda ind: (self.calculate_fitness(
            ind, original_features, sample, target_class, metric, population),))
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("mutate", self._mutate_individual, 
                        sample=sample, 
                        feature_names=feature_names,
                        mutation_rate=mutation_rate)
        
        # Create initial population
        population = [self._create_deap_individual(self.get_valid_sample(sample, target_class), feature_names) 
                     for _ in range(population_size)]
        
        # Setup statistics
        # Define INVALID_FITNESS threshold for filtering statistics
        INVALID_FITNESS = 1e6
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.nanmean([val[0] for val in x if not np.isinf(val[0]) and val[0] < INVALID_FITNESS]))
        stats.register("min", lambda x: np.nanmin([val[0] for val in x if not np.isinf(val[0]) and val[0] < INVALID_FITNESS]) if any(not np.isinf(val[0]) and val[0] < INVALID_FITNESS for val in x) else np.inf)
        
        # Setup hall of fame to keep best individuals
        hof = tools.HallOfFame(1)
        
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
            
            # Reduce mutation rate over generations (adaptive mutation)
            current_mutation_rate *= 0.99
            toolbox.unregister("mutate")
            toolbox.register("mutate", self._mutate_individual, 
                           sample=sample, 
                           feature_names=feature_names,
                           mutation_rate=current_mutation_rate)
            
            # Replace population
            population[:] = offspring
        
        # Clean up multiprocessing pool if used
        if n_jobs != 1:
            pool.close()
            pool.join()
        
        # Return the best individual found
        if hof[0].fitness.values[0] == np.inf:
            return None
        
        return dict(hof[0])

    def generate_counterfactual(self, sample, target_class, population_size=100, generations=100, mutation_rate=0.8, n_jobs=-1):
        """
        Generate a counterfactual for the given sample and target class using a genetic algorithm.

        Args:
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.
            population_size (int): Size of the population for the genetic algorithm.
            generations (int): Number of generations to run.
            mutation_rate (float): Per-feature mutation probability.
            n_jobs (int): Number of parallel jobs. -1=all CPUs (default), 1=sequential.

        Returns:
            dict: A modified sample representing the counterfactual or None if not found.
        """
        sample_class = self.model.predict(pd.DataFrame([sample]))[0]

        if sample_class == target_class:
            raise ValueError("Target class need to be different from the predicted class label.")

        counterfactual = self.genetic_algorithm(sample, target_class, population_size, generations, mutation_rate=mutation_rate, n_jobs=n_jobs)
        return counterfactual
