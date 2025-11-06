import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ast
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.spatial.distance import cdist

class CounterFactualModel:
    def __init__(self, model, constraints, dict_non_actionable=None, verbose=False):
        """
        Initialize the CounterFactualDPG object.

        Args:
            model: The machine learning model used for predictions.
            constraints (dict): Nested dictionary containing constraints for features.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
              non_decreasing: feature cannot decrease
              non_increasing: feature cannot increase
              no_change: feature cannot change
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable #non_decreasing, non_increasing, no_change
        self.average_fitness_list = []
        self.best_fitness_list = []
        self.verbose = verbose

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
        #print('self.model.predict(counterfactual_sample)[0]', self.model.predict(counterfactual_sample)[0])
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
        and converting to lowercase. This helps match features that may have slight 
        variations in naming (e.g., "sepal width" vs "sepal width (cm)").
        
        Args:
            feature (str): The feature name to normalize.
            
        Returns:
            str: Normalized feature name.
        """
        import re
        # Remove anything in parentheses (like units)
        feature = re.sub(r'\s*\([^)]*\)', '', feature)
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
                for condition in class_constraints:
                    if self._features_match(condition["feature"], feature):
                        operator = condition["operator"]
                        constraint_value = condition["value"]

                        #print("Feature:", feature)
                        #print("Operator:", operator)
                        #print("Constraint Value:", constraint_value)
                        #print("New Value:", new_value)

                        # Check if the new value violates any constraints
                        if operator == "<" and not (new_value < constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == "<=" and not (new_value <= constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">" and not (new_value > constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">=" and not (new_value >= constraint_value):
                            valid_change = False
                            penalty += constraint_value

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
                for condition in non_target_class_constraints:
                    if self._features_match(condition["feature"], feature):
                        operator = condition["operator"]
                        constraint_value = condition["value"]

                        # Check if the new value violates any constraints
                        if operator == "<" and (new_value < constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == "<=" and (new_value <= constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">" and (new_value > constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">=" and (new_value >= constraint_value):
                            valid_change = False
                            penalty += constraint_value


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

        for feature, original_value in sample.items():
            min_value = -np.inf
            max_value = np.inf

            # Filter the constraints for the specified target class
            class_constraints = self.constraints.get(f"Class {target_class}", [])

            # Find the constraints for this feature
            for condition in class_constraints:
                if self._features_match(condition["feature"], feature):
                    operator = condition["operator"]
                    constraint_value = condition["value"]

                    # Update the min and max values based on the constraints
                    if operator == "<":
                        max_value = min(max_value, constraint_value - 1e-5)
                    elif operator == "<=":
                        max_value = min(max_value, constraint_value)
                    elif operator == ">":
                        min_value = max(min_value, constraint_value + 1e-5)
                    elif operator == ">=":
                        min_value = max(min_value, constraint_value)

            # Incorporate non-actionable constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    min_value = max(min_value, original_value)
                elif actionability == "non_increasing":
                    max_value = min(max_value, original_value)
                elif actionability == "no_change":
                    adjusted_sample[feature] = original_value
                    continue

            # Generate a random value within the valid range
            if min_value == -np.inf:
                min_value = 0  # Default lower bound if no constraint is specified
            if max_value == np.inf:
                max_value = min_value + 10  # Default upper bound if no constraint is specified

            adjusted_sample[feature] = np.random.uniform(min_value, max_value)

        return adjusted_sample

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        total_features = len(original_sample)
        unchanged_features = sum(
            original_sample[feature]*3 for feature in original_sample if original_sample[feature] != counterfactual_sample[feature]
        )
        sparsity = unchanged_features / total_features
        return sparsity

    def calculate_fitness(self, individual, original_features, sample, target_class, metric="cosine"):
            """
            Calculate the fitness score for an individual sample.

            Args:
                individual (dict): The individual sample with feature values.
                original_features (np.array): The original feature values.
                sample (dict): The original sample with feature values.
                target_class (int): The desired class for the counterfactual.
                metric (str): The distance metric to use for calculating distance.

            Returns:
                float: The fitness score for the individual.
            """
            #print('individual', individual)

            # Convert individual feature values to a numpy array
            features = np.array([individual[feature] for feature in sample.keys()]).reshape(1, -1)

            # Calculate validity score based on class
            is_valid_class = self.check_validity(features.flatten(), original_features.flatten(), target_class)
            #print('is_valid_class', is_valid_class)

            # Calculate distance score
            distance_score = self.calculate_distance(original_features, features.flatten(), metric)

            #Calculate sparcity (number of features modified)
            sparsity_score = self.calculate_sparsity(sample, individual)

            # Calculate_manufold_distance
            #manifold_distance = self.calculate_manifold_distance(self.X, individual)
            #print('calculate_manifold_distance', manifold_distance)

            # Check the constraints
            is_valid_constraint, penalty_constraints = self.validate_constraints(individual, sample, target_class)

            # Check if the change is actionable
            if not self.is_actionable_change(individual, sample) or not is_valid_class:
                fitness = +np.inf
                return fitness

            if is_valid_class :
                fitness = (2*distance_score) + penalty_constraints + sparsity_score
            elif is_valid_constraint:
                fitness = 5 * ((2*distance_score) + penalty_constraints + sparsity_score)  # High penalty for invalid samples
            else:
                fitness = 10 * ((2*distance_score) + penalty_constraints + sparsity_score)  # High penalty for invalid samples

            return fitness


    def genetic_algorithm(self, sample, target_class, population_size=100, generations=100, mutation_rate=0.8, metric="euclidean", delta_threshold=0.01, patience=10):
      # Initialize population with random values within a reasonable range
      population = []
      feature_names = list(sample.keys())
      previous_best_fitness = float('inf')
      stable_generations = 0  # Counter for generations with minimal fitness improvement

      for _ in range(population_size):
          individual = self.get_valid_sample(sample, target_class)
          population.append(individual)

      original_features = np.array([sample[feature] for feature in feature_names])

      self.best_fitness_list = []
      self.average_fitness_list = []
      # Main loop for generations
      for generation in range(generations):
          fitness_scores = []

          # Calculate fitness for each individual
          for individual in population:
              fitness = self.calculate_fitness(individual, original_features, sample, target_class, metric)
              fitness_scores.append(fitness)

          # Find the best candidate and its fitness score
          best_index = np.argmin(fitness_scores)
          best_candidate = population[best_index]
          best_fitness = fitness_scores[best_index]

          # Check for convergence based on the fitness delta threshold
          fitness_improvement = previous_best_fitness - best_fitness
          if fitness_improvement < delta_threshold:
              stable_generations += 1
          else:
              stable_generations = 0  # Reset if there's sufficient improvement

          # Print the average fitness and the best candidate

          finite_fitness_scores = np.array(fitness_scores)
          finite_fitness_scores[np.isinf(finite_fitness_scores)] = np.nan
          average_fitness = np.nanmean(finite_fitness_scores)
          if self.verbose:
            print(f"****** Generation {generation + 1}: Average Fitness = {average_fitness:.4f}, Best Fitness = {best_fitness:.4f}, fitness improvement = {fitness_improvement:.4f}")

          previous_best_fitness = best_fitness
          self.best_fitness_list.append(best_fitness)
          self.average_fitness_list.append(average_fitness)

          # Stop if improvement is less than the threshold for a consecutive number of generations
          if stable_generations >= patience:
              if self.verbose:
                print(f"Convergence reached at generation {generation + 1}")
              break

            # Use tournament selection to choose parents
          selected_parents = []
          for _ in range(population_size):
              tournament = np.random.choice(population, size=4, replace=False)
              tournament_fitness = [fitness_scores[population.index(ind)] for ind in tournament]
              selected_parents.append(tournament[np.argmin(tournament_fitness)])

          # Generate new population using crossover and mutation
          new_population = []
          for parent in selected_parents:
              offspring = parent.copy()
              for feature in feature_names:
                  if np.random.rand() < mutation_rate:
                      # Apply mutation only if the feature is actionable
                      if self.dict_non_actionable and feature in self.dict_non_actionable:
                          actionability = self.dict_non_actionable[feature]
                          original_value = parent[feature]
                          if actionability == "non_decreasing":
                              mutation_value = np.random.uniform(0, 0.5)  # Only allow increase
                              offspring[feature] += mutation_value
                          elif actionability == "non_increasing":
                              mutation_value = np.random.uniform(-0.5, 0)  # Only allow decrease
                              offspring[feature] += mutation_value
                          elif actionability == "no_change":
                              offspring[feature] = original_value  # Do not change
                          else:
                              # If no specific actionability rule, apply normal mutation
                              offspring[feature] += np.random.uniform(-0.5, 0.5)
                      else:
                          # If the feature is not in the non-actionable list, apply normal mutation
                          offspring[feature] += np.random.uniform(-0.5, 0.5)

                      # Ensure offspring values stay within valid domain constraints
                      offspring[feature] = np.round(max(0, offspring[feature]), 2)  # Adjust this based on domain-specific constraints
              new_population.append(offspring)


          # Reduce mutation rate over generations (adaptive mutation)
          mutation_rate *= 0.99

          # Update population
          population = new_population

      if best_fitness == np.inf:
          return None

      # Return the best individual based on the lowest fitness score
      best_index = np.argmin(fitness_scores)
      return population[best_index]

    def generate_counterfactual(self, sample, target_class, population_size=100, generations=100 ):
        """
        Generate a counterfactual for the given sample and target class using a genetic algorithm.

        Args:
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.

        Returns:
            dict: A modified sample representing the counterfactual or None if not found.
        """
        sample_class = self.model.predict(pd.DataFrame([sample]))[0]
        # Check if the predicted class matches the desired class
        if sample_class == target_class:
            raise ValueError("Target class need to be different from the predicted class label.")
        #counterfactual = None
        #while counterfactual is None:
        counterfactual = self.genetic_algorithm(sample, target_class, population_size, generations)
        return counterfactual
