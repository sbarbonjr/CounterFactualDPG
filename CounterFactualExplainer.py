import numpy as np
import pandas as pd

# Import comprehensive metrics module
try:
    from CounterFactualMetrics import evaluate_cf_list as evaluate_cf_list_comprehensive
    COMPREHENSIVE_METRICS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_METRICS_AVAILABLE = False


class CounterFactualExplainer:
    def __init__(self, model, original_sample, counterfactual_sample, target_class):
        """
        Initialize the explainer object.

        Args:
            model (CounterFactualModel): The model used to generate the counterfactual.
            original_sample (dict): Original input sample.
            counterfactual_sample (dict): Generated counterfactual sample.
            target_class (int): Target class for the counterfactual.
        """
        self.model = model
        self.original_sample = original_sample
        self.counterfactual_sample = counterfactual_sample
        self.target_class = target_class

    def explain_feature_modifications(self):
        """
        Returns a list of dictionaries containing feature modification details.
        Each dictionary contains: feature_name, old_value, new_value
        """
        changes = []
        for feature in self.original_sample:
            original_value = self.original_sample[feature]
            new_value = self.counterfactual_sample.get(feature, original_value)
            if original_value != new_value:
                changes.append({
                    'feature_name': feature,
                    'old_value': original_value,
                    'new_value': new_value
                })
        return changes

    def check_constraints_respect(self):
        valid, penalty = self.model.validate_constraints(self.counterfactual_sample, self.original_sample, self.target_class)
        if valid:
            return "All constraints were respected."
        else:
            return f"Constraints were violated. Total penalty for violations: {penalty}"

    def explain_stopping_criteria(self):
        if hasattr(self.model, 'stopped_reason'):
            return f"The genetic algorithm stopped due to: {self.model.stopped_reason}."
        else:
            return "Stopping criteria not set or not reached."

    def summarize_final_results(self):
        # Convert samples to DataFrame to use model.predict
        original_class = self.model.model.predict(pd.DataFrame([self.original_sample]))[0]
        counterfactual_class = self.model.model.predict(pd.DataFrame([self.counterfactual_sample]))[0]
        
        # Convert original_sample to numpy array
        original_features_array = np.array(list(self.original_sample.values()))
        
        # Convert counterfactual_sample to numpy array
        counterfactual_features_array = np.array(list(self.counterfactual_sample.values()))
        
        # Now calculate fitness
        original_fitness = self.model.calculate_fitness(self.original_sample, original_features_array, self.original_sample, original_class)
        best_fitness = self.model.calculate_fitness(self.counterfactual_sample, counterfactual_features_array, self.original_sample, self.target_class)
        
        return (f"Original class: {original_class}, Counterfactual class: {counterfactual_class}\n"
                f"Original fitness: {original_fitness}, Best fitness: {best_fitness}")

    def get_all_metrics(self, X_train=None, X_test=None, variable_features=None, 
                        continuous_features=None, categorical_features=None, 
                        compute_comprehensive=False):
        """
        Return all metrics as a flat dictionary suitable for experiment tracking (e.g., wandb).
        
        Args:
            X_train: Training data (optional, required for comprehensive metrics)
            X_test: Test data (optional, required for comprehensive metrics)
            variable_features: List of actionable feature indices (optional)
            continuous_features: List of continuous feature indices (optional)
            categorical_features: List of categorical feature indices (optional)
            compute_comprehensive: If True and dependencies available, compute full ECE metrics
        
        Returns:
            dict: Dictionary containing all extractable metrics from the counterfactual generation.
        """
        # Get feature modifications
        feature_changes = self.explain_feature_modifications()
        num_feature_changes = len(feature_changes)
        
        # Check constraints
        constraints_message = self.check_constraints_respect()
        constraints_respected = "All" in constraints_message
        
        # Get constraint penalty
        _, constraint_penalty = self.model.validate_constraints(
            self.counterfactual_sample, 
            self.original_sample, 
            self.target_class
        )
        
        # Get classes
        original_class = int(self.model.model.predict(pd.DataFrame([self.original_sample]))[0])
        counterfactual_class = int(self.model.model.predict(pd.DataFrame([self.counterfactual_sample]))[0])
        
        # Convert to arrays for distance calculations
        original_features_array = np.array(list(self.original_sample.values()))
        counterfactual_features_array = np.array(list(self.counterfactual_sample.values()))
        
        # Calculate fitness scores
        original_fitness = self.model.calculate_fitness(
            self.original_sample, 
            original_features_array, 
            self.original_sample, 
            original_class
        )
        best_fitness = self.model.calculate_fitness(
            self.counterfactual_sample, 
            counterfactual_features_array, 
            self.original_sample, 
            self.target_class
        )
        
        # Calculate distances
        distance_euclidean = self.model.calculate_distance(
            original_features_array, 
            counterfactual_features_array, 
            metric="euclidean"
        )
        distance_manhattan = self.model.calculate_distance(
            original_features_array, 
            counterfactual_features_array, 
            metric="manhattan"
        )
        
        # Calculate sparsity
        sparsity = self.model.calculate_sparsity(
            self.original_sample, 
            self.counterfactual_sample
        )
        
        basic_metrics = {
            'num_feature_changes': num_feature_changes,
            'constraints_respected': constraints_respected,
            'constraint_penalty': float(constraint_penalty),
            'original_class': original_class,
            'counterfactual_class': counterfactual_class,
            'target_class': int(self.target_class),
            'class_changed': original_class != counterfactual_class,
            'reached_target_class': counterfactual_class == self.target_class,
            'original_fitness': float(original_fitness),
            'best_fitness': float(best_fitness),
            'fitness_improvement': float(original_fitness - best_fitness),
            'distance_euclidean': float(distance_euclidean),
            'distance_manhattan': float(distance_manhattan),
            'sparsity': float(sparsity),
        }
        
        # If comprehensive metrics requested and available
        if compute_comprehensive and COMPREHENSIVE_METRICS_AVAILABLE and X_train is not None:
            try:
                # Prepare data
                cf_array = counterfactual_features_array.reshape(1, -1)
                x_array = original_features_array.reshape(1, -1)
                
                # Default feature indices if not provided
                nbr_features = len(original_features_array)
                if variable_features is None:
                    variable_features = list(range(nbr_features))
                if continuous_features is None:
                    continuous_features = list(range(nbr_features))
                if categorical_features is None:
                    categorical_features = []
                
                # Compute ratio of continuous features
                ratio_cont = len(continuous_features) / nbr_features if nbr_features > 0 else 1.0
                
                # Use test data if provided, otherwise use training data
                if X_test is None:
                    X_test = X_train
                
                # Compute comprehensive metrics
                comprehensive = evaluate_cf_list_comprehensive(
                    cf_list=cf_array,
                    x=x_array[0],
                    model=self.model.model,
                    y_val=original_class,
                    max_nbr_cf=1,
                    variable_features=variable_features,
                    continuous_features_all=continuous_features,
                    categorical_features_all=categorical_features,
                    X_train=X_train,
                    X_test=X_test,
                    ratio_cont=ratio_cont,
                    nbr_features=nbr_features
                )
                
                # Merge with basic metrics (basic takes precedence for overlapping keys)
                comprehensive.update(basic_metrics)
                return comprehensive
                
            except Exception as e:
                print(f"WARNING: Comprehensive metrics computation failed: {e}")
                return basic_metrics
        
        return basic_metrics

# Example of usage
#model = CounterFactualModel(...)  # Assume this is your model
#original_sample = {...}  # Original sample dictionary
#counterfactual_sample = model.generate_counterfactual(original_sample, target_class=1)  # Generated counterfactual
#explainer = CounterFactualExplainer(model, original_sample, counterfactual_sample, target_class=1)

#print(explainer.explain_feature_modifications())
#print(explainer.check_constraints_respect())
#print(explainer.explain_stopping_criteria())
#print(explainer.summarize_final_results())
