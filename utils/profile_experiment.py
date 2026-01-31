"""
Performance profiling script for CounterFactual experiment
Analyzes the performance of the main cell up to the 10th combination
"""

import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import itertools
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
from typing import cast
from CounterFactualModel import CounterFactualModel
from ConstraintParser import ConstraintParser
import CounterFactualVisualizer as CounterFactualVisualizer
from CounterFactualVisualizer import (plot_pca_with_counterfactual, plot_sample_and_counterfactual_heatmap, 
                                     plot_pca_loadings, plot_constraints, 
                                     plot_sample_and_counterfactual_comparison, plot_pairwise_with_counterfactual_df,
                                     plot_pca_with_counterfactuals, plot_explainer_summary)
from CounterFactualExplainer import CounterFactualExplainer
import warnings

warnings.filterwarnings("ignore")

# Setup constants (same as notebook)
CLASS_COLORS_LIST = ['purple', 'green', 'orange']
IRIS: Bunch = cast(Bunch, load_iris())
IRIS_FEATURES = IRIS.data
IRIS_LABELS = IRIS.target

# Create DataFrame with feature names for consistent handling
IRIS_FEATURES_DF = pd.DataFrame(IRIS_FEATURES, columns=IRIS.feature_names)

TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
    IRIS_FEATURES_DF, IRIS_LABELS, test_size=0.3, random_state=42
)

MODEL = RandomForestClassifier(n_estimators=3, random_state=42)
MODEL.fit(TRAIN_FEATURES, TRAIN_LABELS)

ORIGINAL_SAMPLE = {
    'petal width (cm)': 6.1,
    'petal length (cm)': 2.8,
    'sepal length (cm)': 4.7,
    'sepal width (cm)': 1.2
}
SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])

CONSTRAINT_PARSER = ConstraintParser("constraints/custom_l100_pv0.001_t2_dpg_metrics.txt")
CONSTRAINTS = CONSTRAINT_PARSER.read_constraints_from_file()

TARGET_CLASS = 0
ORIGINAL_SAMPLE_PREDICTED_CLASS = MODEL.predict(SAMPLE_DATAFRAME)

RULES = ['no_change', 'non_increasing', 'non_decreasing']
FEATURES_NAMES = list(ORIGINAL_SAMPLE.keys())
RULES_COMBINATIONS = list(itertools.product(RULES, repeat=len(FEATURES_NAMES)))

NUMBER_OF_COMBINATIONS_TO_TEST = 10  # Profile only first 10 combinations
NUMBER_OF_REPLICATIONS_PER_COMBINATION = 10
INITIAL_POPULATION_SIZE = 20
MAX_GENERATIONS = 60


def run_profiled_experiment():
    """Run the experiment with profiling enabled"""
    
    counterfactuals_df_combinations = []
    visualizations = []
    
    print(f"Running profiled experiment for {NUMBER_OF_COMBINATIONS_TO_TEST} combinations...")
    print(f"Each combination has {NUMBER_OF_REPLICATIONS_PER_COMBINATION} replications\n")
    
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:NUMBER_OF_COMBINATIONS_TO_TEST]):
        print(f"Processing Combination {combination_num + 1}/{NUMBER_OF_COMBINATIONS_TO_TEST}...")
        
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
        counterfactuals_df_replications = []
        combination_viz = {
            'label': combination,
            'pairwise': None,
            'pca': None,
            'replication': []
        }
        
        skip_combination = False
        
        for replication in range(NUMBER_OF_REPLICATIONS_PER_COMBINATION):
            if skip_combination:
                break
            
            print(f"  Replication {replication + 1}/{NUMBER_OF_REPLICATIONS_PER_COMBINATION}...", end=" ")
            
            cf_dpg = CounterFactualModel(MODEL, CONSTRAINTS)
            cf_dpg.dict_non_actionable = dict_non_actionable
            
            counterfactual = cf_dpg.generate_counterfactual(
                ORIGINAL_SAMPLE, TARGET_CLASS, INITIAL_POPULATION_SIZE, MAX_GENERATIONS
            )
            
            if counterfactual is None:
                if replication == 2:
                    print("(FAILED - skipping remaining)")
                    skip_combination = True
                else:
                    print("(FAILED)")
                continue
            
            print("(SUCCESS)")
            
            # Create visualizations
            replication_viz = [
                plot_sample_and_counterfactual_heatmap(
                    ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, counterfactual,
                    MODEL.predict(pd.DataFrame([counterfactual])), dict_non_actionable
                ),
                plot_sample_and_counterfactual_comparison(
                    MODEL, ORIGINAL_SAMPLE, SAMPLE_DATAFRAME, counterfactual, CLASS_COLORS_LIST
                ),
                cf_dpg.plot_fitness()
            ]
            combination_viz['replication'].append(replication_viz)
            
            EXPLAINER = CounterFactualExplainer(cf_dpg, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
            plot_explainer_summary(EXPLAINER, ORIGINAL_SAMPLE, counterfactual)
            
            # Prepare data for DataFrame
            cf_data = counterfactual.copy()
            cf_data.update({'Rule_' + k: v for k, v in dict_non_actionable.items()})
            cf_data['Replication'] = replication + 1
            counterfactuals_df_replications.append(cf_data)
        
        # Convert replications to DataFrame and plot for this specific combination
        if counterfactuals_df_replications:
            counterfactuals_df_replications = pd.DataFrame(counterfactuals_df_replications)
            
            # Extract only the feature columns for plotting
            feature_cols = [
                col for col in counterfactuals_df_replications.columns
                if not col.startswith('Rule_') and col != 'Replication'
            ]
            cf_features_only = counterfactuals_df_replications[feature_cols]
            
            # Predict classes for counterfactuals (for reuse in plots)
            cf_predicted_classes = MODEL.predict(cf_features_only)
            
            combination_viz['pairwise'] = plot_pairwise_with_counterfactual_df(
                MODEL, IRIS_FEATURES, IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_only
            )
            combination_viz['pca'] = plot_pca_with_counterfactuals(
                MODEL, pd.DataFrame(IRIS_FEATURES), IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_only,
                cf_predicted_classes=cf_predicted_classes
            )
            
            # Add all replications to the overall combinations list
            counterfactuals_df_combinations.extend(counterfactuals_df_replications.to_dict('records'))
        
        if combination_viz['replication']:
            visualizations.append(combination_viz)
    
    # Convert all combinations to DataFrame
    counterfactuals_df_combinations = pd.DataFrame(counterfactuals_df_combinations)
    
    print("\n✓ Processing complete!")
    return counterfactuals_df_combinations, visualizations


if __name__ == "__main__":
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    result_df, result_viz = run_profiled_experiment()
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("PROFILING RESULTS - Sorted by Total Time (cumulative)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Sorted by Function Time (self)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('time')
    ps.print_stats(30)  # Print top 30 functions by self time
    print(s.getvalue())
    
    # Save to file for later analysis
    profile_file = "profile_results.txt"
    with open(profile_file, 'w') as f:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        f.write("PROFILING RESULTS - Sorted by Total Time (cumulative)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('time')
        ps.print_stats(50)
        f.write("\n" + "="*80 + "\n")
        f.write("PROFILING RESULTS - Sorted by Function Time (self)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
    
    print(f"\nProfile results saved to: {profile_file}")
