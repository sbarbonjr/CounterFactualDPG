"""
Performance profiling script for Visualization Cell
Analyzes the performance of visualization generation for counterfactuals
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for profiling

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

NUMBER_OF_COMBINATIONS_TO_TEST = 10  # Profile first 10 combinations
NUMBER_OF_REPLICATIONS_PER_COMBINATION = 10
INITIAL_POPULATION_SIZE = 20
MAX_GENERATIONS = 60


def generate_counterfactuals_only():
    """Generate counterfactuals without visualizations (for baseline)"""
    
    print("Phase 1: Generating counterfactuals (no visualizations)...")
    visualizations = []
    
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:NUMBER_OF_COMBINATIONS_TO_TEST]):
        print(f"  Combination {combination_num + 1}/{NUMBER_OF_COMBINATIONS_TO_TEST}...")
        
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
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
            
            cf_dpg = CounterFactualModel(MODEL, CONSTRAINTS)
            cf_dpg.dict_non_actionable = dict_non_actionable
            
            counterfactual = cf_dpg.generate_counterfactual(
                ORIGINAL_SAMPLE, TARGET_CLASS, INITIAL_POPULATION_SIZE, MAX_GENERATIONS
            )
            
            if counterfactual is None:
                if replication == 2:
                    skip_combination = True
                continue
            
            # Store counterfactual and model in replication_viz object
            replication_viz = {
                'counterfactual': counterfactual,
                'cf_model': cf_dpg,  # Store the model so we can access fitness data later
                'visualizations': []
            }
            combination_viz['replication'].append(replication_viz)
        
        if combination_viz['replication']:
            visualizations.append(combination_viz)
    
    print(f"✓ Generated {len(visualizations)} combinations with counterfactuals\n")
    return visualizations


def run_visualization_profiling(visualizations):
    """Run visualization generation with profiling enabled"""
    
    print(f"Phase 2: Generating visualizations for {len(visualizations)} combinations...")
    
    # Iterate over all combinations and generate visualizations
    for combination_idx, combination_viz in enumerate(visualizations):
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination_viz['label']))
        
        print(f"\n  Combination {combination_idx + 1}/{len(visualizations)}: {dict_non_actionable}")
        
        # Generate visualizations for each replication
        for replication_idx, replication_viz in enumerate(combination_viz['replication']):
            counterfactual = replication_viz['counterfactual']
            cf_dpg = replication_viz['cf_model']  # Use the stored model instead of regenerating
            
            print(f"    Replication {replication_idx + 1}/{len(combination_viz['replication'])}...", end=" ")
            
            # Generate replication visualizations
            replication_visualizations = [
                plot_sample_and_counterfactual_heatmap(
                    ORIGINAL_SAMPLE, ORIGINAL_SAMPLE_PREDICTED_CLASS, counterfactual,
                    MODEL.predict(pd.DataFrame([counterfactual])), dict_non_actionable
                ),
                plot_sample_and_counterfactual_comparison(
                    MODEL, ORIGINAL_SAMPLE, SAMPLE_DATAFRAME, counterfactual, CLASS_COLORS_LIST
                ),
                CounterFactualVisualizer.plot_fitness(cf_dpg)  # Use the stored model's fitness data
            ]
            
            # Store visualizations in the replication object
            replication_viz['visualizations'] = replication_visualizations
            
            # Generate explainer summary
            EXPLAINER = CounterFactualExplainer(cf_dpg, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
            plot_explainer_summary(EXPLAINER, ORIGINAL_SAMPLE, counterfactual)
            
            print("Done")
        
        # Generate combination-level visualizations (PCA and Pairwise)
        print(f"    Generating combination-level plots...", end=" ")
        
        # Extract all counterfactuals for this combination
        counterfactuals_list = [rep['counterfactual'] for rep in combination_viz['replication']]
        cf_features_df = pd.DataFrame(counterfactuals_list)
        
        # Predict classes for counterfactuals (for reuse in plots)
        cf_predicted_classes = MODEL.predict(cf_features_df)
        
        combination_viz['pairwise'] = plot_pairwise_with_counterfactual_df(
            MODEL, IRIS_FEATURES, IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_df
        )
        combination_viz['pca'] = plot_pca_with_counterfactuals(
            MODEL, pd.DataFrame(IRIS_FEATURES), IRIS_LABELS, ORIGINAL_SAMPLE, cf_features_df,
            cf_predicted_classes=cf_predicted_classes
        )
        
        print("Done")
    
    print("\n✓ Visualization generation complete!")
    return visualizations


if __name__ == "__main__":
    # First generate counterfactuals (not profiled)
    visualizations = generate_counterfactuals_only()
    
    # Now profile just the visualization generation
    print("="*80)
    print("Starting profiling of visualization generation...")
    print("="*80 + "\n")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result_viz = run_visualization_profiling(visualizations)
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("VISUALIZATION PROFILING RESULTS - Sorted by Total Time (cumulative)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)  # Print top 40 functions
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("VISUALIZATION PROFILING RESULTS - Sorted by Function Time (self)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('time')
    ps.print_stats(40)  # Print top 40 functions by self time
    print(s.getvalue())
    
    # Print function-specific breakdown
    print("\n" + "="*80)
    print("BREAKDOWN BY VISUALIZATION FUNCTION")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Filter for specific visualization functions
    viz_functions = [
        'plot_sample_and_counterfactual_heatmap',
        'plot_sample_and_counterfactual_comparison',
        'plot_fitness',
        'plot_explainer_summary',
        'plot_pairwise_with_counterfactual_df',
        'plot_pca_with_counterfactuals'
    ]
    
    for func_name in viz_functions:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.print_stats(func_name)
        output = s.getvalue()
        if output.strip() and 'function calls' in output:
            print(f"\n{func_name}:")
            print("-" * 40)
            print(output)
    
    # Save to file for later analysis
    profile_file = "profile_visualization_results.txt"
    with open(profile_file, 'w') as f:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(60)
        f.write("VISUALIZATION PROFILING RESULTS - Sorted by Total Time (cumulative)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('time')
        ps.print_stats(60)
        f.write("\n" + "="*80 + "\n")
        f.write("VISUALIZATION PROFILING RESULTS - Sorted by Function Time (self)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
        
        # Add function-specific breakdown
        f.write("\n" + "="*80 + "\n")
        f.write("BREAKDOWN BY VISUALIZATION FUNCTION\n")
        f.write("="*80 + "\n\n")
        
        for func_name in viz_functions:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.print_stats(func_name)
            output = s.getvalue()
            if output.strip() and 'function calls' in output:
                f.write(f"\n{func_name}:\n")
                f.write("-" * 40 + "\n")
                f.write(output + "\n")
    
    print(f"\nProfile results saved to: {profile_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_combinations = len(visualizations)
    total_replications = sum(len(viz['replication']) for viz in visualizations)
    
    print(f"\nTotal combinations processed: {total_combinations}")
    print(f"Total replications processed: {total_replications}")
    print(f"Total visualizations generated: {total_replications * 4 + total_combinations * 2}")
    print(f"  - Heatmaps: {total_replications}")
    print(f"  - Comparison plots: {total_replications}")
    print(f"  - Fitness plots: {total_replications}")
    print(f"  - Explainer summaries: {total_replications}")
    print(f"  - PCA plots: {total_combinations}")
    print(f"  - Pairwise plots: {total_combinations}")
