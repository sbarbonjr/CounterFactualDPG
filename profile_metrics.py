"""
Performance profiling script for Metrics Cell
Analyzes the performance of metrics/explainer generation for counterfactuals
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
from CounterFactualVisualizer import plot_explainer_summary
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

TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
    IRIS_FEATURES, IRIS_LABELS, test_size=0.3, random_state=42
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


def generate_and_prepare_data():
    """Generate counterfactuals without metrics (for baseline)"""
    
    print("Phase 1: Generating counterfactuals (no metrics)...")
    visualizations = []
    
    for combination_num, combination in enumerate(RULES_COMBINATIONS[:NUMBER_OF_COMBINATIONS_TO_TEST]):
        print(f"  Combination {combination_num + 1}/{NUMBER_OF_COMBINATIONS_TO_TEST}...")
        
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination))
        combination_viz = {
            'label': combination,
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
            
            # Store counterfactual and model
            replication_viz = {
                'counterfactual': counterfactual,
                'cf_model': cf_dpg
            }
            combination_viz['replication'].append(replication_viz)
        
        if combination_viz['replication']:
            visualizations.append(combination_viz)
    
    print(f"✓ Generated {len(visualizations)} combinations with counterfactuals\n")
    return visualizations


def run_metrics_profiling(visualizations):
    """Run metrics generation with profiling enabled"""
    
    print(f"Phase 2: Generating metrics for {len(visualizations)} combinations...")
    
    # Iterate over all combinations and generate metrics/explainers
    for combination_idx, combination_viz in enumerate(visualizations):
        dict_non_actionable = dict(zip(FEATURES_NAMES, combination_viz['label']))
        
        print(f"\n  Combination {combination_idx + 1}/{len(visualizations)}: {dict_non_actionable}")
        
        # Generate metrics for each replication
        for replication_idx, replication_viz in enumerate(combination_viz['replication']):
            counterfactual = replication_viz['counterfactual']
            cf_dpg = replication_viz['cf_model']
            
            print(f"    Replication {replication_idx + 1}/{len(combination_viz['replication'])}...", end=" ")
            
            # Generate explainer
            EXPLAINER = CounterFactualExplainer(cf_dpg, ORIGINAL_SAMPLE, counterfactual, TARGET_CLASS)
            plot_explainer_summary(EXPLAINER, ORIGINAL_SAMPLE, counterfactual)
            
            print("Done")
    
    print("\n✓ Metrics generation complete!")
    return visualizations


if __name__ == "__main__":
    # First generate counterfactuals (not profiled)
    visualizations = generate_and_prepare_data()
    
    # Now profile just the metrics generation
    print("="*80)
    print("Starting profiling of metrics generation...")
    print("="*80 + "\n")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result_viz = run_metrics_profiling(visualizations)
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("METRICS PROFILING RESULTS - Sorted by Total Time (cumulative)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)  # Print top 40 functions
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("METRICS PROFILING RESULTS - Sorted by Function Time (self)")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('time')
    ps.print_stats(40)  # Print top 40 functions by self time
    print(s.getvalue())
    
    # Print function-specific breakdown
    print("\n" + "="*80)
    print("BREAKDOWN BY METRICS/EXPLAINER FUNCTION")
    print("="*80 + "\n")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Filter for specific metrics functions
    metrics_functions = [
        'CounterFactualExplainer',
        'plot_explainer_summary',
        '__init__',
        'compute_',
        'extract_'
    ]
    
    for func_name in metrics_functions:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.print_stats(func_name)
        output = s.getvalue()
        if output.strip() and 'function calls' in output:
            print(f"\n{func_name}:")
            print("-" * 40)
            print(output)
    
    # Save to file for later analysis
    profile_file = "profile_metrics_results.txt"
    with open(profile_file, 'w') as f:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(60)
        f.write("METRICS PROFILING RESULTS - Sorted by Total Time (cumulative)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('time')
        ps.print_stats(60)
        f.write("\n" + "="*80 + "\n")
        f.write("METRICS PROFILING RESULTS - Sorted by Function Time (self)\n")
        f.write("="*80 + "\n\n")
        f.write(s.getvalue())
        
        # Add function-specific breakdown
        f.write("\n" + "="*80 + "\n")
        f.write("BREAKDOWN BY METRICS/EXPLAINER FUNCTION\n")
        f.write("="*80 + "\n\n")
        
        for func_name in metrics_functions:
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
    print(f"Total metrics/explainers generated: {total_replications}")
