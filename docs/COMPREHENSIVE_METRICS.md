# Comprehensive Metrics Implementation

This document describes the comprehensive counterfactual evaluation metrics that have been integrated into the CounterFactualDPG repository.

## Overview

The implementation adds **70+ evaluation metrics** for counterfactual explanations, matching the metrics used in the ECE (Exemplar Counterfactual Explanations) benchmark. These metrics are now fully integrated with:

- ✅ Per-replication metrics logging
- ✅ Per-combination aggregated metrics
- ✅ Per-sample and experiment-level summaries
- ✅ WandB real-time logging
- ✅ CSV persistence for offline analysis
- ✅ Automatic feature type detection

## Files Created/Modified

### New Files

1. **`CounterFactualMetrics.py`** (1,200+ lines)
   - Standalone metrics module extracted from ECE/cf_eval/metrics.py
   - All metrics reimplemented without ECE dependencies
   - Main function: `evaluate_cf_list()` - computes all 70+ metrics

### Modified Files

1. **`CounterFactualExplainer.py`**
   - Extended `get_all_metrics()` method with optional comprehensive metrics
   - Now accepts training/test data, feature indices for advanced computations
   - Backwards compatible with existing usage

2. **`scripts/run_experiment.py`**
   - Added `determine_feature_types()` helper function
   - Updated `load_dataset()` to return feature type information
   - Modified `run_single_sample()` to compute comprehensive metrics
   - Added metrics persistence (CSV files)
   - Enhanced WandB logging with all new metrics
   - Added experiment-level aggregation and summary statistics

3. **`configs/iris.yaml`** & **`configs/german_credit.yaml`**
   - Added feature type specifications (optional)
   - Added `compute_comprehensive_metrics` flag
   - Added comments explaining configuration options

## Metrics Categories

### 1. Validity Metrics (2 metrics)
- `nbr_valid_cf`: Number of valid counterfactuals
- `perc_valid_cf`: Percentage of valid counterfactuals

### 2. Actionability Metrics (6 metrics)
- `nbr_actionable_cf`: Number of actionable CFs
- `perc_actionable_cf`: Percentage of actionable CFs
- `nbr_valid_actionable_cf`: Valid AND actionable CFs
- `perc_valid_actionable_cf`: Percentage valid AND actionable
- `avg_nbr_violations_per_cf`: Average constraint violations per CF
- `avg_nbr_violations`: Normalized violations

### 3. Distance Metrics (18 metrics)
Per metric type (mean, min, max):
- `distance_l2`: Euclidean distance (continuous features)
- `distance_mad`: MAD-normalized distance (continuous)
- `distance_j`: Jaccard distance (categorical features)
- `distance_h`: Hamming distance (categorical)
- `distance_l2j`: Combined L2 + Jaccard
- `distance_mh`: Combined MAD + Hamming

### 4. Sparsity Metrics (2 metrics)
- `avg_nbr_changes_per_cf`: Average feature changes per CF
- `avg_nbr_changes`: Normalized by total features

### 5. Diversity Metrics (18 metrics)
Pairwise diversity among CFs (mean, min, max):
- `diversity_l2`: Euclidean diversity
- `diversity_mad`: MAD-normalized diversity
- `diversity_j`: Jaccard diversity
- `diversity_h`: Hamming diversity
- `diversity_l2j`: Combined L2 + Jaccard
- `diversity_mh`: Combined MAD + Hamming

### 6. Count Diversity Metrics (3 metrics)
- `count_diversity_cont`: Count-based diversity (continuous)
- `count_diversity_cate`: Count-based diversity (categorical)
- `count_diversity_all`: Count-based diversity (all features)

### 7. Plausibility Metrics (5 metrics)
Distance to nearest training sample of CF's predicted class:
- `plausibility_sum`: Sum of plausibility distances
- `plausibility_max_nbr_cf`: Normalized by max CFs
- `plausibility_nbr_cf`: Normalized by actual CFs
- `plausibility_nbr_valid_cf`: Normalized by valid CFs
- `plausibility_nbr_actionable_cf`: Normalized by actionable CFs

### 8. Model Fidelity Metrics (3 metrics)
- `accuracy_knn_sklearn`: KNN accuracy (sklearn)
- `accuracy_knn_dist`: KNN accuracy (distance-based)
- `lof`: Local Outlier Factor

### 9. Delta Probability Metrics (3 metrics)
- `delta`: Mean prediction change
- `delta_min`: Minimum prediction change
- `delta_max`: Maximum prediction change

### 10. Basic Metrics (14 metrics from CounterFactualExplainer)
- All existing metrics from the original implementation
- Fitness scores, class changes, constraint penalties, etc.

**Total: 70+ metrics**

## Usage

### Basic Usage (Backwards Compatible)

```python
from CounterFactualExplainer import CounterFactualExplainer

explainer = CounterFactualExplainer(cf_model, original, counterfactual, target_class)
metrics = explainer.get_all_metrics()  # Returns 14 basic metrics
```

### Comprehensive Metrics

```python
# With comprehensive metrics enabled
metrics = explainer.get_all_metrics(
    X_train=train_data,
    X_test=test_data,
    variable_features=[0, 1, 2, 3],  # Actionable feature indices
    continuous_features=[0, 1, 2, 3],  # Continuous feature indices
    categorical_features=[],  # Categorical feature indices
    compute_comprehensive=True  # Enable comprehensive metrics
)
# Returns 70+ metrics
```

### Configuration

Enable comprehensive metrics in your config YAML:

```yaml
experiment_params:
  compute_comprehensive_metrics: true  # Enable comprehensive ECE-style metrics
  
data:
  # Optional: explicitly specify feature types
  continuous_features: ["feature1", "feature2"]
  categorical_features: ["feature3", "feature4"]
  variable_features: null  # null = all features actionable
```

If feature types are not specified, the system will auto-detect:
- Numeric columns → continuous features
- Non-numeric columns → categorical features
- All features → actionable (variable)

## Output Files

### Per-Sample Files

Each sample generates the following files in `experiment_results/<config_name>/sample_<id>/`:

1. **`replication_metrics.csv`** - Per-replication comprehensive metrics
   - Columns: `sample_id`, `combination_idx`, `combination`, `replication_idx`, + all 70+ metrics
   - One row per counterfactual generated

2. **`combination_metrics.csv`** - Per-combination aggregated metrics
   - Aggregates all replications within each rule combination
   - Includes diversity metrics across multiple CFs

3. **`raw_counterfactuals.pkl`** - Raw counterfactual data (existing)

4. **`after_viz_generation.pkl`** - Visualization data (existing)

5. **Visualization files** (PNG, CSV) - Existing visualization outputs

### Experiment-Level Files

At the end of an experiment, aggregated files are saved to `experiment_results/<config_name>/`:

1. **`all_replication_metrics.csv`** - All replications from all samples
   - Concatenation of all sample replication metrics
   - Enables cross-sample analysis

2. **`all_combination_metrics.csv`** - All combinations from all samples
   - Concatenation of all sample combination metrics

3. **`metrics_summary_statistics.csv`** - Statistical summary
   - Mean, std, min, max, quartiles for all numeric metrics
   - Computed using pandas `.describe()`

4. **`experiment_config.yaml`** - Copy of experiment configuration
   - Full configuration used for reproducibility

## WandB Logging

All metrics are logged to Weights & Biases under organized namespaces:

### Per-Replication Logging
```
replication/sample_id
replication/combination
replication/replication_num
replication/success
replication/final_fitness
replication/generations_to_converge
metrics/<metric_name>  # All 70+ metrics logged here
```

### Per-Combination Logging
```
combo/sample_id
combo/combination
combo_metrics/<metric_name>  # Aggregated metrics for the combination
```

### Sample-Level Logging
```
sample/sample_id
sample/original_class
sample/target_class
sample/num_valid_counterfactuals
sample/total_replications
sample/success_rate
```

### Experiment-Level Logging
```
experiment/total_samples
experiment/total_valid_counterfactuals
experiment/total_replications
experiment/overall_success_rate
experiment/summary_table  # WandB Table with per-sample summary
```

## Feature Type Detection

The system automatically detects feature types from the dataset:

```python
def determine_feature_types(features_df, config=None):
    """
    Auto-detect or read from config:
    - continuous_features: numeric dtypes
    - categorical_features: object/category dtypes
    - variable_features: all features (default)
    """
```

### Manual Override

You can explicitly specify feature types in the config:

```yaml
data:
  continuous_features: ["age", "income", "credit_score"]
  categorical_features: ["gender", "employment_status"]
  variable_features: ["age", "income"]  # Only these can change
```

## Performance Considerations

### Computational Cost

Comprehensive metrics add computational overhead:
- **MAD distance**: Requires median computation on training data
- **KNN metrics**: O(n²) nearest neighbor search
- **LOF**: Local outlier factor computation
- **Plausibility**: Distance to nearest neighbor in test set

### Recommendations

1. **For quick experiments**: Set `compute_comprehensive_metrics: false`
   - Only computes the 14 basic metrics
   - Much faster for large datasets

2. **For thorough evaluation**: Set `compute_comprehensive_metrics: true`
   - Full 70+ metrics for publication-quality results
   - Recommended for final experiments

3. **For large datasets**: Consider subsampling X_test
   - Plausibility and KNN metrics scale with test set size

## Comparison with ECE Metrics

This implementation provides **identical metrics** to the ECE benchmark:

| Metric Category | ECE/cf_eval | CounterFactualMetrics | Status |
|----------------|-------------|----------------------|--------|
| Validity | ✓ | ✓ | ✅ Identical |
| Actionability | ✓ | ✓ | ✅ Identical |
| Distance (L2, MAD, Jaccard, Hamming) | ✓ | ✓ | ✅ Identical |
| Diversity | ✓ | ✓ | ✅ Identical |
| Sparsity | ✓ | ✓ | ✅ Identical |
| Plausibility | ✓ | ✓ | ✅ Identical |
| KNN Accuracy | ✓ | ✓ | ✅ Identical |
| LOF | ✓ | ✓ | ✅ Identical |
| Delta Proba | ✓ | ✓ | ✅ Identical |

## Example Analysis Workflow

### 1. Run Experiment with Comprehensive Metrics

```bash
python scripts/run_experiment.py --config configs/iris.yaml
```

### 2. Load and Analyze Results

```python
import pandas as pd

# Load aggregated replication metrics
df = pd.read_csv('experiment_results/counterfactual_dpg_v1/all_replication_metrics.csv')

# Analyze validity rates
validity_by_combo = df.groupby('combination')['perc_valid_cf'].mean()
print(validity_by_combo)

# Analyze distance metrics
print(df[['distance_l2', 'distance_mad', 'distance_l2j']].describe())

# Analyze diversity
combo_df = pd.read_csv('experiment_results/counterfactual_dpg_v1/all_combination_metrics.csv')
print(combo_df[['diversity_l2', 'diversity_mad']].describe())
```

### 3. Compare with ECE Baseline

```python
# Your CounterFactualDPG results
cfm_results = pd.read_csv('experiment_results/counterfactual_dpg_v1/all_combination_metrics.csv')

# ECE baseline results (from ECE experiments)
ece_results = pd.read_csv('ECE/results/sace_ens_results.csv')

# Compare key metrics
comparison = pd.DataFrame({
    'CounterFactualDPG': cfm_results[['perc_valid_cf', 'distance_l2', 'diversity_l2']].mean(),
    'ECE_SACE': ece_results[['perc_valid_cf', 'distance_l2', 'diversity_l2']].mean()
})
print(comparison)
```

## Troubleshooting

### ImportError: No module named 'scipy'

Comprehensive metrics require scipy. Install with:
```bash
pip install scipy scikit-learn
```

### MAD Distance Returns NaN

MAD distance requires training data (`X_train`). Ensure you pass it:
```python
metrics = explainer.get_all_metrics(X_train=train_data, ...)
```

### KNN Metrics Take Too Long

KNN metrics are O(n²). For large datasets, subsample X_test:
```python
X_test_subsample = X_test[::10]  # Use every 10th sample
metrics = explainer.get_all_metrics(X_test=X_test_subsample, ...)
```

### Feature Type Detection Incorrect

Manually specify feature types in config:
```yaml
data:
  continuous_features: ["feat1", "feat2"]
  categorical_features: ["feat3"]
```

## References

1. **ECE Paper**: Guidotti, R., et al. (2022). "Exemplar Counterfactual Explanations"
2. **Original ECE Implementation**: https://github.com/riccotti/ECE
3. **CounterFactualDPG**: This repository

## Future Work

Potential extensions:
- [ ] Add image/time-series specific metrics
- [ ] Parallel computation for large-scale experiments
- [ ] Metric visualization dashboard
- [ ] Statistical significance testing
- [ ] Metric correlation analysis
