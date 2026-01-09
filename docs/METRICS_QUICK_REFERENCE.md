# Quick Reference: Comprehensive Metrics

## Metric Groups & Interpretations

### Validity (Higher is Better)
| Metric | Range | Interpretation |
|--------|-------|----------------|
| `nbr_valid_cf` | [0, k] | Number of CFs that changed the prediction |
| `perc_valid_cf` | [0, 1] | Percentage of valid CFs |

**Good values:** >0.8 (80% validity)

---

### Actionability (Higher is Better)
| Metric | Range | Interpretation |
|--------|-------|----------------|
| `nbr_actionable_cf` | [0, k] | CFs that only modify actionable features |
| `perc_actionable_cf` | [0, 1] | Percentage actionable |
| `nbr_valid_actionable_cf` | [0, k] | CFs both valid AND actionable |
| `perc_valid_actionable_cf` | [0, 1] | Percentage valid + actionable |
| `avg_nbr_violations_per_cf` | [0, n_features] | Avg constraint violations per CF |
| `avg_nbr_violations` | [0, 1] | Normalized violations |

**Good values:** `perc_valid_actionable_cf` >0.7, `avg_nbr_violations` <0.1

---

### Distance (Lower is Better)
Measures how far CFs are from original sample.

| Metric | Aggregation | Feature Type | Interpretation |
|--------|------------|--------------|----------------|
| `distance_l2` | mean | continuous | Euclidean distance |
| `distance_l2_min` | min | continuous | Closest CF distance |
| `distance_l2_max` | max | continuous | Farthest CF distance |
| `distance_mad` | mean | continuous | MAD-normalized distance |
| `distance_mad_min` | min | continuous | MAD min |
| `distance_mad_max` | max | continuous | MAD max |
| `distance_j` | mean | categorical | Jaccard distance |
| `distance_j_min` | min | categorical | Jaccard min |
| `distance_j_max` | max | categorical | Jaccard max |
| `distance_h` | mean | categorical | Hamming distance |
| `distance_h_min` | min | categorical | Hamming min |
| `distance_h_max` | max | categorical | Hamming max |
| `distance_l2j` | mean | mixed | Combined L2 + Jaccard |
| `distance_l2j_min` | min | mixed | Combined min |
| `distance_l2j_max` | max | mixed | Combined max |
| `distance_mh` | mean | mixed | Combined MAD + Hamming |
| `distance_mh_min` | min | mixed | MAD+Hamming min |
| `distance_mh_max` | max | mixed | MAD+Hamming max |

**Good values:** Lower is better, but not too low (trivial changes)
- `distance_l2`: 0.5-2.0 (dataset dependent)
- `distance_mad`: 1.0-3.0

---

### Sparsity (Lower is Better)
Measures number of feature changes.

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `avg_nbr_changes_per_cf` | [0, n_features] | Avg features changed per CF |
| `avg_nbr_changes` | [0, 1] | Normalized sparsity |

**Good values:** <5 features changed for typical datasets

---

### Diversity (Higher is Better)
Measures variety among multiple CFs.

| Metric | Aggregation | Feature Type | Interpretation |
|--------|------------|--------------|----------------|
| `diversity_l2` | mean | continuous | Avg pairwise Euclidean |
| `diversity_l2_min` | min | continuous | Min diversity |
| `diversity_l2_max` | max | continuous | Max diversity |
| `diversity_mad` | mean | continuous | MAD-normalized diversity |
| `diversity_mad_min` | min | continuous | MAD min |
| `diversity_mad_max` | max | continuous | MAD max |
| `diversity_j` | mean | categorical | Jaccard diversity |
| `diversity_j_min` | min | categorical | Jaccard min |
| `diversity_j_max` | max | categorical | Jaccard max |
| `diversity_h` | mean | categorical | Hamming diversity |
| `diversity_h_min` | min | categorical | Hamming min |
| `diversity_h_max` | max | categorical | Hamming max |
| `diversity_l2j` | mean | mixed | Combined L2 + Jaccard |
| `diversity_l2j_min` | min | mixed | Combined min |
| `diversity_l2j_max` | max | mixed | Combined max |
| `diversity_mh` | mean | mixed | MAD + Hamming |
| `diversity_mh_min` | min | mixed | MAD+Hamming min |
| `diversity_mh_max` | max | mixed | MAD+Hamming max |

**Good values:** Higher diversity = more varied explanations
- `diversity_l2`: >1.0

---

### Count Diversity (Higher is Better)
Alternative diversity measure based on feature value differences.

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `count_diversity_cont` | [0, 1] | Diversity on continuous features |
| `count_diversity_cate` | [0, 1] | Diversity on categorical features |
| `count_diversity_all` | [0, 1] | Diversity on all features |

**Good values:** >0.3

---

### Plausibility (Lower is Better)
Distance to nearest training sample of CF's predicted class.

| Metric | Normalization | Interpretation |
|--------|--------------|----------------|
| `plausibility_sum` | none | Total plausibility distance |
| `plausibility_max_nbr_cf` | by max CFs | Normalized by k |
| `plausibility_nbr_cf` | by actual CFs | Normalized by n_cfs |
| `plausibility_nbr_valid_cf` | by valid CFs | Normalized by valid |
| `plausibility_nbr_actionable_cf` | by actionable | Normalized by actionable |

**Good values:** Lower = more plausible (closer to real data)
- Dataset-dependent, compare relative to distance metrics

---

### Model Fidelity (Higher is Better)
How well CFs represent local decision boundary.

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `accuracy_knn_sklearn` | [0, 1] | KNN accuracy (sklearn) |
| `accuracy_knn_dist` | [0, 1] | KNN accuracy (distance-based) |
| `lof` | [0, ∞] | Local outlier factor |

**Good values:** 
- `accuracy_knn_*`: >0.7 (good local approximation)
- `lof`: <1.5 (not outliers)

---

### Delta Probability (Higher is Better)
Change in prediction confidence.

| Metric | Aggregation | Interpretation |
|--------|------------|----------------|
| `delta` | mean | Avg prediction change |
| `delta_min` | min | Min prediction change |
| `delta_max` | max | Max prediction change |

**Good values:** Higher = stronger class flips

---

## Usage Examples

### Evaluate Single CF Quality
```python
# Load metrics
df = pd.read_csv('sample_001/replication_metrics.csv')

# Check if CF is good
is_good = (
    df['perc_valid_cf'] > 0.8 and           # Valid
    df['distance_l2'] < 2.0 and             # Not too far
    df['avg_nbr_changes_per_cf'] < 5 and    # Sparse
    df['avg_nbr_violations'] < 0.1          # Respects constraints
)
```

### Compare CF Methods
```python
# Load results from two methods
method1 = pd.read_csv('method1/all_combination_metrics.csv')
method2 = pd.read_csv('method2/all_combination_metrics.csv')

# Compare key metrics
comparison = pd.DataFrame({
    'Method 1': method1[['perc_valid_cf', 'distance_l2', 'diversity_l2', 'plausibility_nbr_cf']].mean(),
    'Method 2': method2[['perc_valid_cf', 'distance_l2', 'diversity_l2', 'plausibility_nbr_cf']].mean(),
})
print(comparison)
```

### Find Best Combination
```python
df = pd.read_csv('all_combination_metrics.csv')

# Define composite score (higher = better)
df['score'] = (
    df['perc_valid_cf'] * 0.3 +           # Validity (30%)
    (1 / (1 + df['distance_l2'])) * 0.3 +  # Proximity (30%)
    df['diversity_l2'] * 0.2 +             # Diversity (20%)
    (1 / (1 + df['plausibility_nbr_cf'])) * 0.2  # Plausibility (20%)
)

best_combination = df.loc[df['score'].idxmax()]
print(f"Best combination: {best_combination['combination']}")
print(f"Score: {best_combination['score']:.3f}")
```

## Metric Selection Guide

### For Publication/Reporting
**Essential metrics:**
- `perc_valid_cf`: Validity
- `distance_l2`: Proximity
- `avg_nbr_changes_per_cf`: Sparsity
- `diversity_l2`: Diversity
- `plausibility_nbr_cf`: Plausibility

### For Development/Debugging
**Diagnostic metrics:**
- `nbr_valid_cf`: How many CFs succeeded?
- `avg_nbr_violations`: Constraint issues?
- `distance_l2_min`: Is closest CF too far?
- `count_diversity_all`: Are CFs diverse enough?

### For User Studies
**Interpretable metrics:**
- `avg_nbr_changes_per_cf`: "On average, X features need to change"
- `distance_l2_min`: "The minimal change required is..."
- `perc_valid_actionable_cf`: "X% of explanations are actionable"

## Common Issues

### All metrics are NaN
- **Cause:** No valid CFs generated
- **Fix:** Check `nbr_valid_cf` and debug CF generation

### MAD metrics are NaN
- **Cause:** `X_train` not provided
- **Fix:** Pass `X_train` to `get_all_metrics()`

### Diversity metrics are 0
- **Cause:** Only 1 CF generated or all CFs identical
- **Fix:** Increase `num_replications` or check diversity weights

### Plausibility very high
- **Cause:** CFs far from training data distribution
- **Fix:** Adjust boundary_weight, check training data quality

### KNN accuracy very low
- **Cause:** CFs don't represent local decision boundary well
- **Fix:** Review CF generation parameters, check model complexity
