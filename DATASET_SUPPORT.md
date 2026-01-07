# Dataset Support Summary

The experiment runner now supports multiple datasets with automatic preprocessing.

## Supported Datasets

### 1. Iris Dataset
- **Config**: `configs/experiment_config.yaml`
- **Features**: 4 numerical features
- **Samples**: 150
- **Classes**: 3
- **Usage**: `python scripts/run_experiment.py --config configs/experiment_config.yaml`

### 2. German Credit Dataset
- **Config**: `configs/german_credit_config.yaml`
- **Features**: 20 features (13 categorical, 7 numerical)
- **Samples**: 1000
- **Classes**: 2 (default: 0/1)
- **Usage**: `python scripts/run_experiment.py --config configs/german_credit_config.yaml`

## Key Features

1. **Automatic Categorical Encoding**: Categorical features are automatically detected and label-encoded
2. **Flexible Configuration**: Each dataset has its own config file with optimized parameters
3. **Model Performance Logging**: Train/test accuracy automatically logged to WandB
4. **Extensible**: Easy to add new datasets by creating a new config file

## Running Experiments

```bash
# Iris dataset
.venv/bin/python scripts/run_experiment.py --config configs/experiment_config.yaml

# German Credit dataset
.venv/bin/python scripts/run_experiment.py --config configs/german_credit_config.yaml

# With parameter overrides
.venv/bin/python scripts/run_experiment.py \
  --config configs/german_credit_config.yaml \
  --set experiment_params.num_samples=20 \
  --set counterfactual.population_size=50
```

## Adding New Datasets

To add a new dataset, update the `load_dataset()` function in `scripts/run_experiment.py`:

1. Add a new `elif` branch for your dataset name
2. Load the data (CSV, sklearn, etc.)
3. Handle categorical encoding if needed
4. Return the standardized dict with keys: `features`, `labels`, `feature_names`, `features_df`, `label_encoders`
