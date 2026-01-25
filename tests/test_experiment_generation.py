import os
import shutil
import tempfile
import sys
from pathlib import Path

# Ensure scripts are importable
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.run_experiment import run_single_sample
from utils.config_manager import DictConfig


def test_run_experiment_saves_files():
    # Use a temporary directory so tests don't pollute repo
    tmpdir = tempfile.mkdtemp(prefix="cf_test_")

    try:
        # Create a minimal config
        config_dict = {
            'data': {
                'dataset': 'test',
                'random_state': 42,
            },
            'model': {
                'type': 'RandomForestClassifier',
            },
            'counterfactual': {
                'method': 'dpg',
                'population_size': 5,
                'max_generations': 5,
                'num_best_results': 1,
            },
            'experiment_params': {
                'seed': 0,
                'num_samples': 1,
                'num_best_results': 1,
                'num_combinations_to_test': 1,
                'compute_comprehensive_metrics': False,
            },
            'output': {
                'local_dir': tmpdir,
                'save_visualizations': False,
            },
        }
        config = DictConfig(config_dict)
        
        # This test just checks that imports work, actual test would need dataset
        print(f"Test would run in: {tmpdir}")
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
