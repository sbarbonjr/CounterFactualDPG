import os
import shutil
import tempfile

from scripts.experiment_generation import run_experiment


def test_run_experiment_saves_files():
    # Use a temporary directory so tests don't pollute repo
    tmpdir = tempfile.mkdtemp(prefix="cf_test_")

    try:
        # Run with reduced settings so it's fast
        result = run_experiment(seed=0, sample_index=1, output_dir=tmpdir, num_replications=1, initial_population_size=5, max_generations=5, verbose=False)

        assert os.path.exists(result['raw_filepath'])
        assert os.path.exists(result['viz_filepath'])

        # Check the raw file has expected structure
        import pickle
        with open(result['raw_filepath'], 'rb') as f:
            data = pickle.load(f)

        assert 'sample_id' in data
        assert 'visualizations_structure' in data

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
