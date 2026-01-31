import os
import shutil
import tempfile

from scripts.experiment_generation import run_experiment
from scripts.experiment_visualization import run_visualization


def test_visualization_exports_summary():
    tmpdir = tempfile.mkdtemp(prefix="cf_viz_test_")

    try:
        # Generate a tiny experiment in tmpdir
        result = run_experiment(seed=0, sample_index=2, output_dir=tmpdir, num_replications=1, initial_population_size=5, max_generations=5, verbose=False)
        sample_id = result['sample_id']

        viz_result = run_visualization(sample_id=sample_id, output_dir=tmpdir, export_plots=False, save_summary=True, save_metrics=False, verbose=False)

        assert 'summary_csv' in viz_result
        assert os.path.exists(viz_result['summary_csv'])

        # Load summary and assert it contains rows
        import pandas as pd
        df = pd.read_csv(viz_result['summary_csv'])
        assert not df.empty

        # Request plot export *index* (do not save PNGs)
        viz_result_idx = run_visualization(sample_id=sample_id, output_dir=tmpdir, export_plots=True, save_plots=False, save_summary=False, save_metrics=False, verbose=False)
        sample_dir = os.path.join(tmpdir, str(sample_id))
        index_path = os.path.join(sample_dir, 'plots_index.json')
        assert os.path.exists(index_path)
        # The index should list available plots
        import json
        with open(index_path, 'r') as f:
            idx = json.load(f)
        assert 'plots' in idx

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
