"""Utilities for storing and loading experiment results."""
import os
import pickle
from datetime import datetime


def get_sample_id(sample_index):
    """Generate a consistent ID for a sample based on its dataset index.

    Args:
        sample_index: The original index of the sample in the dataset

    Returns:
        The sample index as the ID (ensures same sample always gets same ID)
    """
    return int(sample_index)


def _get_sample_dir(sample_id, output_dir="experiment_results"):
    """Return the directory path for a given sample id."""
    sample_dir = os.path.join(output_dir, str(sample_id))
    return sample_dir


def save_sample_metadata(sample_id, original_sample, predicted_class, target_class,
                         sample_index=None, output_dir="experiment_results"):
    """Save metadata about the sample into a per-sample folder.

    New layout:
        experiment_results/<sample_id>/metadata.pkl

    Args:
        sample_id: Unique identifier for the sample
        original_sample: Dictionary containing the original sample features
        predicted_class: The class predicted by the model
        target_class: The target class for counterfactual generation
        sample_index: Optional original index in the dataset
        output_dir: Base directory to save results (default: "experiment_results")

    Returns:
        Path to the saved metadata file
    """
    sample_dir = _get_sample_dir(sample_id, output_dir)
    os.makedirs(sample_dir, exist_ok=True)

    metadata = {
        'sample_id': sample_id,
        'original_sample': original_sample,
        'predicted_class': predicted_class,
        'target_class': target_class,
        'sample_index': sample_index,
        'timestamp': datetime.now().isoformat()
    }
    filepath = os.path.join(sample_dir, "metadata.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved sample metadata to {filepath}")
    return filepath


def save_visualizations_data(sample_id, visualizations, original_sample, constraints,
                             features_names, target_class, output_dir="experiment_results"):
    """Save the visualizations data structure to disk under the sample directory.

    New layout:
        experiment_results/<sample_id>/after_viz_generation.pkl

    Args:
        sample_id: Unique identifier for the sample
        visualizations: List of visualization objects
        original_sample: Dictionary containing the original sample features
        constraints: Constraints dictionary
        features_names: List of feature names
        target_class: The target class for counterfactual generation
        output_dir: Base directory to save results (default: "experiment_results")

    Returns:
        Path to the saved visualizations file
    """
    sample_dir = _get_sample_dir(sample_id, output_dir)
    os.makedirs(sample_dir, exist_ok=True)

    data = {
        'sample_id': sample_id,
        'original_sample': original_sample,
        'visualizations': visualizations,
        'constraints': constraints,
        'features_names': features_names,
        'target_class': target_class,
        'timestamp': datetime.now().isoformat()
    }
    filepath = os.path.join(sample_dir, "after_viz_generation.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved visualizations data to {filepath}")
    return filepath


def load_visualizations_data(sample_id, output_dir="experiment_results"):
    """Load the visualizations data structure from disk (from the sample folder).

    Args:
        sample_id: Unique identifier for the sample
        output_dir: Base directory where results are stored (default: "experiment_results")

    Returns:
        Dictionary containing the visualizations data

    Raises:
        FileNotFoundError: If no visualizations data is found for the given sample_id
    """
    sample_dir = _get_sample_dir(sample_id, output_dir)
    filepath = os.path.join(sample_dir, "after_viz_generation.pkl")

    # Fallback: maintain backwards compatibility with older root-level filenames
    if not os.path.exists(filepath):
        alt_filepath = os.path.join(output_dir, f"sample_{sample_id}_visualizations.pkl")
        if os.path.exists(alt_filepath):
            filepath = alt_filepath

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No visualizations data found for sample {sample_id}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded visualizations data from {filepath}")
    return data


def list_available_samples(output_dir="experiment_results"):
    """List all available sample IDs that have been processed.

    New layout expects per-sample directories: experiment_results/<sample_id>/metadata.pkl

    Args:
        output_dir: Directory where results are stored (default: "experiment_results")

    Returns:
        Dictionary mapping sample_ids to their metadata
    """
    if not os.path.exists(output_dir):
        return {}

    samples = {}

    # First, check for per-sample directories (new layout)
    for name in os.listdir(output_dir):
        sample_dir = os.path.join(output_dir, name)
        if os.path.isdir(sample_dir) and name.isdigit():
            metadata_path = os.path.join(sample_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                samples[int(name)] = metadata

    # Fallback: old flat files at root (backwards compatibility)
    for filename in os.listdir(output_dir):
        if filename.startswith("sample_") and filename.endswith("_metadata.pkl"):
            try:
                sample_id = int(filename.split("_")[1])
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'rb') as f:
                    metadata = pickle.load(f)
                samples[sample_id] = metadata
            except Exception:
                continue

    return dict(sorted(samples.items()))
