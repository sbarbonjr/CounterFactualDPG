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


def _get_sample_dir(sample_id, output_dir="experiment_results", configname=None):
    """Return the directory path for a given sample id.

    New layout supports: experiment_results/{configname}/{sample_id}
    If `configname` is None, falls back to the legacy layout: experiment_results/{sample_id}
    """
    if configname:
        sample_dir = os.path.join(output_dir, str(configname), str(sample_id))
    else:
        sample_dir = os.path.join(output_dir, str(sample_id))
    return sample_dir


def save_sample_metadata(sample_id, original_sample, predicted_class, target_class,
                         sample_index=None, configname=None, output_dir="experiment_results"):
    """Save metadata about the sample into a per-sample folder.

    New layout:
        experiment_results/{configname}/{sample_id}/metadata.pkl

    Args:
        sample_id: Unique identifier for the sample
        original_sample: Dictionary containing the original sample features
        predicted_class: The class predicted by the model
        target_class: The target class for counterfactual generation
        sample_index: Optional original index in the dataset
        configname: Optional experiment name to group samples under
        output_dir: Base directory to save results (default: "experiment_results")

    Returns:
        Path to the saved metadata file
    """
    sample_dir = _get_sample_dir(sample_id, output_dir=output_dir, configname=configname)
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
                             features_names, target_class, configname=None, output_dir="experiment_results"):
    """Save the visualizations data structure to disk under the sample directory.

    New layout:
        experiment_results/{configname}/{sample_id}/after_viz_generation.pkl

    Args:
        sample_id: Unique identifier for the sample
        visualizations: List of visualization objects
        original_sample: Dictionary containing the original sample features
        constraints: Constraints dictionary
        features_names: List of feature names
        target_class: The target class for counterfactual generation
        configname: Optional experiment name to group samples under
        output_dir: Base directory to save results (default: "experiment_results")

    Returns:
        Path to the saved visualizations file
    """
    sample_dir = _get_sample_dir(sample_id, output_dir=output_dir, configname=configname)
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


def load_visualizations_data(sample_id, output_dir="experiment_results", configname=None):
    """Load the visualizations data structure from disk (from the sample folder).

    Args:
        sample_id: Unique identifier for the sample
        configname: Optional experiment name to group samples under
        output_dir: Base directory where results are stored (default: "experiment_results")

    Returns:
        Dictionary containing the visualizations data

    Raises:
        FileNotFoundError: If no visualizations data is found for the given sample_id
    """
    sample_dir = _get_sample_dir(sample_id, output_dir=output_dir, configname=configname)
    filepath = os.path.join(sample_dir, "after_viz_generation.pkl")

    # Fallback: maintain backwards compatibility with older root-level filenames
    if not os.path.exists(filepath):
        # Check inside configname folder if available
        if configname:
            alt_filepath = os.path.join(output_dir, configname, f"sample_{sample_id}_visualizations.pkl")
            if os.path.exists(alt_filepath):
                filepath = alt_filepath
        # Then fallback to legacy root-level filename
        alt_filepath = os.path.join(output_dir, f"sample_{sample_id}_visualizations.pkl")
        if os.path.exists(alt_filepath):
            filepath = alt_filepath

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No visualizations data found for sample {sample_id}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded visualizations data from {filepath}")
    return data


def list_available_samples(output_dir="experiment_results", configname=None):
    """List all available sample IDs that have been processed.

    New layout expects per-sample directories: experiment_results/{configname}/{sample_id}/metadata.pkl

    Args:
        output_dir: Directory where results are stored (default: "experiment_results")
        configname: Optional experiment name to look under. If None, will search both legacy root-level samples and any configname subdirectories.

    Returns:
        Dictionary mapping sample_ids to their metadata
    """
    if not os.path.exists(output_dir):
        return {}

    samples = {}

    def load_from_dir(base_dir):
        for name in os.listdir(base_dir):
            sample_dir = os.path.join(base_dir, name)
            if os.path.isdir(sample_dir) and name.isdigit():
                metadata_path = os.path.join(sample_dir, 'metadata.pkl')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                        samples[int(name)] = metadata
                    except Exception:
                        continue

    # If configname provided, look specifically under that folder
    if configname:
        conf_dir = os.path.join(output_dir, str(configname))
        if os.path.exists(conf_dir):
            load_from_dir(conf_dir)
        # Also keep legacy check in root for backward compatibility
        load_from_dir(output_dir)
    else:
        # Check for per-sample directories at the root (legacy)
        load_from_dir(output_dir)

        # Also check for configname subdirectories and load their samples
        for name in os.listdir(output_dir):
            conf_dir = os.path.join(output_dir, name)
            if os.path.isdir(conf_dir) and not name.isdigit():
                load_from_dir(conf_dir)

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