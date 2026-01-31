#!/usr/bin/env python3
"""Quick test to verify dataset loading works for both iris and german_credit."""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from scripts.run_experiment import load_config, load_dataset

def test_iris():
    """Test iris dataset loading."""
    print("Testing Iris dataset loading...")
    config = load_config('configs/experiment_config.yaml')
    dataset_info = load_dataset(config)
    
    print(f"  ✓ Features shape: {dataset_info['features'].shape}")
    print(f"  ✓ Labels shape: {dataset_info['labels'].shape}")
    print(f"  ✓ Feature names ({len(dataset_info['feature_names'])}): {dataset_info['feature_names']}")
    print(f"  ✓ Categorical features encoded: {len(dataset_info['label_encoders'])}")
    print()

def test_german_credit():
    """Test german credit dataset loading."""
    print("Testing German Credit dataset loading...")
    config = load_config('configs/german_credit_config.yaml')
    dataset_info = load_dataset(config)
    
    print(f"  ✓ Features shape: {dataset_info['features'].shape}")
    print(f"  ✓ Labels shape: {dataset_info['labels'].shape}")
    print(f"  ✓ Feature names ({len(dataset_info['feature_names'])}): {dataset_info['feature_names'][:5]}...")
    print(f"  ✓ Categorical features encoded: {len(dataset_info['label_encoders'])}")
    if dataset_info['label_encoders']:
        print(f"  ✓ Encoded features: {list(dataset_info['label_encoders'].keys())[:5]}...")
    print()

if __name__ == '__main__':
    try:
        test_iris()
        test_german_credit()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
