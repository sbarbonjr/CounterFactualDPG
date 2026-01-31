"""Configuration management utilities for CounterFactualDPG experiments.

This module provides utilities for loading, merging, and managing experiment configurations
including support for unified configs, method selection, and CLI overrides.
"""

import os
import yaml
from typing import List, Optional


class DictConfig:
    """Simple dict-based config wrapper for dot notation access."""
    
    def __init__(self, config_dict):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Convert integer/string keys to valid attribute names
                attr_key = str(key) if not isinstance(key, str) else key
                setattr(self, attr_key, DictConfig(value))
            else:
                # Convert integer/string keys to valid attribute names
                attr_key = str(key) if not isinstance(key, str) else key
                setattr(self, attr_key, value)
    
    def to_dict(self):
        """Convert config back to plain dict."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        self._config[key] = value
        setattr(self, key, value)
    
    def get(self, key, default=None):
        """Get a value from the config, returning default if not found."""
        return self._config.get(key, default)
    
    def __contains__(self, key):
        """Check if key exists in config."""
        return key in self._config
    
    def keys(self):
        """Return config keys."""
        return self._config.keys()
    
    def items(self):
        """Return config items."""
        return self._config.items()
    
    def values(self):
        """Return config values."""
        return self._config.values()


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary with default values
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, method: str = None, repo_root: str = None) -> DictConfig:
    """Load YAML config file with optional method selection for unified configs.
    
    Supports config inheritance:
    1. Base defaults from configs/config.yaml (if exists)
    2. Dataset-specific config from config_path
    3. Method selection from 'methods' section
    
    Args:
        config_path: Path to config YAML file
        method: Optional method name (dpg, dice, etc.) to select from unified config
        repo_root: Optional repository root path for resolving relative paths
        
    Returns:
        DictConfig with merged configuration
    """
    if repo_root is None:
        # Try to infer repo root from config path
        # config_path is usually configs/<dataset>/config.yaml
        # So we need to go up 3 levels from the file
        abs_config_path = os.path.abspath(config_path)
        # dirname once: configs/<dataset>
        # dirname twice: configs
        # dirname thrice: repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(abs_config_path)))
    
    # Load base defaults from configs/config.yaml if it exists
    base_config_path = os.path.join(repo_root, 'configs', 'config.yaml')
    base_config = {}
    if os.path.exists(base_config_path) and os.path.abspath(config_path) != os.path.abspath(base_config_path):
        try:
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
            print(f"INFO: Loaded base defaults from configs/config.yaml")
        except Exception as e:
            print(f"WARNING: Failed to load base config: {e}")
    
    # Load dataset-specific config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Deep merge: base defaults + dataset config
    if base_config:
        config_dict = deep_merge_dicts(base_config, config_dict)
    
    # Check if this is a unified config (has 'methods' section)
    if 'methods' in config_dict:
        available_methods = [k for k in config_dict['methods'].keys() if k != '_default']
        
        if not method:
            # No method specified - use 'dpg' as default if available, else first method
            method = 'dpg' if 'dpg' in available_methods else available_methods[0]
            print(f"INFO: No method specified, defaulting to '{method}'")
        
        if method not in config_dict['methods']:
            raise ValueError(f"Method '{method}' not found in config. Available: {available_methods}")
        
        # Get method-specific config
        method_config = config_dict['methods'][method]
        
        # Merge _default if present (inside methods)
        defaults = config_dict['methods'].get('_default', {})
        merged_cf = {**defaults, **method_config}
        
        # Preserve existing counterfactual settings (like actionability) and merge with method config
        existing_cf = config_dict.get('counterfactual', {})
        config_dict['counterfactual'] = deep_merge_dicts(existing_cf, merged_cf)
        
        # Ensure method is set in counterfactual
        config_dict['counterfactual']['method'] = method
        
        # Update experiment name to include method
        if 'experiment' not in config_dict:
            config_dict['experiment'] = {}
        base_name = config_dict['experiment'].get('name', config_dict.get('data', {}).get('dataset', 'experiment'))
        config_dict['experiment']['name'] = f"{base_name}_{method}"
        
        # Add method to tags
        if 'tags' not in config_dict['experiment']:
            config_dict['experiment']['tags'] = []
        if method not in config_dict['experiment']['tags']:
            config_dict['experiment']['tags'].append(method)
        
        # Remove 'methods' from final config (including _default)
        config_dict.pop('methods', None)
        
        print(f"INFO: Using unified config with method '{method}'")
    elif method:
        # Legacy config but method was specified - just set it in counterfactual
        if 'counterfactual' not in config_dict:
            config_dict['counterfactual'] = {}
        config_dict['counterfactual']['method'] = method
        print(f"INFO: Using legacy config, setting method to '{method}'")
    
    return DictConfig(config_dict)


def apply_overrides(config: DictConfig, overrides: List[str]) -> DictConfig:
    """Apply CLI overrides to config.
    
    Args:
        config: Base configuration
        overrides: List of "key.subkey=value" strings
    
    Returns:
        Updated config
    """
    import ast
    
    for override in overrides:
        if '=' not in override:
            print(f"WARNING: Invalid override format: {override}. Expected 'key=value'")
            continue
        
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # Try to parse value as Python literal
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Keep as string if can't parse
            value = value_str
        
        # Navigate to the right place in config
        current = config
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                print(f"WARNING: Config key not found: {key_path}")
                break
        else:
            # Set the final value
            final_key = keys[-1]
            if hasattr(current, final_key):
                setattr(current, final_key, value)
                current._config[final_key] = value
                print(f"INFO: Override applied: {key_path} = {value}")
            else:
                print(f"WARNING: Config key not found: {key_path}")
    
    return config


def build_dict_non_actionable(config, feature_names, variable_indices):
    """Build dict_non_actionable from config, supporting per-feature rules.
    
    Args:
        config: Experiment configuration
        feature_names: List of all feature names
        variable_indices: Indices of features that are actionable (from config, legacy support)
        
    Returns:
        dict: Mapping of feature names to actionability rules
              ("none", "non_increasing", "non_decreasing", "no_change")
    """
    dict_non_actionable = {}
    
    # Check for unified actionability config (preferred)
    actionability_config = getattr(config.counterfactual, 'actionability', None)
    
    # Convert DictConfig to regular dict if needed
    if actionability_config is not None:
        if hasattr(actionability_config, '_config'):
            actionability = actionability_config._config
        elif hasattr(actionability_config, 'to_dict'):
            actionability = actionability_config.to_dict()
        else:
            actionability = dict(actionability_config) if actionability_config else {}
    else:
        actionability = None
    
    # Fall back to legacy feature_rules if actionability not set
    if not actionability:
        feature_rules_config = getattr(config.counterfactual, 'feature_rules', None)
        if feature_rules_config is not None:
            if hasattr(feature_rules_config, '_config'):
                actionability = feature_rules_config._config
            elif hasattr(feature_rules_config, 'to_dict'):
                actionability = feature_rules_config.to_dict()
            else:
                actionability = dict(feature_rules_config) if feature_rules_config else {}
    
    # If actionability rules are defined, use them directly
    if actionability:
        for feature_name in feature_names:
            if feature_name in actionability:
                dict_non_actionable[feature_name] = actionability[feature_name]
            else:
                # Default: no restriction
                dict_non_actionable[feature_name] = "none"
    else:
        # Legacy fallback: use variable_indices
        for idx, feature_name in enumerate(feature_names):
            if idx not in variable_indices:
                # Non-variable features are frozen
                dict_non_actionable[feature_name] = "no_change"
            else:
                # Default: no restriction
                dict_non_actionable[feature_name] = "none"
    
    return dict_non_actionable
