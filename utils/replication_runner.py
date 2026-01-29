"""Counterfactual generation worker functions.

This module provides worker functions for running DPG and DiCE counterfactual
generation, supporting both methods with unified interface.
"""

import traceback
import pandas as pd
from typing import Dict, Optional

try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False

from CounterFactualModel import CounterFactualModel
from utils.config_manager import DictConfig


def run_counterfactual_generation_dpg(args):
    """Run DPG counterfactual generation to produce requested counterfactuals.
    
    Args:
        args: Tuple containing (ORIGINAL_SAMPLE, TARGET_CLASS, 
              FEATURES_NAMES, dict_non_actionable, config_dict, model, constraints,
              train_df, continuous_features, categorical_features)
    
    Returns:
        Dict with generation results or None if failed
    """
    (
        ORIGINAL_SAMPLE,
        TARGET_CLASS,
        FEATURES_NAMES,
        dict_non_actionable,
        config_dict,
        model,
        constraints,
        train_df,  # Training data for nearest neighbor fallback
        _,  # continuous_features (not used for DPG)
        _,  # categorical_features (not used for DPG)
    ) = args
    
    # Reconstruct config from dict
    config = DictConfig(config_dict)
    
    # Extract X_train and y_train from train_df for nearest neighbor fallback
    X_train = None
    y_train = None
    if train_df is not None:
        # train_df includes target column ('_target_' for DiCE compatibility)
        target_col = '_target_' if '_target_' in train_df.columns else (
            'target' if 'target' in train_df.columns else train_df.columns[-1]
        )
        if target_col in train_df.columns:
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
    
    try:
        # Get verbose mode from config if available
        # Check multiple possible locations: top-level, experiment, methods.dpg, counterfactual
        verbose_mode = (
            getattr(config, 'verbose', False) or 
            getattr(config.experiment, 'verbose', False) or
            getattr(getattr(config, 'methods', type('', (), {'dpg': type('', (), {'verbose': False})()})), 'dpg', type('', (), {'verbose': False})()).verbose or
            getattr(config.counterfactual, 'verbose', False)
        )
        
        if verbose_mode:
            print("[VERBOSE-DPG] Verbose mode enabled for counterfactual generation")
        
        # Create CF model with config parameters (including dual-boundary parameters)
        cf_model = CounterFactualModel(
            model, 
            constraints,
            dict_non_actionable=dict_non_actionable,
            verbose=verbose_mode,
            diversity_weight=config.counterfactual.diversity_weight,
            repulsion_weight=config.counterfactual.repulsion_weight,
            boundary_weight=config.counterfactual.boundary_weight,
            distance_factor=config.counterfactual.distance_factor,
            sparsity_factor=config.counterfactual.sparsity_factor,
            constraints_factor=config.counterfactual.constraints_factor,
            # Dual-boundary parameters
            original_escape_weight=getattr(config.counterfactual, 'original_escape_weight', 2.0),
            escape_pressure=getattr(config.counterfactual, 'escape_pressure', 0.5),
            prioritize_non_overlapping=getattr(config.counterfactual, 'prioritize_non_overlapping', True),
            # Fitness calculation parameters
            max_bonus_cap=getattr(config.counterfactual, 'max_bonus_cap', 50.0),
            # Training data for nearest neighbor fallback
            X_train=X_train,
            y_train=y_train,
            # Generation debugging
            generation_debugging=getattr(config.counterfactual, 'generation_debugging', False),
        )
        
        # Get the number of counterfactuals to generate
        requested_counterfactuals = getattr(config.experiment_params, 'requested_counterfactuals', 5)
        
        counterfactuals = cf_model.generate_counterfactual(
            ORIGINAL_SAMPLE, 
            TARGET_CLASS, 
            config.counterfactual.population_size,
            config.counterfactual.max_generations,
            mutation_rate=config.counterfactual.mutation_rate,
            num_best_results=requested_counterfactuals
        )
        
        # Handle list return from generate_counterfactual
        if counterfactuals is None or len(counterfactuals) == 0:
            return None
        
        # First counterfactual is the best one
        counterfactual = counterfactuals[0]
        
        if counterfactual is None:
            return None
        
        # Ensure evolution_history includes final counterfactual for visualization
        if counterfactual is not None:
            try:
                if not getattr(cf_model, 'evolution_history', []):
                    cf_model.evolution_history = [counterfactual.copy() if isinstance(counterfactual, dict) else dict(counterfactual)]
            except Exception:
                pass
        
        # Extract data needed for later processing
        evolution_history = getattr(cf_model, 'evolution_history', [])
        best_fitness_list = getattr(cf_model, 'best_fitness_list', [])
        average_fitness_list = getattr(cf_model, 'average_fitness_list', [])
        std_fitness_list = getattr(cf_model, 'std_fitness_list', [])
        generation_debug_table = getattr(cf_model, 'generation_debug_table', [])
        
        # Get per-CF evolution histories (each CF gets its own history path)
        per_cf_evolution_histories = getattr(cf_model, 'per_cf_evolution_histories', None)
        # Fallback: if per-CF histories not available, create list of shared history for each CF
        if per_cf_evolution_histories is None or len(per_cf_evolution_histories) == 0:
            per_cf_evolution_histories = [evolution_history] * len(counterfactuals)
        
        # Get generation_found info for each CF
        cf_generation_found = getattr(cf_model, 'cf_generation_found', None)
        # Fallback: if not available, set to None for each CF
        if cf_generation_found is None or len(cf_generation_found) == 0:
            cf_generation_found = [None] * len(counterfactuals)
        
        print(f"DEBUG counterfactual_runner: Generated {len(counterfactuals)} counterfactuals, best_fitness_list length = {len(best_fitness_list)}, avg_fitness_list length = {len(average_fitness_list)}, evolution_history length = {len(evolution_history)}")
        
        return {
            'counterfactual': counterfactual,
            'all_counterfactuals': counterfactuals,  # All requested counterfactuals from single GA run
            'evolution_history': evolution_history,  # Shared best-per-generation history
            'per_cf_evolution_histories': per_cf_evolution_histories,  # Per-CF evolution paths
            'cf_generation_found': cf_generation_found,  # Generation where each CF was found
            'best_fitness_list': best_fitness_list,
            'average_fitness_list': average_fitness_list,
            'std_fitness_list': std_fitness_list,
            'generation_debug_table': generation_debug_table,  # Per-generation fitness component breakdown
            'dict_non_actionable': dict_non_actionable,
            'method': 'dpg',
            # Store serializable versions of model properties needed later
            'constraints': constraints,
            'feature_names': getattr(cf_model, 'feature_names', None),
            'diversity_weight': cf_model.diversity_weight,
            'repulsion_weight': cf_model.repulsion_weight,
            'boundary_weight': cf_model.boundary_weight,
            'distance_factor': cf_model.distance_factor,
            'sparsity_factor': cf_model.sparsity_factor,
            'constraints_factor': cf_model.constraints_factor,
            # Dual-boundary parameters
            'original_escape_weight': cf_model.original_escape_weight,
            'escape_pressure': cf_model.escape_pressure,
            'prioritize_non_overlapping': cf_model.prioritize_non_overlapping,
        }
        
    except Exception as exc:
        print(f"WARNING: DPG counterfactual generation failed: {exc}")
        traceback.print_exc()
        return None


def run_counterfactual_generation_dice(args):
    """Run DiCE counterfactual generation to produce requested counterfactuals.
    
    Args:
        args: Tuple containing (ORIGINAL_SAMPLE, TARGET_CLASS, 
              FEATURES_NAMES, dict_non_actionable, config_dict, model, constraints,
              train_df, continuous_features, categorical_features)
    
    Returns:
        Dict with generation results or None if failed
    """
    if not DICE_AVAILABLE:
        print("ERROR: dice-ml not available. Install with: pip install dice-ml")
        return None
    
    (
        ORIGINAL_SAMPLE,
        TARGET_CLASS,
        FEATURES_NAMES,
        dict_non_actionable,
        config_dict,
        model,
        constraints,
        train_df,
        continuous_features,
        categorical_features,
    ) = args
    
    # Reconstruct config from dict
    config = DictConfig(config_dict)
    
    try:
        # Build features_to_vary from actionability rules
        features_to_vary = []
        for feat in FEATURES_NAMES:
            rule = dict_non_actionable.get(feat, "none")
            if rule != "no_change":
                features_to_vary.append(feat)
        
        # If all features are actionable, use 'all'
        if len(features_to_vary) == len(FEATURES_NAMES):
            features_to_vary = 'all'
        elif len(features_to_vary) == 0:
            print(f"WARNING: DiCE generation: No actionable features!")
            return None
        
        # Build permitted_range from config if specified
        permitted_range = {}
        if hasattr(config.counterfactual, 'permitted_range') and config.counterfactual.permitted_range:
            permitted_range_config = config.counterfactual.permitted_range
            if hasattr(permitted_range_config, '_config'):
                permitted_range = permitted_range_config._config
            elif hasattr(permitted_range_config, 'to_dict'):
                permitted_range = permitted_range_config.to_dict()
            else:
                permitted_range = dict(permitted_range_config) if permitted_range_config else {}
        
        # Build feature_weights from config if specified, otherwise use equal weights
        # This prevents DICE's "invalid value encountered in divide" warning when sum(feature_weights) is 0
        feature_weights = {}
        if hasattr(config.counterfactual, 'feature_weights') and config.counterfactual.feature_weights:
            feature_weights_config = config.counterfactual.feature_weights
            if hasattr(feature_weights_config, '_config'):
                feature_weights = feature_weights_config._config
            elif hasattr(feature_weights_config, 'to_dict'):
                feature_weights = feature_weights_config.to_dict()
            else:
                feature_weights = dict(feature_weights_config) if feature_weights_config else {}
        
        # If no feature_weights specified, use equal weights for all features
        # This prevents DICE from using internal zero weights that cause division by zero
        if not feature_weights:
            feature_weights = {feat: 1.0 for feat in FEATURES_NAMES}
        
        # Create DiCE data interface
        # DiCE needs a DataFrame with the outcome column
        outcome_name = '_target_'
        train_df_with_target = train_df.copy()
        
        # Get continuous and categorical feature names
        continuous_feature_names = [FEATURES_NAMES[i] for i in continuous_features] if continuous_features else []
        categorical_feature_names = [FEATURES_NAMES[i] for i in categorical_features] if categorical_features else []
        
        # Ensure all continuous features are float type for DiCE compatibility
        for feat in continuous_feature_names:
            if feat in train_df_with_target.columns:
                train_df_with_target[feat] = train_df_with_target[feat].astype(float)
        
        # Fix DiCE precision detection bug: When the mode of a continuous feature is an integer 
        # (e.g., 0, 1, -1), DiCE's get_decimal_precisions() fails with IndexError because
        # str(mode).split('.') returns only one element when there's no decimal point.
        # Solution: Explicitly provide continuous_features_precision to bypass auto-detection.
        # Use precision of 4 decimal places as a reasonable default for all continuous features.
        continuous_features_precision = {feat: 4 for feat in continuous_feature_names}
        
        # Create DiCE Data object with explicit precision to avoid the bug
        d = dice_ml.Data(
            dataframe=train_df_with_target,
            continuous_features=continuous_feature_names,
            outcome_name=outcome_name,
            continuous_features_precision=continuous_features_precision
        )
        
        # Create DiCE Model object
        m = dice_ml.Model(model=model, backend='sklearn')
        
        # Get the number of counterfactuals to generate
        requested_counterfactuals = getattr(config.experiment_params, 'requested_counterfactuals', 5)
        
        # Get DiCE parameters from config
        proximity_weight = getattr(config.counterfactual, 'proximity_weight', 0.5)
        diversity_weight = getattr(config.counterfactual, 'diversity_weight', 1.0)
        generation_method = getattr(config.counterfactual, 'generation_method', 'genetic')
        # Additional genetic algorithm parameters for performance tuning
        maxiterations = getattr(config.counterfactual, 'maxiterations', 500)
        posthoc_sparsity_algorithm = getattr(config.counterfactual, 'posthoc_sparsity_algorithm', 'linear')
        verbose_dice = getattr(config.counterfactual, 'verbose_dice', False)
        
        # Create DiCE explainer with specified method
        exp = dice_ml.Dice(d, m, method=generation_method)
        
        # Prepare query instance as DataFrame
        query_df = pd.DataFrame([ORIGINAL_SAMPLE])
        
        # Generate counterfactuals
        # Note: For high-dimensional datasets, consider reducing maxiterations or using 'random' method
        dice_exp = exp.generate_counterfactuals(
            query_df,
            total_CFs=requested_counterfactuals,
            desired_class=int(TARGET_CLASS),
            features_to_vary=features_to_vary,
            permitted_range=permitted_range if permitted_range else None,
            proximity_weight=proximity_weight,
            diversity_weight=diversity_weight,
            feature_weights=feature_weights,  # Pass feature_weights to avoid division by zero warning
            maxiterations=maxiterations,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            verbose=verbose_dice,
        )
        
        # Extract results
        if dice_exp.cf_examples_list and len(dice_exp.cf_examples_list) > 0:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or len(cf_df) == 0:
                print(f"WARNING: DiCE generation: No counterfactuals generated")
                return None
            
            # Remove the outcome column if present
            if outcome_name in cf_df.columns:
                cf_df = cf_df.drop(columns=[outcome_name])
            
            # Convert all CFs to list of dicts
            all_counterfactuals = []
            for _, row in cf_df.iterrows():
                cf_dict = {feat: float(row[feat]) for feat in FEATURES_NAMES if feat in row}
                all_counterfactuals.append(cf_dict)
            
            if not all_counterfactuals:
                return None
            
            # Sort counterfactuals by distance to original (closest first)
            # Use L2 distance to order them
            sorted_cfs = []
            for cf in all_counterfactuals:
                # Verify it predicts the target class
                cf_pred = model.predict(pd.DataFrame([cf]))[0]
                if cf_pred == TARGET_CLASS:
                    distance = sum((ORIGINAL_SAMPLE[f] - cf[f])**2 for f in FEATURES_NAMES)**0.5
                    sorted_cfs.append((distance, cf))
            
            # Sort by distance and extract just the counterfactuals
            sorted_cfs.sort(key=lambda x: x[0])
            all_counterfactuals = [cf for _, cf in sorted_cfs]
            
            # If no CF reaches target, use original list
            if not all_counterfactuals:
                all_counterfactuals = []
                for _, row in cf_df.iterrows():
                    cf_dict = {feat: float(row[feat]) for feat in FEATURES_NAMES if feat in row}
                    all_counterfactuals.append(cf_dict)
            
            best_cf = all_counterfactuals[0] if all_counterfactuals else None
            
            print(f"DEBUG counterfactual_runner: DiCE generated {len(all_counterfactuals)} counterfactuals")
            
            return {
                'counterfactual': best_cf,
                'all_counterfactuals': all_counterfactuals,
                'evolution_history': [best_cf] if best_cf else [],  # DiCE doesn't have evolution history
                'best_fitness_list': [],  # DiCE doesn't track fitness
                'average_fitness_list': [],
                'dict_non_actionable': dict_non_actionable,
                'method': 'dice',
                # Store DiCE-specific params
                'constraints': constraints,
                'feature_names': FEATURES_NAMES,
                'requested_counterfactuals': requested_counterfactuals,
                'proximity_weight': proximity_weight,
                'diversity_weight': diversity_weight,
                'generation_method': generation_method,
                # Dummy DPG params for compatibility
                'repulsion_weight': 0.0,
                'boundary_weight': 0.0,
                'distance_factor': 0.0,
                'sparsity_factor': 0.0,
                'constraints_factor': 0.0,
                'original_escape_weight': 0.0,
                'escape_pressure': 0.0,
                'prioritize_non_overlapping': False,
            }
        else:
            print(f"WARNING: DiCE generation: No counterfactuals in result")
            return None
        
    except Exception as exc:
        print(f"WARNING: DiCE counterfactual generation failed: {exc}")
        traceback.print_exc()
        return None


def run_counterfactual_generation(args):
    """Dispatcher for running counterfactual generation - routes to DPG or DiCE.
    
    Args:
        args: Tuple containing generation parameters including method config
    
    Returns:
        Dict with generation results or None if failed
    """
    # Extract config to determine method
    config_dict = args[4]  # config_dict is at index 4 (after removing replication_num)
    config = DictConfig(config_dict)
    method = getattr(config.counterfactual, 'method', 'dpg').lower()
    
    if method == 'dice':
        return run_counterfactual_generation_dice(args)
    else:
        return run_counterfactual_generation_dpg(args)
