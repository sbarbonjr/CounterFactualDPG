"""Replication worker functions for parallel counterfactual generation.

This module provides worker functions for running DPG and DiCE replications
in parallel, supporting both methods with unified interface.
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


def _run_single_replication_dpg(args):
    """Helper function to run a single DPG replication in parallel.
    
    Args:
        args: Tuple containing (replication_num, ORIGINAL_SAMPLE, TARGET_CLASS, 
              FEATURES_NAMES, dict_non_actionable, config_dict, model, constraints,
              train_df, continuous_features, categorical_features)
    
    Returns:
        Dict with replication results or None if failed
    """
    (
        replication_num,
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
        # Create CF model with config parameters (including dual-boundary parameters)
        cf_model = CounterFactualModel(
            model, 
            constraints,
            dict_non_actionable=dict_non_actionable,
            verbose=False,
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
        )
        
        # Enable relaxation fallback for difficult samples
        allow_relaxation = getattr(config.counterfactual, 'allow_relaxation', True)
        relaxation_factor = getattr(config.counterfactual, 'relaxation_factor', 2.0)
        
        counterfactual = cf_model.generate_counterfactual(
            ORIGINAL_SAMPLE, 
            TARGET_CLASS, 
            config.counterfactual.population_size,
            config.counterfactual.max_generations,
            mutation_rate=config.counterfactual.mutation_rate,
            allow_relaxation=allow_relaxation,
            relaxation_factor=relaxation_factor
        )
        
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
        # Note: We return serializable data, but also keep a reference to constraints and model
        # for validate_constraints to work
        evolution_history = getattr(cf_model, 'evolution_history', [])
        best_fitness_list = getattr(cf_model, 'best_fitness_list', [])
        average_fitness_list = getattr(cf_model, 'average_fitness_list', [])
        
        return {
            'replication_num': replication_num,
            'counterfactual': counterfactual,
            'all_counterfactuals': [counterfactual],  # DPG returns single CF
            'evolution_history': evolution_history,
            'best_fitness_list': best_fitness_list,
            'average_fitness_list': average_fitness_list,
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
        print(f"WARNING: DPG Replication {replication_num} failed: {exc}")
        traceback.print_exc()
        return None


def _run_single_replication_dice(args):
    """Helper function to run a single DiCE replication in parallel.
    
    Args:
        args: Tuple containing (replication_num, ORIGINAL_SAMPLE, TARGET_CLASS, 
              FEATURES_NAMES, dict_non_actionable, config_dict, model, constraints,
              train_df, continuous_features, categorical_features)
    
    Returns:
        Dict with replication results or None if failed
    """
    if not DICE_AVAILABLE:
        print("ERROR: dice-ml not available. Install with: pip install dice-ml")
        return None
    
    (
        replication_num,
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
            print(f"WARNING: DiCE replication {replication_num}: No actionable features!")
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
        
        # Build feature_weights from config if specified
        feature_weights = None
        if hasattr(config.counterfactual, 'feature_weights') and config.counterfactual.feature_weights:
            feature_weights_config = config.counterfactual.feature_weights
            if hasattr(feature_weights_config, '_config'):
                feature_weights = feature_weights_config._config
            elif hasattr(feature_weights_config, 'to_dict'):
                feature_weights = feature_weights_config.to_dict()
            else:
                feature_weights = dict(feature_weights_config) if feature_weights_config else None
        
        # Create DiCE data interface
        # DiCE needs a DataFrame with the outcome column
        outcome_name = '_target_'
        train_df_with_target = train_df.copy()
        
        # Get continuous and categorical feature names
        continuous_feature_names = [FEATURES_NAMES[i] for i in continuous_features] if continuous_features else []
        categorical_feature_names = [FEATURES_NAMES[i] for i in categorical_features] if categorical_features else []
        
        # Create DiCE Data object
        d = dice_ml.Data(
            dataframe=train_df_with_target,
            continuous_features=continuous_feature_names,
            outcome_name=outcome_name
        )
        
        # Create DiCE Model object
        m = dice_ml.Model(model=model, backend='sklearn')
        
        # Get DiCE parameters from config
        total_CFs = getattr(config.counterfactual, 'total_CFs', 4)
        proximity_weight = getattr(config.counterfactual, 'proximity_weight', 0.5)
        diversity_weight = getattr(config.counterfactual, 'diversity_weight', 1.0)
        generation_method = getattr(config.counterfactual, 'generation_method', 'genetic')
        
        # Create DiCE explainer with specified method
        exp = dice_ml.Dice(d, m, method=generation_method)
        
        # Prepare query instance as DataFrame
        query_df = pd.DataFrame([ORIGINAL_SAMPLE])
        
        # Generate counterfactuals
        dice_exp = exp.generate_counterfactuals(
            query_df,
            total_CFs=total_CFs,
            desired_class=int(TARGET_CLASS),
            features_to_vary=features_to_vary,
            permitted_range=permitted_range if permitted_range else None,
            proximity_weight=proximity_weight,
            diversity_weight=diversity_weight,
        )
        
        # Extract results
        if dice_exp.cf_examples_list and len(dice_exp.cf_examples_list) > 0:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or len(cf_df) == 0:
                print(f"WARNING: DiCE replication {replication_num}: No counterfactuals generated")
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
            
            # Select the "best" counterfactual (closest to original that reaches target)
            # Use L2 distance to pick the best one
            best_cf = None
            best_distance = float('inf')
            
            for cf in all_counterfactuals:
                # Verify it predicts the target class
                cf_pred = model.predict(pd.DataFrame([cf]))[0]
                if cf_pred == TARGET_CLASS:
                    distance = sum((ORIGINAL_SAMPLE[f] - cf[f])**2 for f in FEATURES_NAMES)**0.5
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = cf
            
            # If no CF reaches target, take the first one anyway
            if best_cf is None:
                best_cf = all_counterfactuals[0]
            
            return {
                'replication_num': replication_num,
                'counterfactual': best_cf,
                'all_counterfactuals': all_counterfactuals,
                'evolution_history': [best_cf],  # DiCE doesn't have evolution history
                'best_fitness_list': [],  # DiCE doesn't track fitness
                'average_fitness_list': [],
                'dict_non_actionable': dict_non_actionable,
                'method': 'dice',
                # Store DiCE-specific params
                'constraints': constraints,
                'feature_names': FEATURES_NAMES,
                'total_CFs': total_CFs,
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
            print(f"WARNING: DiCE replication {replication_num}: No counterfactuals in result")
            return None
        
    except Exception as exc:
        print(f"WARNING: DiCE Replication {replication_num} failed: {exc}")
        traceback.print_exc()
        return None


def _run_single_replication(args):
    """Dispatcher for running a single replication - routes to DPG or DiCE.
    
    Args:
        args: Tuple containing replication parameters including method config
    
    Returns:
        Dict with replication results or None if failed
    """
    # Extract config to determine method
    config_dict = args[5]  # config_dict is at index 5
    config = DictConfig(config_dict)
    method = getattr(config.counterfactual, 'method', 'dpg').lower()
    
    if method == 'dice':
        return _run_single_replication_dice(args)
    else:
        return _run_single_replication_dpg(args)
