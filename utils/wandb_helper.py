"""Weights & Biases integration utilities for CounterFactualDPG experiments.

This module provides helpers for initializing and configuring WandB runs,
including metric definitions for proper visualization and tracking.
"""

from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from utils.config_manager import DictConfig


def init_wandb(config: DictConfig, resume_id: Optional[str] = None, offline: bool = False):
    """Initialize Weights & Biases run.
    
    Args:
        config: Experiment configuration
        resume_id: Optional WandB run ID to resume from
        offline: Whether to run in offline mode (no syncing)
        
    Returns:
        WandB run object or None if WandB not available
    """
    if not WANDB_AVAILABLE:
        print("WARNING: WandB not available, skipping initialization")
        return None
    
    mode = "offline" if offline else "online"
    
    # Allow optional entity (organization/team) to be specified in config
    entity = getattr(config.experiment, 'entity', None)
    
    if resume_id:
        run = wandb.init(
            entity=entity,
            project=config.experiment.project,
            id=resume_id,
            resume="must",
            mode=mode
        )
    else:
        run = wandb.init(
            entity=entity,
            project=config.experiment.project,
            name=config.experiment.name,
            config=config.to_dict(),
            tags=getattr(config.experiment, 'tags', None),
            notes=getattr(config.experiment, 'notes', None),
            mode=mode
        )

    # Use WandB's built-in git integration; manual collection removed

    return run


def configure_wandb_metrics():
    """Configure WandB metric definitions for improved visualization.
    
    Sets up step relationships for different metric categories:
    - Fitness metrics use generation as the step
    - Replication and combination metrics use default step
    """
    if not WANDB_AVAILABLE or wandb is None:
        return
    
    # Fitness metrics use generation as the step (for proper x-axis alignment)
    wandb.define_metric("generation")
    wandb.define_metric("fitness/*", step_metric="generation")
    
    # Replication and combination metrics use default step (sequential logging)
    wandb.define_metric("replication/*")
    wandb.define_metric("metrics/per_counterfactual/*")  # Per-counterfactual metrics (individual CF quality)
    wandb.define_metric("metrics/combination/*")  # Combination-level aggregate metrics
