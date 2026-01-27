"""
Constants: Centralized constants for counterfactual generation.

This module contains all magic numbers and hardcoded values used throughout
the counterfactual generation system, making them easier to tune and maintain.
"""

# ============================================================================
# Fitness Calculation Constants
# ============================================================================

# Invalid fitness penalty for samples that violate hard constraints
INVALID_FITNESS = 1e6

# Class prediction penalty coefficients
CLASS_PENALTY_SOFT_BASE = 100.0  # Base penalty for low target class probability
CLASS_PENALTY_HARD_BOOST = 50.0  # Additional penalty when not predicting target
BOUNDARY_PENALTY_THRESHOLD = 0.1  # Distance threshold for boundary penalty
BOUNDARY_PENALTY_VALUE = 30.0  # Penalty for being too far from boundary

# Constraint violation multiplier
CONSTRAINT_VIOLATION_MULTIPLIER = 2.0

# ============================================================================
# Mutation Constants
# ============================================================================

# Small step for boundary crossing in dual-boundary mutation
MUTATION_EPSILON = 0.01

# Mutation range scaling factors
MUTATION_RANGE_SCALE_ACTIONABLE = 0.1  # Scale for actionable constrained features
MUTATION_RANGE_SCALE_GENERAL = 0.15  # Scale for general dual-boundary mutation
MUTATION_RANGE_SCALE_BOUNDARY_PUSH = 0.2  # Scale when pushing toward boundaries
MUTATION_RANGE_SCALE_OVERLAP = 0.1  # Scale for overlapping bounds mutation

# Default mutation ranges
MUTATION_RANGE_DEFAULT = 0.5
MUTATION_RANGE_MIN = 0.1  # Minimum mutation range

# Non-overlapping feature mutation rate boost
MUTATION_RATE_BOOST_NON_OVERLAPPING = 1.5

# ============================================================================
# Sample Generation Constants
# ============================================================================

# Range adjustment for unbounded constraints
SAMPLE_GEN_RANGE_SCALE = 0.5  # Scale for range around original value

# Escape pressure influence on sample generation
SAMPLE_GEN_ESCAPE_BIAS = 0.3  # Bias factor for escape-aware generation

# Actionability constraint adjustments
ACTIONABILITY_RANGE_ADJUST = 0.1  # Adjustment factor for ensuring valid ranges

# ============================================================================
# Probability and Validation Constants
# ============================================================================

# Default fallback for distance to boundary when predict_proba unavailable
DEFAULT_BOUNDARY_DISTANCE = 0.05

# ============================================================================
# Fitness Sharing Constants
# ============================================================================

# Base sigma_share for fitness sharing (scaled with dimensionality)
FITNESS_SHARING_BASE_SIGMA = 3.0


# ============================================================================
# Rounding and Precision
# ============================================================================

# Decimal places for rounding feature values after mutation
FEATURE_VALUE_PRECISION = 2

# ============================================================================
# Unconstrained Feature Penalty Constants
# ============================================================================

# Penalty multiplier for changing features without target class constraints
# Higher values make unconstrained feature changes more costly (changed as last resort)
UNCONSTRAINED_CHANGE_PENALTY_FACTOR = 10.0

# Mutation rate reduction for unconstrained features
# Lower values reduce probability of mutating unconstrained features
UNCONSTRAINED_MUTATION_RATE_FACTOR = 0.3
