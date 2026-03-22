"""
Machine Learning modules for CarbonX solver
Contains ML classifiers and optimization routines for parametric studies
"""

# Import ML modules
try:
    from . import ml_classifier_2D
    from . import gradient_descent
    from . import compute_cost
    from . import compute_cost_reg
    from . import map_feature1
    from . import compute_gradient
    from . import normalize_features
    from . import sigmoid1
    from . import utils
except ImportError as e:
    print(f"Warning: Some ML modules could not be imported: {e}")

# Import specific functions for convenience
try:
    from .gradient_descent import gradient_descent
    from .normalize_features import normalize_features, denormalize_features
    from .utils import *  # If utils has helper functions
except ImportError:
    pass

# Define public API
__all__ = [
    'ml_classifier_2D',
    'gradient_descent',
    'compute_cost',
    'compute_cost_reg',
    'map_feature1',
    'compute_gradient',
    'normalize_features',
    'denormalize_features',
    'sigmoid1',
    'utils'
]