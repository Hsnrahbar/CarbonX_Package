"""
Core modules for CarbonX solver
Contains the main reactor models and wrapper classes
"""

# Import core classes
try:
    from .carbonx_wrapper import GasReactor
except ImportError as e:
    print(f"Warning: Could not import GasReactor: {e}")
    GasReactor = None

try:
    from .mapping_wrapper import MappingWrapper
except ImportError as e:
    print(f"Warning: Could not import MappingWrapper: {e}")
    MappingWrapper = None

try:
    from . import cythonization_module_mean
    from . import cythonization_module_fmr
    from . import cythonization_module_cr
    from . import cythonization_module_fuchs
except ImportError as e:
    print(f"Warning: Could not import helper core modules: {e}")

# Define public API
__all__ = [
    'GasReactor',
    'MappingWrapper',
    'cythonization_module_mean',
    'cythonization_module_cr',
    'cythonization_module_fmr',
    'cythonization_module_fuchs',
    
]


