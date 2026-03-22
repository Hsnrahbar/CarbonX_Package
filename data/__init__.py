"""
Top-level CarbonX package exports.
"""

from .core import GasReactor, MappingWrapper

# Backward-compatible aliases used by your tests/manual
MovingSectionalModel = GasReactor
Mapping_Wrapper = MappingWrapper

__all__ = [
    "GasReactor",
    "MappingWrapper",
    "MovingSectionalModel",
    "Mapping_Wrapper",
]