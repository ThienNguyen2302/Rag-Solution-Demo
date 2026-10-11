"""
Multi-representation indexing package for RAG systems.
"""

from .enums import RepresentationType, FusionStrategy
from .datastructures import RepresentationConfig, MultiRepresentationResult
from .multi_representation_index import MultiRepresentationIndex
from .helper_functions import create_default_multi_representation_configs

__all__ = [
    'RepresentationType',
    'FusionStrategy',
    'RepresentationConfig',
    'MultiRepresentationResult',
    'MultiRepresentationIndex',
    'create_default_multi_representation_configs'
]