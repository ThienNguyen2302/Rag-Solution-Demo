"""
Data structures for multi-representation indexing.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class RepresentationConfig:
    """Configuration for a specific representation"""
    name: str                           # Unique identifier for this representation
    type: 'RepresentationType'          # Type of representation
    index_type: str                     # Index type to use (inverted, vector, hybrid)
    chunking_strategy: Optional[str] = None  # Chunking strategy for this representation
    weight: float = 1.0                 # Weight in fusion (default 1.0)
    enabled: bool = True                # Whether this representation is active
    transform_function: Optional[Callable[[str], str]] = None  # Function to transform content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional configuration


@dataclass
class MultiRepresentationResult:
    """Result from multi-representation search"""
    document_id: str
    score: float
    representation_scores: Dict[str, float]  # Scores from each representation
    representations_used: List[str]          # Which representations contributed
    metadata: Dict[str, Any] = field(default_factory=dict)