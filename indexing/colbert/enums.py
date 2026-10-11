"""
Enumerations for ColBERT technique.
"""

from enum import Enum


class MatchScoreType(Enum):
    """Types of matching scores"""
    MAXSIM = "maxsim"          # MaxSim operation (ColBERT-style)
    DOT_PRODUCT = "dot_product" # Simple dot product
    COSINE = "cosine"          # Cosine similarity