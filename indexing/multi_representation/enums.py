"""
Enumerations for multi-representation indexing.
"""

from enum import Enum


class RepresentationType(Enum):
    """Types of document representations"""
    FULL_TEXT = "full_text"           # Complete document
    SUMMARY = "summary"               # Document summary
    TITLE_ABSTRACT = "title_abstract" # Title and abstract only
    QUESTIONS = "questions"           # Questions the document answers
    HYPOTHETICAL_ANSWERS = "hypothetical_answers"  # Hypothetical answers (HyDE-style)
    KEY_PHRASES = "key_phrases"       # Important phrases/keywords
    ENTITIES = "entities"             # Named entities
    STRUCTURED = "structured"         # Structured data/views


class FusionStrategy(Enum):
    """Strategies for fusing results from multiple representations"""
    WEIGHTED_SUM = "weighted_sum"         # Weighted sum of scores
    MAX_SCORE = "max_score"               # Take maximum score per document
    RANK_FUSION = "rank_fusion"           # Fusion based on rankings (RRF)
    DISTRIBUTION_BASED = "distribution_based"  # Score distribution-based fusion
    LEARNING_BASED = "learning_based"     # Learned fusion (requires training)