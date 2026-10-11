"""
Helper functions for multi-representation indexing.
"""

from typing import List, Dict, Any
from .enums import RepresentationType
from .datastructures import RepresentationConfig
import re


def create_default_multi_representation_configs() -> List[RepresentationConfig]:
    """Create a standard set of representation configurations."""
    return [
        RepresentationConfig(
            name="full_text_inverted",
            type=RepresentationType.FULL_TEXT,
            index_type="inverted",
            weight=1.0,
            metadata={"description": "Full text with keyword indexing"},
        ),
        RepresentationConfig(
            name="full_text_vector",
            type=RepresentationType.FULL_TEXT,
            index_type="vector",
            weight=1.0,
            metadata={"description": "Full text with semantic indexing"},
        ),
        RepresentationConfig(
            name="title_vector",
            type=RepresentationType.TITLE_ABSTRACT,
            index_type="vector",
            weight=1.2,
            metadata={"description": "Title-focused semantic search"},
            transform_function=lambda text: _extract_title(text),
        ),
        RepresentationConfig(
            name="summary_hybrid",
            type=RepresentationType.SUMMARY,
            index_type="hybrid",
            weight=1.1,
            metadata={"description": "Summary with hybrid indexing"},
            transform_function=lambda text: _extract_summary(text),
        ),
        RepresentationConfig(
            name="questions_vector",
            type=RepresentationType.QUESTIONS,
            index_type="vector",
            weight=1.0,
            metadata={"description": "Questions the document answers"},
            transform_function=lambda text: _generate_questions(text),
        ),
    ]


def _extract_title(text: str) -> str:
    """Extract title from text (simplified)."""
    lines = text.strip().split("\n")
    if lines:
        for line in lines[:3]:
            stripped = line.strip()
            if len(stripped) < 100 and len(stripped) > 0:
                return stripped
    return lines[0] if lines else text[:100]


def _extract_summary(text: str) -> str:
    """Extract summary from text (simplified - first few sentences)."""
    sentences = re.split(r"[.!?]+", text)
    summary_sentences = [sentence.strip() for sentence in sentences if sentence.strip()][:3]
    return ". ".join(summary_sentences) + ("." if summary_sentences else "")


def _generate_questions(text: str) -> str:
    """Generate questions that the text answers (simplified)."""
    questions: List[str] = []

    if " is " in text.lower() or " are " in text.lower():
        lines = text.split(".")
        for line in lines:
            lower_line = line.lower()
            if " is " in lower_line or " are " in lower_line:
                if " is " in lower_line:
                    subject = line.split(" is ")[0].strip()
                    questions.append(f"What is {subject}?")
                if " are " in lower_line:
                    subject = line.split(" are ")[0].strip()
                    questions.append(f"What are {subject}?")

    if " how to " in text.lower():
        lines = text.split(".")
        for line in lines:
            lower_line = line.lower()
            if " how to " in lower_line:
                action = line.split(" how to ", 1)[1].strip()
                questions.append(f"How to {action}?")

    if not questions:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if words:
            key_words = [word for word in words if len(word) > 4][:3]
            if key_words:
                questions.append(f"What is {key_words[0]}?")
                questions.append(f"How does {key_words[0]} work?")
                if len(key_words) > 1:
                    questions.append(
                        f"What is the relationship between {key_words[0]} and {key_words[1]}?"
                    )

    return " ".join(questions)