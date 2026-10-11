"""
Simple extractive summarizer for demonstration (replace with real model).
"""

from typing import List, Optional
import re
from .base_summarizer import BaseSummarizer


class SimpleSummarizer(BaseSummarizer):
    """Simple extractive summarizer for demonstration (replace with real model)"""

    def __init__(self, max_summary_length: int = 100):
        self.max_summary_length = max_summary_length

    def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract key sentences using simple scoring"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + ('.' if sentences else '')

        # Simple scoring: prefer sentences with key terms
        # In practice, would use TF-IDF, TextRank, or neural models
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Position bias: prefer beginning and end
            position_score = 1.0 - abs(0.5 - i / max(len(sentences) - 1, 1)) * 0.5
            # Length penalty: prefer medium length sentences
            length_score = min(len(sentence.split()) / 20.0, 1.0) if len(sentence.split()) > 0 else 0
            # Keyword bonus: sentences with numbers or proper nouns
            keyword_bonus = 0.2 if any(c.isdigit() for c in sentence) else 0.0
            keyword_bonus += 0.2 if any(c.isupper() and c.isalpha() for c in sentence) else 0.0

            total_score = position_score * 0.4 + length_score * 0.3 + keyword_bonus * 0.3
            scored_sentences.append((total_score, i, sentence))

        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        selected = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])  # Sort by original order
        result = '. '.join([s[2] for s in selected])
        return result + ('.' if result and not result.endswith('.') else '')

    def summarize(self, texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """Summarize texts using extractive approach"""
        max_len = max_length or self.max_summary_length
        summaries = []

        for text in texts:
            if len(text) <= max_len:
                summaries.append(text)
            else:
                # For longer texts, extract key sentences
                summary = self._extract_key_sentences(text, max_sentences=3)
                # Truncate if still too long
                if len(summary) > max_len:
                    summary = summary[:max_len].rsplit(' ', 1)[0] + '...'
                summaries.append(summary)

        return summaries

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Simple embedding function (placeholder)"""
        embeddings = []
        dim = 384  # Standard embedding dimension

        for text in texts:
            # Very simple embedding - in practice use SBERT, BERT, etc.
            embedding = [0.0] * dim
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

            if not words:
                embeddings.append(embedding)
                continue

            # Bag-of-words style embedding with positional encoding
            for i, word in enumerate(words[:50]):  # Limit to first 50 words
                word_hash = hash(word) % 1000
                idx = (i * 7 + word_hash) % dim
                embedding[idx] = min(embedding[idx] + 0.1, 1.0)

            # Add some basic text features
            if len(embedding) > 0:
                embedding[0] = min(len(text) / 1000.0, 1.0)  # Length feature
            if len(embedding) > 1:
                embedding[1] = min(len(words) / 100.0, 1.0)   # Word count feature

            # Normalize
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings