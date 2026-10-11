"""
Simple ColBERT-style embedder for demonstration (replace with real BERT model).
"""

import re
import math
from typing import List, Dict, Any
from .base_embedder import BaseColBERTEmbedder


class SimpleColBERTEmbedder(BaseColBERTEmbedder):
    """Simple ColBERT-style embedder for demonstration (replace with real BERT model)"""

    def __init__(self, embedding_dimension: int = 128):
        self._embedding_dimension = embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def _simple_word_embedding(self, word: str, position: int, context_words: List[str]) -> List[float]:
        """Generate a simple contextualized embedding for a word"""
        # This is a simplified version - real ColBERT uses BERT
        embedding = [0.0] * self._embedding_dimension

        if not word:
            return embedding

        # Word features
        word_hash = hash(word) % 1000
        for i in range(min(20, self._embedding_dimension)):
            embedding[i] = ((word_hash + i * 17) % 100) / 100.0

        # Position features (normalized)
        if self._embedding_dimension > 20:
            pos_norm = position / max(len(context_words), 1)
            embedding[20] = pos_norm

        # Context features (simplified)
        context_hash = sum(hash(w) for w in context_words) % 1000
        for i in range(10):
            idx = 21 + i
            if idx < self._embedding_dimension:
                embedding[idx] = ((context_hash + i * 13) % 100) / 100.0

        # Length feature
        if self._embedding_dimension > 31:
            word_len_norm = min(len(word) / 20.0, 1.0)
            embedding[31] = word_len_norm

        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_tokens(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Embed texts into token embeddings (simplified version)"""
        all_token_embeddings = []

        for text in texts:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', text)
            if not tokens:
                all_token_embeddings.append([])
                continue

            # Find token positions in original text
            token_positions = []
            search_pos = 0
            for token in tokens:
                pos = text.find(token, search_pos)
                if pos != -1:
                    token_positions.append((pos, pos + len(token)))
                    search_pos = pos + len(token)
                else:
                    # Fallback
                    token_positions.append((search_pos, search_pos + len(token)))
                    search_pos += len(token)

            # Create token embeddings
            token_embeddings = []
            for i, (token, (start_pos, end_pos)) in enumerate(zip(tokens, token_positions)):
                # Simple contextualization: use surrounding words as context
                context_start = max(0, i - 2)
                context_end = min(len(tokens), i + 3)
                context_words = tokens[context_start:context_end]

                embedding = self._simple_word_embedding(token, i, context_words)
                token_embeddings.append({
                    'token_id': i,
                    'token_text': token,
                    'embedding': embedding,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })

            all_token_embeddings.append(token_embeddings)

        return all_token_embeddings