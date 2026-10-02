from collections.abc import Sequence


def fuse_results_rrf(
    ranked_lists: Sequence[Sequence[str]],
    rrf_constant: int = 60,
) -> list[str]:
    """Fuse ranked document lists using Reciprocal Rank Fusion."""
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")

    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        seen_documents: set[str] = set()
        for rank, document in enumerate(ranked_list, start=1):
            if document in seen_documents:
                continue

            seen_documents.add(document)
            scores[document] = scores.get(document, 0.0) + 1 / (rrf_constant + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)
