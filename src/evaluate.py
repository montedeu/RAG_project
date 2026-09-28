"""Retrieval metrics. Each function scores one query and assumes binary relevance."""

import math


def _check(relevant: set[str], k: int) -> None:
    """Raise ValueError if the query has no judgements or k is not a valid cutoff."""
    if not relevant:
        raise ValueError("Query has no relevant documents, so it cannot be scored.")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k.

    Args:
        ranked_doc_ids: retrieved doc ids, best first, already deduped.
        relevant: doc ids judged relevant for this query.
        k: cutoff; only the first k ranked ids are scored.

    Returns:
        Recall in [0, 1].

    Raises:
        ValueError: if `relevant` is empty or k < 1.
    """
    _check(relevant, k)
    return len(relevant & set(ranked_doc_ids[:k])) / len(relevant)


def reciprocal_rank(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant document, or 0.0 if none is in the top k.

    Average over queries to get MRR.

    Args:
        ranked_doc_ids: retrieved doc ids, best first, already deduped.
        relevant: doc ids judged relevant for this query.
        k: cutoff; only the first k ranked ids are scored.

    Returns:
        Reciprocal rank in [0, 1].

    Raises:
        ValueError: if `relevant` is empty or k < 1.
    """
    _check(relevant, k)
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Rank-discounted gain in [0, 1], normalised by the best ranking possible.

    The ideal ranking is `min(k, len(relevant))` relevant documents in the top
    slots, so IDCG depends on the judgements alone, never on what was retrieved.

    Args:
        ranked_doc_ids: retrieved doc ids, best first, already deduped.
        relevant: doc ids judged relevant for this query.
        k: cutoff; only the first k ranked ids are scored.

    Returns:
        nDCG in [0, 1].

    Raises:
        ValueError: if `relevant` is empty or k < 1.
    """
    _check(relevant, k)
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1)
        if doc_id in relevant
    )
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg
