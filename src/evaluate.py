"""Retrieval metrics. Each metric scores one query and assumes binary relevance;
`evaluate` averages them over a whole run."""

import math
from collections import defaultdict


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


METRICS = {"recall": recall_at_k, "mrr": reciprocal_rank, "ndcg": ndcg_at_k}


def map_chunks_to_docs(ranked_chunk_ids: list[str]) -> list[str]:
    """Collapse ranked chunk ids to ranked doc ids; each doc keeps its best chunk's rank.

    Args:
        ranked_chunk_ids: one query's retrieved chunk ids (`'{doc_id}#{position}'`), best first.

    Returns:
        Doc ids, best first, each appearing once. Empty input gives an empty list.
    """
    doc_ids = (chunk_id.rsplit("#", 1)[0] for chunk_id in ranked_chunk_ids)
    # Ordered dedupe: dicts keep insertion order, so the first occurrence wins.
    return list(dict.fromkeys(doc_ids))


def evaluate(
    run: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks: tuple[int, ...]
) -> dict[str, float]:
    """Mean of every metric at every k over the queries in qrels.

    Queries missing from `run` score 0; queries missing from `qrels` are ignored.

    Args:
        run: `{query_id: [chunk_id, ...]}`, each list ranked best first.
        qrels: `{query_id: {doc_id: score}}`; docs with score > 0 count as relevant.
        ks: cutoffs to report, e.g. `(1, 5, 10, 20)`.

    Returns:
        `{"recall@k" | "mrr@k" | "ndcg@k": mean over queries}` for every k, grouped by
        metric, e.g. `{"recall@1": 0.5, "recall@5": 0.8, ..., "ndcg@5": 0.7}`.

    Raises:
        ValueError: if qrels is empty or a query has no doc with score > 0.
    """
    if not qrels:
        raise ValueError("qrels is empty, so there is nothing to evaluate.")
    scores = defaultdict(list)
    for query_id, judgements in qrels.items():
        ranked_doc_ids = map_chunks_to_docs(run.get(query_id, []))
        relevant = {doc_id for doc_id, score in judgements.items() if score > 0}
        for name, metric in METRICS.items():
            for k in ks:
                scores[f"{name}@{k}"].append(metric(ranked_doc_ids, relevant, k))
    return {name: sum(values) / len(values) for name, values in scores.items()}


if __name__ == "__main__":
    import random

    from config import EVAL_KS, HANDWRITTEN_DIR, SCIFACT_DIR
    from ingest import load_corpus, load_query_relations

    # Sanity check before any retriever exists: random runs must score ~0 and perfect runs 1,
    # except recall@k for queries with more than k relevant docs.
    doc_ids = list(load_corpus(SCIFACT_DIR / "corpus.jsonl"))
    rng = random.Random(0)
    for dataset, path in (("scifact", SCIFACT_DIR), ("handwritten", HANDWRITTEN_DIR)):
        qrels = load_query_relations(path / "qrels" / "test.tsv")
        runs = {
            "perfect": {qid: [f"{d}#0" for d in rel] for qid, rel in qrels.items()},
            "random": {
                qid: [f"{d}#0" for d in rng.sample(doc_ids, max(EVAL_KS))]
                for qid in qrels
            },
        }
        for run_name, run in runs.items():
            results = evaluate(run, qrels, EVAL_KS)
            print(f"{dataset} ({len(qrels)} queries), {run_name}:")
            print(
                "  "
                + "  ".join(f"{name} {value:.3f}" for name, value in results.items())
            )
