"""
Test scenarios:
- All metrics:
1. Raises an error if relevant is empty
2. Raises an error if k is less than 1
3. Empty ranked_doc_ids in all functions returns 0
- Recall:
1. Correctly returning the values that are present both in ranked_doc_ids and relevant
2. If there is no intersections returns 0
3. If all relevant docs found returns 1
4. If 1 of 2 relevant docs found returns 0.5
5. If len(relevant) > k, returns less than 1
- RR:
1. Correctly returns 1 / rank for the first relevant doc
2. The first relevant doc is at rank 1 returns 1
3. If the relevant doc is past the cutoff returns 0
4. If returns relevant doc at rank 2 returns 0.5
- nDCG
1. Correctly returns ndcg
2. If there are no intersections in ranked_doc_ids and relevant returns 0
3. A perfect ranking returns 1
4. More relevant docs than k, all top k relevant, returns 1
5. Ranking relevant docs higher scores higher
- map_chunks_to_docs:
1. Strips the position and dedupes, keeping each doc's best rank
2. Doc ids containing "#" are kept whole
- evaluate:
1. Averages every metric at every k over queries (hand-computed example)
2. A perfect run scores 1 everywhere when k >= number of relevant docs
3. Queries missing from the run score 0; queries missing from qrels are ignored
4. Judgements with score 0 are not relevant
5. Empty qrels raises
"""

import math

import pytest

from evaluate import (
    evaluate,
    map_chunks_to_docs,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# All metrics


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank, ndcg_at_k])
@pytest.mark.parametrize(
    "relevant, k, message",
    [
        (set(), 5, "no relevant documents"),
        ({"d1"}, 0, "k must be at least 1"),
        ({"d1"}, -1, "k must be at least 1"),
    ],
    ids=["empty-relevant", "k-zero", "k-negative"],
)
def test_invalid_input_raises(metric, relevant, k, message):
    with pytest.raises(ValueError, match=message):
        metric(["d1", "d2"], relevant, k)


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank, ndcg_at_k])
def test_empty_ranked_doc_ids(metric):
    assert metric([], {"d1"}, 5) == 0


# Recall_at_k


@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["d1"], {"d2"}, 1, 0),
        (["d1", "d2"], {"d1", "d2"}, 2, 1),
        (["d1", "d3"], {"d1", "d2"}, 2, 0.5),
        (["d1", "d2"], {"d1", "d2"}, 1, 0.5),
    ],
    ids=["no-intersection", "all-found", "half-found", "k-less-than-relevant"],
)
def test_recall(ranked, relevant, k, expected):
    assert recall_at_k(ranked, relevant, k) == pytest.approx(expected)


# Reciprocal rank


@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["d1", "d2"], {"d1"}, 2, 1),
        (["d1", "d2"], {"d2"}, 1, 0),
        (["d1", "d2"], {"d2"}, 2, 0.5),
        (["d1", "d2", "d3"], {"d2", "d3"}, 3, 0.5),
    ],
    ids=["first-found", "k-cutoff", "second-found", "several-relevant"],
)
def test_reciprocal_rank(ranked, relevant, k, expected):
    assert reciprocal_rank(ranked, relevant, k) == pytest.approx(expected)


# NDCG


@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["d1"], {"d2"}, 1, 0),
        (["d1", "d2", "d3"], {"d1", "d2"}, 3, 1),
        (
            ["d1", "d2", "d3"],
            {"d1", "d3"},
            3,
            (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
        ),
        (["d1", "d2"], {"d1", "d2", "d3"}, 2, 1),
        (["d2", "d1"], {"d1"}, 1, 0),
    ],
    ids=[
        "no-intersection",
        "perfect",
        "hand-computed",
        "more-relevant-than-k",
        "past-cutoff",
    ],
)
def test_ndcg(ranked, relevant, k, expected):
    assert ndcg_at_k(ranked, relevant, k) == pytest.approx(expected)


def test_ndcg_rewards_higher_ranks():
    relevant = {"d1"}
    assert ndcg_at_k(["d1", "d2"], relevant, 2) > ndcg_at_k(["d2", "d1"], relevant, 2)


# map_chunks_to_docs


@pytest.mark.parametrize(
    "chunk_ids, expected",
    [
        (["123#0", "456#2", "123#1"], ["123", "456"]),
        (["456#1", "123#0", "456#0"], ["456", "123"]),
        (["a#b#0", "a#b#1", "c#0"], ["a#b", "c"]),
        ([], []),
    ],
    ids=["dedupe", "best-rank-kept", "hash-in-doc-id", "empty"],
)
def test_map_chunks_to_docs(chunk_ids, expected):
    assert map_chunks_to_docs(chunk_ids) == expected


# evaluate


def test_evaluate_averages_over_queries():
    run = {"q1": ["10#0", "20#1", "10#1", "30#0"], "q2": ["40#0", "50#0"]}
    qrels = {"q1": {"10": 1}, "q2": {"50": 1, "60": 1}}
    q2_ndcg_at_3 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert evaluate(run, qrels, (1, 3)) == pytest.approx(
        {
            "recall@1": 0.5,
            "recall@3": 0.75,
            "mrr@1": 0.5,
            "mrr@3": 0.75,
            "ndcg@1": 0.5,
            "ndcg@3": (1 + q2_ndcg_at_3) / 2,
        }
    )


def test_evaluate_perfect_run():
    qrels = {"q1": {"10": 1, "20": 1}, "q2": {"30": 1}}
    run = {qid: [f"{doc_id}#0" for doc_id in rel] for qid, rel in qrels.items()}
    # k >= 2 so both of q1's relevant docs fit; at k=1 recall can't reach 1.
    assert all(
        value == pytest.approx(1) for value in evaluate(run, qrels, (2, 5)).values()
    )


def test_evaluate_missing_and_extra_queries():
    qrels = {"q1": {"10": 1}, "q2": {"20": 1}}
    run = {"q1": ["10#0"], "unjudged": ["20#0"]}
    assert evaluate(run, qrels, (1,))["recall@1"] == pytest.approx(0.5)


def test_evaluate_ignores_zero_score_judgements():
    qrels = {"q1": {"10": 1, "20": 0}}
    assert evaluate({"q1": ["20#0", "10#0"]}, qrels, (1,))["recall@1"] == 0


def test_evaluate_empty_qrels_raises():
    with pytest.raises(ValueError, match="qrels is empty"):
        evaluate({"q1": ["10#0"]}, {}, (1,))
