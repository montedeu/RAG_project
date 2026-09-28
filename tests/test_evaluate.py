"""
Test scenarioes:
- ALl metrics:
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
"""

import math

import pytest

from evaluate import ndcg_at_k, recall_at_k, reciprocal_rank

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
    ],
    ids=["first-found", "k-cutoff", "second-found"],
)
def test_reciprocal_rank_returns(ranked, relevant, k, expected):
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
