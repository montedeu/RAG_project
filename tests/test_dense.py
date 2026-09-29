"""
Scenarios:
format_passage:
- title without end punctuation is joined with ". "
- title ending in ".", "!" or "?" is joined with " " (no "..")
- title with trailing whitespace is stripped before joining
- empty or whitespace-only title behaves like with_title=False
- with_title=False leaves the title out
- body is passed through unchanged (no period added to a mid-word ending)
format_query:
- only the prefix is added; the text is unchanged
cache_path:
- changing any one argument changes the path
- "/" in the model id doesn't create a subdirectory (parent is ARTIFACTS_DIR)
- path ends in .npz, matching the file np.savez writes
"""

import pytest

from config import ARTIFACTS_DIR
from dense import cache_path, format_passage, format_query
from schema import Chunk


@pytest.mark.parametrize(
    "title, text, with_title, expected",
    [
        (
            "Aspirin and stroke",
            "We studied 40 patients.",
            True,
            "passage: Aspirin and stroke. We studied 40 patients.",
        ),
        (
            "Induction of X by Y.",
            "We studied 40 pati",
            True,
            "passage: Induction of X by Y. We studied 40 pati",
        ),
        ("Is X safe? ", "We...", True, "passage: Is X safe? We..."),
        (
            "Induction of X by Y.",
            "We studied 40 pati",
            False,
            "passage: We studied 40 pati",
        ),
        (" ", "We...", True, "passage: We..."),
        ("", "You...", True, "passage: You..."),
    ],
    ids=[
        "without-punctuation",
        "with-punctuation",
        "trailing-whitespace",
        "with-title-false",
        "only-whitespace",
        "empty-title",
    ],
)
def test_format_passage(title, text, with_title, expected):
    assert format_passage(Chunk("1#0", "1", 0, title, text), with_title) == expected


def test_format_query():
    assert format_query("Aspirin prevents stroke") == "query: Aspirin prevents stroke"


BASE = {
    "model_name": "intfloat/multilingual-e5-small",
    "chunker_name": "chunk_by_sentence",
    "size": 400,
    "overlap": 50,
    "with_title": True,
}


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_name", "BAAI/bge-m3"),
        ("chunker_name", "chunk"),
        ("size", 300),
        ("overlap", 40),
        ("with_title", False),
    ],
    ids=["model", "chunker", "size", "overlap", "title"],
)
def test_cache_path_changes_with_each_argument(field, value):
    assert cache_path(**{**BASE, field: value}) != cache_path(**BASE)


def test_cache_path_parent_directory():
    assert cache_path(**BASE).parent == ARTIFACTS_DIR


def test_cache_path_ends_in_npz():
    assert cache_path(**BASE).suffix == ".npz"
