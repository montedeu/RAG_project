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
embed_chunks:
- ids come back in the input dict's order (fixture is deliberately unsorted)
- vectors are float32 with shape (n_chunks, dim)
- the model receives format_passage strings, with and without title
- row i of the vectors belongs to ids[i]
- normalize_embeddings=True is requested from the model
- the first call writes the cache file, creating missing parent directories
- a second identical call is served from the cache (no encode, same result)
- a stale cache (chunk added, removed, text changed, title toggled) is re-embedded
  and the new vectors are returned
- a stale cache is overwritten, so the next identical call is a cache hit
- empty chunks raise ValueError
build_index (real Milvus Lite in tmp_path):
- the collection holds one row per id
- each vector is stored under its own id
- missing parent directories of the .db path are created
- a rebuild drops the old collection (old ids are gone, count is the new one)
- more rows than one insert batch are all inserted (partial last batch included)
- mismatched ids/vectors lengths and empty input raise ValueError
"""

import dataclasses
import hashlib

import numpy as np
import pytest

from config import ARTIFACTS_DIR
from dense import build_index, cache_path, embed_chunks, format_passage, format_query
from schema import Chunk

DIM = 3


def fake_vector(text: str) -> np.ndarray:
    """Deterministic DIM-length vector derived from the text's hash."""
    digest = hashlib.sha256(text.encode()).digest()
    return np.frombuffer(digest[:DIM], dtype=np.uint8).astype(np.float32)


class FakeEncoder:
    """Stands in for SentenceTransformer: records each encode call, returns fake vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), "kwargs": kwargs})
        return np.stack([fake_vector(text) for text in texts])


def _chunk(doc_id: str, position: int, title: str, text: str) -> Chunk:
    """Build a Chunk with a consistent '{doc_id}#{position}' id."""
    return Chunk(f"{doc_id}#{position}", doc_id, position, title, text)


@pytest.fixture(scope="function")
def encoder():
    """A fresh FakeEncoder with no recorded calls."""
    return FakeEncoder()


@pytest.fixture(scope="function")
def chunks():
    """Three chunks, two from doc 1, deliberately not in sorted id order."""
    return {
        c.chunk_id: c
        for c in [
            _chunk("2", 0, "Is X safe?", "Short answer: yes."),
            _chunk("1", 0, "Aspirin and stroke", "We studied 40 patients."),
            _chunk("1", 1, "Aspirin and stroke", "Risk fell by half."),
        ]
    }


def _add_chunk(chunks):
    """Variant with one extra chunk from a new doc."""
    return {**chunks, "3#0": _chunk("3", 0, "New doc", "Extra text.")}


def _remove_chunk(chunks):
    """Variant without the last chunk."""
    return dict(list(chunks.items())[:-1])


def _change_text(chunks):
    """Variant with the same ids but the first chunk's text edited."""
    first = next(iter(chunks))
    return {**chunks, first: dataclasses.replace(chunks[first], text="Edited.")}


def _expected_vectors(chunks, with_title=True):
    """FakeEncoder's vectors for these chunks' passages, in dict order."""
    return np.stack(
        [fake_vector(format_passage(c, with_title)) for c in chunks.values()]
    )


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


def test_embed_chunks_returns_ids_in_input_order(encoder, tmp_path, chunks):
    ids, _ = embed_chunks(
        chunks,
        encoder,
        tmp_path / "emb.npz",
    )
    assert ids == list(chunks)


def test_embed_chunks_vector_shape_and_dtype(encoder, tmp_path, chunks):
    _, vectors = embed_chunks(
        chunks,
        encoder,
        tmp_path / "emb.npz",
    )
    assert vectors.shape == (len(chunks), DIM)
    assert vectors.dtype == np.float32


@pytest.mark.parametrize(
    "with_title",
    [
        pytest.param(True, id="with-title"),
        pytest.param(False, id="without-title"),
    ],
)
def test_embed_chunks_encodes_formatted_passages(encoder, tmp_path, chunks, with_title):
    embed_chunks(
        chunks,
        encoder,
        tmp_path / "emb.npz",
        with_title=with_title,
    )
    assert encoder.calls[0]["texts"] == [
        format_passage(chunk, with_title) for chunk in chunks.values()
    ]


def test_embed_chunks_rows_match_ids(encoder, tmp_path, chunks):
    ids, vectors = embed_chunks(
        chunks,
        encoder,
        tmp_path / "emb.npz",
    )
    np.testing.assert_array_equal(
        vectors, np.stack([fake_vector(format_passage(chunks[i])) for i in ids])
    )


def test_embed_chunks_requests_normalisation(encoder, tmp_path, chunks):
    embed_chunks(
        chunks,
        encoder,
        tmp_path / "emb.npz",
    )
    assert encoder.calls[0]["kwargs"]["normalize_embeddings"] is True


def test_embed_chunks_writes_cache(encoder, tmp_path, chunks):
    path = tmp_path / "emb.npz"
    embed_chunks(chunks, encoder, path)
    assert path.exists()


def test_embed_chunks_cache_hit(encoder, tmp_path, chunks):
    path = tmp_path / "emb.npz"
    first_ids, first_vectors = embed_chunks(chunks, encoder, path)
    ids, vectors = embed_chunks(chunks, encoder, path)
    assert len(encoder.calls) == 1
    assert ids == first_ids
    np.testing.assert_array_equal(vectors, first_vectors)


stale_variants = pytest.mark.parametrize(
    "make_variant",
    [_add_chunk, _remove_chunk, _change_text],
    ids=["add-chunk", "remove-chunk", "change-text"],
)


@stale_variants
def test_embed_chunks_stale_cache_reembeds(encoder, tmp_path, chunks, make_variant):
    path = tmp_path / "emb.npz"
    variant = make_variant(chunks)
    embed_chunks(chunks, encoder, path)
    ids, vectors = embed_chunks(variant, encoder, path)
    assert len(encoder.calls) == 2
    assert ids == list(variant)
    np.testing.assert_array_equal(vectors, _expected_vectors(variant))


@stale_variants
def test_embed_chunks_stale_cache_is_overwritten(
    encoder, tmp_path, chunks, make_variant
):
    path = tmp_path / "emb.npz"
    variant = make_variant(chunks)
    embed_chunks(chunks, encoder, path)
    embed_chunks(variant, encoder, path)
    ids, vectors = embed_chunks(variant, encoder, path)
    assert len(encoder.calls) == 2
    assert ids == list(variant)
    np.testing.assert_array_equal(vectors, _expected_vectors(variant))


def test_embed_chunks_title_toggle_reembeds(encoder, tmp_path, chunks):
    path = tmp_path / "emb.npz"
    embed_chunks(chunks, encoder, path, with_title=True)
    _, vectors = embed_chunks(chunks, encoder, path, with_title=False)
    assert len(encoder.calls) == 2
    np.testing.assert_array_equal(vectors, _expected_vectors(chunks, with_title=False))


def test_embed_chunks_creates_parent_dir(encoder, tmp_path, chunks):
    path = tmp_path / "subdir" / "emb.npz"
    embed_chunks(
        chunks,
        encoder,
        path,
    )
    assert path.exists()


def test_embed_chunks_empty_raises(encoder, tmp_path):
    with pytest.raises(ValueError, match="chunks is empty"):
        embed_chunks({}, encoder, tmp_path / "emb.npz")


COLLECTION = "test_collection"
INDEX_IDS = ["a#0", "b#0", "c#0"]
# Unit length, no ties: query [1, 0, 0] ranks a (1.0), b (0.8), c (0.0).
INDEX_VECTORS = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)


@pytest.fixture
def index(tmp_path):
    """A Milvus Lite index of INDEX_IDS / INDEX_VECTORS, closed after the test."""
    client = build_index(INDEX_IDS, INDEX_VECTORS, tmp_path / "index.db", COLLECTION)
    yield client
    client.close()


def test_build_index_row_count(index):
    assert index.get_collection_stats(COLLECTION)["row_count"] == len(INDEX_IDS)


def test_build_index_stores_each_vector_under_its_id(index):
    rows = index.get(COLLECTION, ids=INDEX_IDS)
    stored = {row["chunk_id"]: row["vector"] for row in rows}
    assert stored.keys() == set(INDEX_IDS)
    for chunk_id, vector in zip(INDEX_IDS, INDEX_VECTORS):
        assert stored[chunk_id] == pytest.approx(vector.tolist())


def test_build_index_creates_missing_parent_dir(tmp_path):
    path = tmp_path / "sub" / "index.db"
    client = build_index(INDEX_IDS, INDEX_VECTORS, path, COLLECTION)
    try:
        assert path.exists()
    finally:
        client.close()


def test_build_index_rebuild_replaces_old_data(tmp_path):
    # New ids are disjoint from the old ones: re-inserting an existing id doesn't
    # add a row, so a rebuild that reused the old ids would pass even without a drop.
    path = tmp_path / "index.db"
    build_index(INDEX_IDS, INDEX_VECTORS, path, COLLECTION).close()
    client = build_index(["x#0", "y#0"], INDEX_VECTORS[:2], path, COLLECTION)
    try:
        assert client.get_collection_stats(COLLECTION)["row_count"] == 2
        assert len(client.get(COLLECTION, ids=INDEX_IDS)) == 0
    finally:
        client.close()


def test_build_index_inserts_more_rows_than_one_batch(tmp_path):
    # 2500 rows: several full insert batches plus a partial last one.
    n = 2500
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"d{i}#0" for i in range(n)]
    client = build_index(ids, vectors, tmp_path / "index.db", COLLECTION)
    try:
        assert client.get_collection_stats(COLLECTION)["row_count"] == n
    finally:
        client.close()


@pytest.mark.parametrize(
    "ids, vectors, message",
    [
        (INDEX_IDS[:-1], INDEX_VECTORS, r"len\(ids\)=2 != len\(vectors\)=3"),
        ([], np.empty((0, DIM), dtype=np.float32), "nothing to index"),
    ],
    ids=["length-mismatch", "empty"],
)
def test_build_index_bad_input_raises(tmp_path, ids, vectors, message):
    with pytest.raises(ValueError, match=message):
        build_index(ids, vectors, tmp_path / "index.db", COLLECTION)
