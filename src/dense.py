"""Dense retrieval: embedding-model input formatting, embeddings and vector search."""

import hashlib
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

from config import ARTIFACTS_DIR, PASSAGE_PREFIX, QUERY_PREFIX
from schema import Chunk

BATCH_SIZE = 1024
ENCODE_BATCH_SIZE = 32


def format_passage(chunk: Chunk, with_title: bool = True) -> str:
    """Build the string the embedding model sees for one chunk.

    The title and body are joined with ". ", or with a single space when the title
    already ends in ".", "!" or "?", so no ".." appears. The body is never modified.

    Args:
        chunk: the chunk to index; its `title` and `text` are used.
        with_title: whether to put the source document's title before the body.

    Returns:
        `PASSAGE_PREFIX` + optional title and separator + `chunk.text` unchanged,
        e.g. `"passage: Aspirin and stroke. We studied 40 patients."`.
    """
    if with_title and chunk.title.strip():
        stripped_title = chunk.title.strip()
        separator = " " if stripped_title.endswith((".", "!", "?")) else ". "
        return f"{PASSAGE_PREFIX}{stripped_title}{separator}{chunk.text}"
    return f"{PASSAGE_PREFIX}{chunk.text}"


def format_query(text: str) -> str:
    """Build the string the embedding model sees for one query.

    Args:
        text: raw query text, as loaded from `queries.jsonl`.

    Returns:
        `QUERY_PREFIX` + `text` unchanged, e.g. `"query: Aspirin prevents stroke"`.
    """
    return f"{QUERY_PREFIX}{text}"


def cache_path(
    model_name: str, chunker_name: str, size: int, overlap: int, with_title: bool
) -> Path:
    """Path of the embedding cache file for one model + chunking configuration.

    Pure: builds the path only and creates nothing on disk. The key covers the
    config, not the chunker's code, so staleness is also checked against the
    file's contents when it is loaded.

    Args:
        model_name: Hugging Face model id, may contain "/"
            (e.g. "intfloat/multilingual-e5-small"); "/" becomes "--".
        chunker_name: the chunker function's `__name__`, e.g. "chunk_by_sentence".
        size: chunk size in tokenizer tokens.
        overlap: chunk overlap in tokenizer tokens.
        with_title: whether passages were embedded with the document title.

    Returns:
        A `.npz` path directly under `ARTIFACTS_DIR`, e.g.
        `emb__intfloat--multilingual-e5-small__chunk_by_sentence__400_50__title.npz`.
    """
    model = model_name.replace("/", "--")
    title_tag = "title" if with_title else "notitle"
    return (
        ARTIFACTS_DIR
        / f"emb__{model}__{chunker_name}__{size}_{overlap}__{title_tag}.npz"
    )


def embed_chunks(
    chunks: dict[str, Chunk], model, path: Path, with_title: bool = True
) -> tuple[list[str], np.ndarray]:
    """Embed every chunk's passage string, reusing the cache file when it still matches.

    The cache stores the ids, the vectors and a sha256 fingerprint of the exact
    passage strings. It is reused only when both the ids (in order) and the
    fingerprint match, so changed text, chunking or title setting is re-embedded
    even at the same path; a stale file is overwritten.

    Args:
        chunks: `{chunk_id: Chunk}` from `chunk_corpus`; dict order sets the row order.
        model: a loaded SentenceTransformer, or anything with the same
            `encode(texts, **kwargs) -> np.ndarray` method.
        path: `.npz` cache file, normally from `cache_path`; missing parent
            directories are created.
        with_title: whether to put each chunk's document title before its body.

    Returns:
        `(ids, vectors)`: `ids` is `list(chunks)`; `vectors` is a float32 array of
        shape `(len(ids), embedding_dim)` whose row i is the unit-length
        embedding of `ids[i]`.

    Raises:
        ValueError: if `chunks` is empty.
    """
    if not chunks:
        raise ValueError("chunks is empty: nothing to embed")

    ids = list(chunks)
    texts = [format_passage(c, with_title) for c in chunks.values()]
    # "\0" separator: without one, ["ab", "c"] and ["a", "bc"] would hash the same.
    fingerprint = hashlib.sha256("\0".join(texts).encode()).hexdigest()

    cached = _load_cache(path, ids, fingerprint)
    if cached is not None:
        return ids, cached

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, vectors=vectors, fingerprint=fingerprint)
    return ids, vectors


def _load_cache(path: Path, ids: list[str], fingerprint: str) -> np.ndarray | None:
    """Cached vectors at `path` if its ids and fingerprint match, else None."""
    if not path.exists():
        return None
    with np.load(path) as cache:
        if cache["ids"].tolist() == ids and str(cache["fingerprint"]) == fingerprint:
            return cache["vectors"]
    return None


def build_index(
    ids: list[str], vectors: np.ndarray, db_path: Path, collection: str
) -> MilvusClient:
    """
    Build a Milvus Lite collection with one row per id and vector.

    Args:
        ids: list of chunk ids, one per row.
        vectors: float32 array of shape (len(ids), embedding_dim) with unit-length rows
        db_path: path to the Milvus Lite database directory; missing parent directories are created.
        collection: name of the Milvus collection to create; if it exists, it is dropped and recreated.

    Returns:
        MilvusClient connected to the new collection.

    Raises:
        ValueError: if `ids` is empty or if `len(ids) != len(vectors)`.
    """
    if not ids:
        raise ValueError("ids is empty: nothing to index")
    if len(ids) != len(vectors):
        raise ValueError(f"len(ids)={len(ids)} != len(vectors)={len(vectors)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = MilvusClient(str(db_path))
    schema = client.create_schema(auto_id=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])

    index = client.prepare_index_params()
    index.add_index("vector", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection):
        client.drop_collection(collection)
    client.create_collection(collection, schema=schema, index_params=index)

    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start : start + BATCH_SIZE]
        batch_vectors = vectors[start : start + BATCH_SIZE]
        client.insert(
            collection,
            [
                {"chunk_id": i, "vector": v.tolist()}
                for i, v in zip(batch_ids, batch_vectors)
            ],
        )
    return client


def embed_queries(queries: dict[str, str], model) -> tuple[list[str], np.ndarray]:
    """
    Embed every query string.

    Args:
        queries: `{query_id: query_text}` from `queries.jsonl`; dict order sets the row order.
        model: a loaded SentenceTransformer, or anything with the same
            `encode(texts, **kwargs) -> np.ndarray` method.

    Returns:
        `(ids, vectors)`:
            `ids` is `list(queries)`;
            `vectors` is a float32 array of shape `(len(ids), embedding_dim)` whose row i is the unit-length embedding of `ids[i]`.

    Raises:
        ValueError: if `queries` is empty.
    """
    if not queries:
        raise ValueError("queries is empty: nothing to embed")

    ids = list(queries)
    texts = [format_query(q) for q in queries.values()]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    return ids, vectors


def search(
    client: MilvusClient,
    collection: str,
    query_ids: list[str],
    query_vectors: np.ndarray,
    depth: int,
) -> dict[str, list[str]]:
    """
    Search the Milvus collection for the top `depth` nearest neighbors of each query vector.

    Args:
        client: MilvusClient connected to the collection.
        collection: name of the Milvus collection to search.
        query_ids: one id per query; `query_ids[i]` belongs to row i of `query_vectors`.
        query_vectors: float32 array of shape (len(query_ids), embedding_dim) with
            unit-length rows, normally from `embed_queries`.
        depth: number of chunks to return per query; use ≥ max(EVAL_KS), since
            `evaluate` dedupes chunks of the same doc.

    Returns:
        A run `{query_id: [chunk_id, ...]}` with keys in `query_ids` order. Each list is
        ranked best first by inner product and has `min(depth, row count)` entries,
        e.g. `{"1": ["31715818#2", "31715818#0", "4983#0"]}`.

    Raises:
        ValueError: if `query_ids` is empty, `len(query_ids) != len(query_vectors)`,
            or `depth < 1`.
    """
    if not query_ids:
        raise ValueError("query_ids is empty: nothing to search")
    if len(query_ids) != len(query_vectors):
        raise ValueError(
            f"len(query_ids)={len(query_ids)} != len(query_vectors)={len(query_vectors)}"
        )
    if depth <= 0:
        raise ValueError(f"depth={depth} is not a positive integer")

    results = client.search(
        collection,
        query_vectors.tolist(),
        anns_field="vector",
        search_params={"metric_type": "IP"},
        limit=depth,
        output_fields=["chunk_id"],
    )

    return {
        query_id: [hit["chunk_id"] for hit in hits]
        for query_id, hits in zip(query_ids, results)
    }
