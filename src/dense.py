"""Dense retrieval: embedding-model input formatting, embeddings and vector search."""

from pathlib import Path

from config import ARTIFACTS_DIR, PASSAGE_PREFIX, QUERY_PREFIX
from schema import Chunk


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
