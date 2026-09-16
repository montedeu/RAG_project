"""Data types shared across the pipeline."""

from dataclasses import dataclass


@dataclass
class Document:
    """A corpus document. `doc_id` stays a string even when numeric-looking."""

    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    """A piece of a document; the unit that gets indexed and retrieved.

    Attributes:
        chunk_id: Corpus-wide unique id, `'{doc_id}#{position}'`.
        doc_id: Id of the source document.
        position: 0-based index within the document.
        title: Source document title, kept separate so indexing decides whether to prepend it.
        text: Body text of the chunk.
    """

    chunk_id: str
    doc_id: str
    position: int
    title: str
    text: str
