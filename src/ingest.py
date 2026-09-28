"""Loading, validating and chunking a BEIR-format dataset."""

import csv
import json
import re
from pathlib import Path
from statistics import mean, median

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, SCIFACT_DIR, TOKEN_LIMIT
from schema import Chunk, Document


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into one dict per non-blank line.

    Args:
        path: `.jsonl` file, one JSON object per line.

    Returns:
        The parsed objects in file order; blank lines are skipped.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_corpus(path: Path) -> dict[str, Document]:
    """Load a BEIR `corpus.jsonl` as `{doc_id: Document}`.

    Args:
        path: `corpus.jsonl` with `_id`, `title` and `text` fields per line.

    Returns:
        `{doc_id: Document}` in file order; doc ids stay strings.
    """
    data = load_jsonl(path)
    corpus = {}
    for item in data:
        doc_id = item["_id"]
        title = item["title"]
        text = item["text"]
        corpus[doc_id] = Document(doc_id=doc_id, title=title, text=text)
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Load a BEIR `queries.jsonl` as `{query_id: text}`, all splits included.

    `metadata` is dropped; filter to judged queries with `get_evaluable_queries`.

    Args:
        path: `queries.jsonl` with `_id` and `text` fields per line.

    Returns:
        `{query_id: query text}` in file order.
    """
    data = load_jsonl(path)
    queries = {}
    for item in data:
        query_id = item["_id"]
        query_text = item["text"]
        queries[query_id] = query_text
    return queries


def load_query_relations(path: Path) -> dict[str, dict[str, int]]:
    """Load a qrels TSV as `{query_id: {doc_id: score}}`.

    Args:
        path: tab-separated file with a `query-id`, `corpus-id`, `score` header row.

    Returns:
        `{query_id: {doc_id: score}}` with integer scores; only judged pairs appear.
    """
    qrels_mapping = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for line in reader:
            query_id = line["query-id"]
            corpus_id = line["corpus-id"]
            score = line["score"]
            if query_id not in qrels_mapping:
                qrels_mapping[query_id] = {}
            qrels_mapping[query_id][corpus_id] = int(score)
    return qrels_mapping


def check_consistency(
    corpus: dict[str, Document],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> None:
    """Check that every query and document referenced in qrels exists.

    Args:
        corpus: `{doc_id: Document}` from `load_corpus`.
        queries: `{query_id: text}` from `load_queries`.
        qrels: `{query_id: {doc_id: score}}` from `load_query_relations`.

    Raises:
        ValueError: on the first qrels query id missing from `queries`, or doc id
            missing from `corpus`.
    """
    for query_id in qrels:
        if query_id not in queries:
            raise ValueError(f"Query ID {query_id} in qrels does not exist in queries.")

    for query_id, corpus_scores in qrels.items():
        for corpus_id in corpus_scores:
            if corpus_id not in corpus:
                raise ValueError(
                    f"Corpus ID {corpus_id} in qrels for query {query_id} does not exist in corpus."
                )


def get_evaluable_queries(
    queries: dict[str, str], qrels: dict[str, dict[str, int]]
) -> dict[str, str]:
    """Keep only queries that have relevance judgements (300 of 1109 for SciFact).

    Args:
        queries: `{query_id: text}`, typically all splits.
        qrels: `{query_id: {doc_id: score}}` for the split being evaluated.

    Returns:
        `{query_id: text}` for the queries present in `qrels`, in `queries` order.
    """
    evaluable_queries = {}
    for query_id, query_text in queries.items():
        if query_id in qrels:
            evaluable_queries[query_id] = query_text
    return evaluable_queries


def _length_stats(lengths: list[int], unit: str) -> dict[str, float]:
    """Mean, median, p95 and max of non-empty `lengths`, keyed like `mean_{unit}s`."""
    ordered = sorted(lengths)
    return {
        f"mean_{unit}s": mean(ordered),
        f"median_{unit}s": median(ordered),
        f"p95_{unit}s": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        f"max_{unit}s": ordered[-1],
    }


def get_corpus_stats(
    corpus: dict[str, Document], tokenizer=None, token_limit: int = TOKEN_LIMIT
) -> dict[str, float]:
    """Length stats of each document's title + text.

    Args:
        corpus: `{doc_id: Document}`.
        tokenizer: optional Hugging Face tokenizer; without it only character stats
            are computed. Token counts include special tokens.
        token_limit: model input limit used for the overflow count.

    Returns:
        `num_documents`, then `mean/median/p95/max_chars`; with a tokenizer also
        `mean/median/p95/max_tokens` and `docs_over_{token_limit}_tokens`.
        An empty corpus gives `{"num_documents": 0}`.
    """
    if not corpus:
        return {"num_documents": 0}

    texts = [f"{doc.title} {doc.text}".strip() for doc in corpus.values()]

    stats: dict[str, float] = {"num_documents": len(corpus)}
    stats.update(_length_stats([len(text) for text in texts], "char"))

    if tokenizer is not None:
        token_counts = [len(ids) for ids in tokenizer(texts)["input_ids"]]
        stats.update(_length_stats(token_counts, "token"))
        stats[f"docs_over_{token_limit}_tokens"] = sum(
            1 for count in token_counts if count > token_limit
        )

    return stats


def chunk(
    text: str, tokenizer, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping windows of `size` tokens.

    Windows are cut from the original string via token offsets, so text is
    preserved exactly, but may start or end mid-word.

    Args:
        text: document body to split.
        tokenizer: Hugging Face *fast* tokenizer (needs `return_offsets_mapping`).
        size: window length in tokens, special tokens not counted.
        overlap: tokens shared by consecutive windows; the step is `size - overlap`.

    Returns:
        Chunk strings in document order; empty or whitespace-only text gives `[]`.

    Raises:
        ValueError: If `overlap >= size`.
    """
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)[
        "offset_mapping"
    ]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(text[char_start:char_end])
        if end == len(offsets):
            break
        start += size - overlap
    return chunks


# Ends at . ! ? before whitespace or end of text, but not after a lone capital
# (C. elegans) or e.g, i.e, et al, vs, Fig.
SENTENCE_PATTERN = re.compile(
    r"\S.*?"
    r"(?:(?<!\b[A-Z])(?<!\be\.g)(?<!\bi\.e)(?<!\bal)(?<!\bvs)(?<!\bFig)"
    r"[.!?](?=\s|\Z)|\Z)",
    re.DOTALL,
)


def chunk_by_sentence(
    text: str, tokenizer, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Greedily pack whole sentences into chunks of at most `size` tokens.

    Trailing sentences fitting in `overlap` tokens are repeated in the next
    chunk. A sentence longer than `size` is split with `chunk`.

    Args:
        text: document body to split; sentences are found with `SENTENCE_PATTERN`.
        tokenizer: Hugging Face fast tokenizer, used for counting tokens.
        size: maximum chunk length in tokens, special tokens not counted.
        overlap: token budget for the whole sentences carried into the next chunk.

    Returns:
        Chunk strings sliced from `text`, in document order; text with no
        non-whitespace characters gives `[]`.

    Raises:
        ValueError: If `overlap >= size`.
    """
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    sentences = list(SENTENCE_PATTERN.finditer(text))
    if not sentences:
        return []
    token_counts = [
        len(ids)
        for ids in tokenizer([s.group() for s in sentences], add_special_tokens=False)[
            "input_ids"
        ]
    ]

    chunks = []
    current = []
    current_tokens = 0

    def close_current():
        if current:
            chunks.append(
                text[sentences[current[0]].start() : sentences[current[-1]].end()]
            )

    for i, n_tokens in enumerate(token_counts):
        if n_tokens > size:
            close_current()
            chunks.extend(chunk(sentences[i].group(), tokenizer, size, overlap))
            current, current_tokens = [], 0
            continue

        if current_tokens + n_tokens > size:
            close_current()
            carried, carried_tokens = [], 0
            for j in reversed(current):
                if carried_tokens + token_counts[j] > overlap:
                    break
                carried.insert(0, j)
                carried_tokens += token_counts[j]
            if carried_tokens + n_tokens > size:
                carried, carried_tokens = [], 0
            current, current_tokens = carried, carried_tokens

        current.append(i)
        current_tokens += n_tokens

    close_current()
    return chunks


def chunk_corpus(
    corpus: dict[str, Document],
    tokenizer,
    chunker=chunk,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict[str, Chunk]:
    """Split every document's text with `chunker` and return `{chunk_id: Chunk}`.

    Only `doc.text` is chunked; the title is copied onto each chunk unchanged.

    Args:
        corpus: `{doc_id: Document}`.
        tokenizer: passed through to `chunker`.
        chunker: `chunk`, `chunk_by_sentence`, or any function with the signature
            `(text, tokenizer, size, overlap) -> list[str]`.
        size: maximum chunk length in tokens, passed to `chunker`.
        overlap: overlap in tokens, passed to `chunker`.

    Returns:
        `{chunk_id: Chunk}` with `chunk_id = '{doc_id}#{position}'`, ordered by
        document then position. Documents with empty text get no chunks.

    Raises:
        ValueError: from `chunker` if `overlap >= size`.
    """
    chunks = {}
    for doc_id, doc in corpus.items():
        for position, text in enumerate(chunker(doc.text, tokenizer, size, overlap)):
            chunk_id = f"{doc_id}#{position}"
            chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                position=position,
                title=doc.title,
                text=text,
            )
    return chunks


def load_scifact_data(
    path: Path,
) -> tuple[dict[str, Document], dict[str, str], dict[str, dict[str, int]]]:
    """Load corpus, queries and test qrels from a BEIR-format directory.

    Args:
        path: directory containing `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.

    Returns:
        `(corpus, queries, qrels)`: `{doc_id: Document}`, `{query_id: text}` for all
        splits, and `{query_id: {doc_id: score}}` for the test split.
    """
    corpus = load_corpus(path / "corpus.jsonl")
    queries = load_queries(path / "queries.jsonl")
    qrels = load_query_relations(path / "qrels" / "test.tsv")
    return corpus, queries, qrels


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    corpus, queries, qrels = load_scifact_data(SCIFACT_DIR)
    check_consistency(corpus, queries, qrels)

    print(f"Corpus stats with tokenizer: {get_corpus_stats(corpus, tokenizer)}")

    for chunker in (chunk, chunk_by_sentence):
        chunks = chunk_corpus(corpus, tokenizer, chunker=chunker)
        indexed = [f"passage: {c.title}. {c.text}" for c in chunks.values()]
        token_counts = [len(ids) for ids in tokenizer(indexed)["input_ids"]]
        chunked_docs = {c.doc_id for c in chunks.values()}
        print(
            f"{chunker.__name__}: {len(chunks)} chunks from {len(corpus)} docs "
            f"({len(chunks) / len(corpus):.2f} per doc), "
            f"max indexed tokens: {max(token_counts)} (limit {TOKEN_LIMIT}), "
            f"docs without chunks: {len(corpus.keys() - chunked_docs)}"
        )
