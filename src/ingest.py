"""Loading, validating and chunking a BEIR-format dataset."""

import csv
import json
import re
from pathlib import Path
from statistics import mean, median

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, SCIFACT_DIR, TOKEN_LIMIT
from schema import Chunk, Document


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into one dict per non-blank line."""
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_corpus(path: Path) -> dict[str, Document]:
    """Load `corpus.jsonl` as `{doc_id: Document}`."""
    data = load_jsonl(path)
    corpus = {}
    for item in data:
        doc_id = item["_id"]
        title = item["title"]
        text = item["text"]
        corpus[doc_id] = Document(doc_id=doc_id, title=title, text=text)
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Load `queries.jsonl` as `{query_id: text}`, all splits included."""
    data = load_jsonl(path)
    queries = {}
    for item in data:
        query_id = item["_id"]
        query_text = item["text"]
        queries[query_id] = query_text
    return queries


def load_query_relations(path: Path) -> dict[str, dict[str, int]]:
    """Load a qrels TSV (`query-id`, `corpus-id`, `score`) as `{query_id: {doc_id: score}}`."""
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
    """Raise ValueError if qrels reference a query or document that doesn't exist."""
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
    """Keep only queries that have relevance judgements (300 of 1109 for SciFact)."""
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
    """Length stats of title + text: chars always, tokens and overflow count with a tokenizer."""
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
    """Split every document with `chunker` and return `{chunk_id: Chunk}`."""
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
    """Load corpus, queries and test qrels from a SciFact directory."""
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
