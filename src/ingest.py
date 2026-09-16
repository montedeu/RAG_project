"""Loading, validating and chunking a BEIR-format dataset.

The only module that knows how the raw files are laid out on disk.
"""
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from statistics import mean, median


@dataclass
class Document:
    """A retrievable document, normalised away from BEIR's field names.

    Attributes:
        doc_id: Identifier, always a string even when numeric-looking (`'4983'`).
        title: Document title.
        text: Body text. Title plus text is what gets indexed.
    """
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    """A chunk of a document, for retrieval and reranking.

    Attributes:
        chunk_id: Corpus-wide unique identifier, `'{doc_id}#{position}'`.
        doc_id: Identifier of the document this chunk came from.
        position: 0-based index of this chunk within its document.
        text: The chunk's text, which is what gets indexed.
    """
    chunk_id: str
    doc_id: str
    position: int
    text: str


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of raw dicts, one per line.

    Args:
        path: A `.jsonl` file — one JSON object per line, not a JSON array.

    Returns:
        One dict per non-blank line, in file order, with keys unrenamed.

    Raises:
        FileNotFoundError: If `path` does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_corpus(path: Path) -> dict[str, Document]:
    """Load `corpus.jsonl` into a mapping of document id to Document.

    Translates BEIR's schema into ours; `metadata` is dropped.

    Args:
        path: Path to `corpus.jsonl`.

    Returns:
        `{doc_id: Document}`, in file order — 5183 documents for SciFact.

    Raises:
        KeyError: If a record lacks `_id`, `title` or `text`.
    """
    data = load_jsonl(path)
    corpus = {}
    for item in data:
        doc_id = item['_id']
        title = item['title']
        text = item['text']
        corpus[doc_id] = Document(doc_id=doc_id, title=title, text=text)
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Load `queries.jsonl` into a mapping of query id to query text.

    Args:
        path: Path to `queries.jsonl`.

    Returns:
        `{query_id: query_text}` for every query in the file — 1109 for SciFact,
        across all splits. Narrow with `get_evaluable_queries` before evaluating.
    """
    data = load_jsonl(path)
    queries = {}
    for item in data:
        query_id = item['_id']
        query_text = item['text']
        queries[query_id] = query_text
    return queries


def load_query_relations(path: Path) -> dict[str, dict[str, int]]:
    """Load a qrels (relevance judgements) TSV.

    Args:
        path: Path to a qrels TSV, e.g. `qrels/test.tsv`. Its header uses
            hyphenated names (`query-id`, `corpus-id`, `score`).

    Returns:
        `{query_id: {corpus_id: score}}` — 300 queries, 339 judgements for
        SciFact, so some queries have several relevant documents. Scores are
        ints; ids stay strings. No inner dict is empty.

    Raises:
        KeyError: If the header lacks the expected hyphenated names.
        ValueError: If a score is not an integer.
    """
    qrels_mapping = {}
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for line in reader:
            query_id = line['query-id']
            corpus_id = line['corpus-id']
            score = line['score']
            if query_id not in qrels_mapping:
                qrels_mapping[query_id] = {}
            qrels_mapping[query_id][corpus_id] = int(score)
    return qrels_mapping


def check_consistency(corpus: dict[str, Document], queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> None:
    """Check that every id referenced by the qrels exists in the other files.

    Args:
        corpus: As returned by `load_corpus`.
        queries: As returned by `load_queries`, before narrowing.
        qrels: As returned by `load_query_relations`.

    Raises:
        ValueError: On the first qrels entry naming a missing query or document.
    """
    for query_id in qrels.keys():
        if query_id not in queries:
            raise ValueError(f"Query ID {query_id} in qrels does not exist in queries.")

    for query_id, corpus_scores in qrels.items():
        for corpus_id in corpus_scores.keys():
            if corpus_id not in corpus:
                raise ValueError(f"Corpus ID {corpus_id} in qrels for query {query_id} does not exist in corpus.")


def get_evaluable_queries(queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict[str, str]:
    """Narrow `queries` to those that have relevance judgements.

    Args:
        queries: The full mapping from `load_queries`.
        qrels: Judgements from `load_query_relations`; only its keys are used.

    Returns:
        The subset of `queries` present in `qrels` — 300 of 1109 for SciFact.
        Evaluating on the rest would count every result as a miss.
    """
    evaluable_queries = {}
    for query_id, query_text in queries.items():
        if query_id in qrels:
            evaluable_queries[query_id] = query_text
    return evaluable_queries


def _length_stats(lengths: list[int], unit: str) -> dict[str, float]:
    """Summarise lengths as mean, median, p95 and max.

    Args:
        lengths: Non-empty list of lengths; sorted internally, caller's list
            left untouched.
        unit: Singular noun used to name the keys, e.g. `'char'` gives
            `mean_chars`, `median_chars`, `p95_chars`, `max_chars`.

    Returns:
        The four values keyed by `unit`. p95 is the nearest-rank percentile.

    Raises:
        StatisticsError: If `lengths` is empty.
    """
    ordered = sorted(lengths)
    return {
        f'mean_{unit}s': mean(ordered),
        f'median_{unit}s': median(ordered),
        f'p95_{unit}s': ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        f'max_{unit}s': ordered[-1],
    }


def get_corpus_stats(corpus: dict[str, Document], tokenizer=None,
                     token_limit: int = 512) -> dict[str, float]:
    """Size distribution of the text that gets indexed (title + text).

    Args:
        corpus: As returned by `load_corpus`.
        tokenizer: Optional HuggingFace tokenizer. Omit for character stats
            only, which keeps this import-free; supply one for exact token
            counts instead of a four-chars-per-token guess.
        token_limit: Context window to count overflows against, used only with
            `tokenizer`. Default 512 matches e5-small; bge-m3 allows 8192.

    Returns:
        `num_documents` plus `mean/median/p95/max_chars`, and with a tokenizer
        the same four in `_tokens` plus `docs_over_<token_limit>_tokens`. For
        SciFact under e5 that overflow count is 712 of 5183. An empty corpus
        returns only `num_documents: 0`.
    """
    if not corpus:
        return {'num_documents': 0}

    texts = [f"{doc.title} {doc.text}".strip() for doc in corpus.values()]

    stats: dict[str, float] = {'num_documents': len(corpus)}
    stats.update(_length_stats([len(text) for text in texts], 'char'))

    if tokenizer is not None:
        token_counts = [len(ids) for ids in tokenizer(texts)['input_ids']]
        stats.update(_length_stats(token_counts, 'token'))
        stats[f'docs_over_{token_limit}_tokens'] = sum(
            1 for count in token_counts if count > token_limit
        )

    return stats


def chunk(text: str, tokenizer, size: int = 450, overlap: int = 50) -> list[str]:
    """Split text into fixed-size overlapping windows, measured in tokens.

    Windows are counted in the tokenizer's tokens but cut from the original
    string via offsets, so case and spacing are preserved exactly.

    Args:
        text: The text to split.
        tokenizer: A fast Hugging Face tokenizer (needs offset mapping), ideally
            the embedding model's own.
        size: Window length in tokens, excluding special tokens. Keep it below
            the model limit to leave room for prefixes like `'passage: '`.
        overlap: Tokens each window shares with its predecessor. Must be smaller
            than `size`.

    Returns:
        Window texts in order, advancing by `size - overlap` tokens. Windows may
        start or end mid-word. Empty text gives an empty list.

    Raises:
        ValueError: If `overlap >= size`, which would never advance and hang.
    """
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
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


def chunk_corpus(corpus: dict[str, Document], tokenizer, size: int = 450,
                 overlap: int = 50) -> dict[str, Chunk]:
    """Chunk every document's text, tagging each piece with where it came from.

    Args:
        corpus: Documents keyed by id, as `load_corpus` returns.
        tokenizer: Passed through to `chunk`.
        size: Passed through to `chunk`.
        overlap: Passed through to `chunk`.

    Returns:
        Chunks keyed by `chunk_id`, in corpus order. A document shorter than
        `size` tokens yields exactly one chunk.
    """
    chunks = {}
    for doc_id, doc in corpus.items():
        for position, text in enumerate(chunk(doc.text, tokenizer, size, overlap)):
            chunk_id = f"{doc_id}#{position}"
            chunks[chunk_id] = Chunk(chunk_id=chunk_id, doc_id=doc_id,
                                     position=position, text=text)
    return chunks


def load_scifact_data(path: Path) -> tuple[dict[str, Document], dict[str, str], dict[str, dict[str, int]]]:
    """Load the three SciFact files from a dataset directory in one call.

    Args:
        path: Directory holding `corpus.jsonl`, `queries.jsonl` and `qrels/`.

    Returns:
        `(corpus, queries, qrels)` as the three loaders produce them —
        unvalidated, queries unnarrowed.

    Raises:
        FileNotFoundError: If any of the three files is missing.
    """
    corpus = load_corpus(path / 'corpus.jsonl')
    queries = load_queries(path / 'queries.jsonl')
    qrels = load_query_relations(path / 'qrels' / 'test.tsv')
    return corpus, queries, qrels


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
    scifact_data_path: Path = Path(__file__).parents[1] / 'data' / 'scifact'
    corpus, queries, qrels = load_scifact_data(scifact_data_path)
    check_consistency(corpus, queries, qrels)

    print(f"Corpus stats with tokenizer: {get_corpus_stats(corpus, tokenizer)}")

    size, overlap = 450, 50
    chunks = chunk_corpus(corpus, tokenizer, size, overlap)
    token_counts = [len(ids) for ids in tokenizer(
        [c.text for c in chunks.values()], add_special_tokens=False)['input_ids']]
    chunked_docs = {c.doc_id for c in chunks.values()}
    print(f"Chunks: {len(chunks)} from {len(corpus)} docs "
          f"({len(chunks) / len(corpus):.2f} per doc), "
          f"max tokens per chunk: {max(token_counts)} (size {size})")
    print(f"Docs without chunks: {len(corpus.keys() - chunked_docs)}")
