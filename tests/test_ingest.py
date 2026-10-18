"""
Scenarios:
chunk and chunk_by_sentence (shared):
- empty text gives []
- overlap >= size raises (equal, greater)
- every chunk is a substring of the input
chunk:
- text shorter than size is one chunk, equal to the text
- the last window isn't repeated
- each chunk re-tokenizes to at most size + 2 tokens (slice-edge effect, see test)
chunk_by_sentence:
- chunks end at sentence boundaries
- the last sentence of a chunk opens the next one when it fits in overlap
- a sentence longer than size is split
- no chunk is longer than size tokens
SENTENCE_PATTERN:
- abbreviations (et al., e.g., i.e., vs., Fig., lone capitals) don't end a sentence
- ".", "!" and "?" end sentences; text without punctuation is one sentence
- known limitations (xfail): "Dr.", an initial right before a sentence end
chunk_corpus:
- empty corpus gives {}
- chunk ids are '{doc_id}#{position}', positions run 0..n-1, title is the doc's
- chunks are ordered by document (corpus order), then position
- a document with empty or whitespace-only text gets no chunks
loaders:
- load_jsonl skips blank lines and keeps file order
- load_corpus keeps doc ids as strings (leading zeros survive), in file order
- load_queries drops metadata, in file order
- load_query_types reads metadata.type, in file order
- load_query_types raises on a query without metadata.type (no type, no metadata)
- load_query_relations groups by query, scores are ints, score-0 rows are kept
- load_beir reads corpus.jsonl, queries.jsonl and qrels/test.tsv from one directory
check_consistency and get_evaluable_queries:
- consistent data passes
- an unknown query id or doc id in qrels raises
- only judged queries are kept, in queries order
get_corpus_stats:
- empty corpus gives only num_documents
- character stats are computed on "title text"
"""

import json

import pytest

from ingest import (
    SENTENCE_PATTERN,
    check_consistency,
    chunk,
    chunk_by_sentence,
    chunk_corpus,
    get_corpus_stats,
    get_evaluable_queries,
    load_beir,
    load_corpus,
    load_jsonl,
    load_queries,
    load_query_types,
    load_query_relations,
)
from schema import Document


def _n_tokens(tokenizer, text):
    """Token count of text without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _write_jsonl(path, rows):
    """Write dicts as JSON Lines, one per line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_qrels(path, rows):
    """Write (query_id, doc_id, score) rows as a BEIR qrels TSV with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore"] + [f"{q}\t{d}\t{s}" for q, d, s in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# chunk and chunk_by_sentence

BOTH_CHUNKERS = pytest.mark.parametrize(
    "chunker", [chunk, chunk_by_sentence], ids=["chunk", "chunk_by_sentence"]
)


@BOTH_CHUNKERS
def test_empty_text_returns_empty_list(tokenizer, chunker):
    assert chunker("", tokenizer, size=10, overlap=2) == []


@BOTH_CHUNKERS
@pytest.mark.parametrize("size, overlap", [(5, 5), (5, 10)], ids=["equal", "greater"])
def test_overlap_not_smaller_than_size_raises(tokenizer, chunker, size, overlap):
    with pytest.raises(ValueError, match="Overlap must be smaller than chunk size"):
        chunker("This is a test.", tokenizer, size=size, overlap=overlap)


@BOTH_CHUNKERS
def test_every_chunk_is_in_input_text(tokenizer, chunker):
    text = "This is a test sentence. After the test sentence, here is another test sentence."
    chunks = chunker(text, tokenizer, size=5, overlap=2)
    assert len(chunks) > 1
    for chk in chunks:
        assert chk in text


# chunk


def test_chunk_text_shorter_than_size_returns_single_chunk(tokenizer):
    text = "This is a short text."
    assert chunk(text, tokenizer, size=50, overlap=10) == [text]


def test_chunk_last_window_is_not_repeated(tokenizer):
    text = "This is a test. This is another test. This is the last test."
    chunks = chunk(text, tokenizer, size=10, overlap=5)
    assert len(chunks) == 3
    assert chunks[-1] != chunks[-2]


def test_chunk_retokenized_length_is_close_to_size(tokenizer):
    # A window holds exactly `size` tokens of the full text's tokenization, but
    # re-tokenizing the slice alone can add a couple: at the window's start the
    # word-boundary marker becomes its own token ("▁architecture" in context ->
    # "▁", "architecture" alone), and a mid-word cut splits differently.
    # Seen on SciFact: at most +2 with size=10.
    text = "This is a test, and here are more tests. Finally, one last test."
    size = 10
    for chk in chunk(text, tokenizer, size=size, overlap=5):
        assert _n_tokens(tokenizer, chk) <= size + 2


# chunk_by_sentence


def test_chunk_by_sentence_ends_at_sentence_boundaries(tokenizer):
    text = "This is a test. This is another test. This is the one more test. And this is the last test."
    chunks = chunk_by_sentence(text, tokenizer, size=10, overlap=3)
    assert len(chunks) > 1
    for chk in chunks:
        assert chk.endswith((".", "!", "?"))


def test_chunk_by_sentence_overlap_repeats_last_sentence(tokenizer):
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_by_sentence(text, tokenizer, size=12, overlap=5)
    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:]):
        assert SENTENCE_PATTERN.findall(current)[-1] == SENTENCE_PATTERN.findall(following)[0]


def test_chunk_by_sentence_sentence_longer_than_size_is_split(tokenizer):
    text = "This is a very long sentence that exceeds the chunk size limit."
    assert len(chunk_by_sentence(text, tokenizer, size=5, overlap=2)) > 1


def test_chunk_by_sentence_no_chunk_is_longer_than_size(tokenizer):
    text = (
        "This is a very long test sentence to check that no chunk is longer than size. "
        "And short sentence. And more sentences. One more."
    )
    chunks = chunk_by_sentence(text, tokenizer, size=5, overlap=2)
    assert len(chunks) > 1
    for chk in chunks:
        assert _n_tokens(tokenizer, chk) <= 5


# SENTENCE_PATTERN


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Smith et al. showed it. Next.", ["Smith et al. showed it.", "Next."]),
        ("e.g. this is an example. Next.", ["e.g. this is an example.", "Next."]),
        pytest.param(
            "Dr. Smith is here. Next.",
            ["Dr. Smith is here.", "Next."],
            marks=pytest.mark.xfail(reason="Dr is not in the abbreviation list"),
        ),
        (
            "This is a test! Is it working? Yes.",
            ["This is a test!", "Is it working?", "Yes."],
        ),
        ("No punctuation here", ["No punctuation here"]),
        (
            "C. elegans is a model organism. Next.",
            ["C. elegans is a model organism.", "Next."],
        ),
        ("Fig. 1 shows the results. Next.", ["Fig. 1 shows the results.", "Next."]),
        (
            "This is a test. e.g. this is an example. Next.",
            ["This is a test.", "e.g. this is an example.", "Next."],
        ),
        (
            "This is a test. i.e. this is an example. Next.",
            ["This is a test.", "i.e. this is an example.", "Next."],
        ),
        (
            "This is a test. vs. this is another test. Next.",
            ["This is a test.", "vs. this is another test.", "Next."],
        ),
        pytest.param(
            "Tyson vs. triple G. Tomorrow at 13:00 at the Global stadium.",
            ["Tyson vs. triple G.", "Tomorrow at 13:00 at the Global stadium."],
            marks=pytest.mark.xfail(
                reason="lone-capital rule can't tell initials from sentence ends"
            ),
        ),
    ],
    ids=[
        "et-al",
        "eg-at-start",
        "dr",
        "exclamation-question",
        "no-punctuation",
        "lone-capital",
        "fig",
        "eg-mid-text",
        "ie-mid-text",
        "vs-mid-text",
        "initial-before-sentence-end",
    ],
)
def test_sentence_pattern_splits_correctly(text, expected):
    assert SENTENCE_PATTERN.findall(text) == expected


# chunk_corpus

LONG_TEXT = "One two three four. Five six seven eight. Nine ten eleven twelve. Thirteen."


def test_chunk_corpus_empty_corpus_returns_empty_dict(tokenizer):
    assert chunk_corpus({}, tokenizer, chunker=chunk) == {}


@BOTH_CHUNKERS
def test_chunk_corpus_ids_positions_and_titles(tokenizer, chunker):
    corpus = {
        "doc1": Document("doc1", "Title 1", LONG_TEXT),
        "doc2": Document("doc2", "Title 2", "Short text."),
    }
    chunks = chunk_corpus(corpus, tokenizer, chunker=chunker, size=6, overlap=2)
    assert {c.doc_id for c in chunks.values()} == set(corpus)
    for doc_id, doc in corpus.items():
        doc_chunks = [c for c in chunks.values() if c.doc_id == doc_id]
        assert [c.position for c in doc_chunks] == list(range(len(doc_chunks)))
        for c in doc_chunks:
            assert c.chunk_id == f"{doc_id}#{c.position}"
            assert chunks[c.chunk_id] is c
            assert c.title == doc.title
    assert len([c for c in chunks.values() if c.doc_id == "doc1"]) > 1


@BOTH_CHUNKERS
def test_chunk_corpus_orders_by_document_then_position(tokenizer, chunker):
    corpus = {
        "b": Document("b", "B", LONG_TEXT),
        "a": Document("a", "A", LONG_TEXT),
    }
    chunks = chunk_corpus(corpus, tokenizer, chunker=chunker, size=6, overlap=2)
    keys = [(c.doc_id, c.position) for c in chunks.values()]
    assert keys == sorted(keys, key=lambda k: (list(corpus).index(k[0]), k[1]))


@BOTH_CHUNKERS
@pytest.mark.parametrize("text", ["", "   \n"], ids=["empty", "whitespace"])
def test_chunk_corpus_document_without_text_gets_no_chunks(tokenizer, chunker, text):
    corpus = {
        "full": Document("full", "Full", "Some text."),
        "blank": Document("blank", "Title only", text),
    }
    chunks = chunk_corpus(corpus, tokenizer, chunker=chunker)
    assert {c.doc_id for c in chunks.values()} == {"full"}


# loaders


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n   \n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        path,
        [
            {"_id": "007", "title": "T1", "text": "Body one."},
            {"_id": "2", "title": "", "text": "Body two."},
        ],
    )
    corpus = load_corpus(path)
    assert corpus == {
        "007": Document("007", "T1", "Body one."),
        "2": Document("2", "", "Body two."),
    }
    assert list(corpus) == ["007", "2"]


def test_load_queries_drops_metadata(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_jsonl(
        path,
        [
            {"_id": "h2", "text": "Query two", "metadata": {"type": "keyword"}},
            {"_id": "1", "text": "Query one", "metadata": {}},
        ],
    )
    queries = load_queries(path)
    assert queries == {"h2": "Query two", "1": "Query one"}
    assert list(queries) == ["h2", "1"]


def test_load_query_types(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_jsonl(
        path,
        [
            {"_id": "h2", "text": "Query two", "metadata": {"type": "keyword"}},
            {"_id": "h1", "text": "Query one", "metadata": {"type": "paraphrase"}},
        ],
    )
    types = load_query_types(path)
    assert types == {"h2": "keyword", "h1": "paraphrase"}
    assert list(types) == ["h2", "h1"]


@pytest.mark.parametrize(
    "row",
    [
        {"_id": "h2", "text": "Query two", "metadata": {}},
        {"_id": "h2", "text": "Query two"},
    ],
    ids=["no-type", "no-metadata"],
)
def test_load_query_types_missing_type_raises(tmp_path, row):
    path = tmp_path / "queries.jsonl"
    _write_jsonl(path, [{"_id": "h1", "text": "q", "metadata": {"type": "keyword"}}, row])
    with pytest.raises(ValueError, match="Query h2 has no metadata.type"):
        load_query_types(path)


def test_load_query_relations(tmp_path):
    path = tmp_path / "test.tsv"
    _write_qrels(path, [("1", "10", 1), ("1", "20", 0), ("2", "10", 2)])
    assert load_query_relations(path) == {"1": {"10": 1, "20": 0}, "2": {"10": 2}}


def test_load_beir(tmp_path):
    _write_jsonl(tmp_path / "corpus.jsonl", [{"_id": "10", "title": "T", "text": "x"}])
    _write_jsonl(tmp_path / "queries.jsonl", [{"_id": "1", "text": "q", "metadata": {}}])
    _write_qrels(tmp_path / "qrels" / "test.tsv", [("1", "10", 1)])
    corpus, queries, qrels = load_beir(tmp_path)
    assert corpus == {"10": Document("10", "T", "x")}
    assert queries == {"1": "q"}
    assert qrels == {"1": {"10": 1}}


# check_consistency and get_evaluable_queries

CORPUS = {"10": Document("10", "T", "x")}
QUERIES = {"1": "q"}


def test_check_consistency_accepts_consistent_data():
    assert check_consistency(CORPUS, QUERIES, {"1": {"10": 1}}) is None


@pytest.mark.parametrize(
    "qrels, message",
    [
        ({"9": {"10": 1}}, "Query ID 9 in qrels does not exist"),
        ({"1": {"99": 1}}, "Corpus ID 99 in qrels for query 1 does not exist"),
    ],
    ids=["unknown-query", "unknown-doc"],
)
def test_check_consistency_raises(qrels, message):
    with pytest.raises(ValueError, match=message):
        check_consistency(CORPUS, QUERIES, qrels)


def test_get_evaluable_queries_keeps_judged_in_queries_order():
    queries = {"3": "c", "1": "a", "2": "b"}
    qrels = {"1": {"10": 1}, "3": {"10": 1}, "4": {"10": 1}}
    evaluable = get_evaluable_queries(queries, qrels)
    assert evaluable == {"3": "c", "1": "a"}
    assert list(evaluable) == ["3", "1"]


# get_corpus_stats


def test_get_corpus_stats_empty_corpus():
    assert get_corpus_stats({}) == {"num_documents": 0}


def test_get_corpus_stats_char_lengths():
    # "Ab cd" -> 5, "xyz" (empty title, stripped) -> 3, "T" (empty text) -> 1.
    # Sorted [1, 3, 5]: p95 index = min(int(3 * 0.95), 2) = 2 -> 5.
    corpus = {
        "1": Document("1", "Ab", "cd"),
        "2": Document("2", "", "xyz"),
        "3": Document("3", "T", ""),
    }
    assert get_corpus_stats(corpus) == pytest.approx(
        {
            "num_documents": 3,
            "mean_chars": (5 + 3 + 1) / 3,
            "median_chars": 3,
            "p95_chars": 5,
            "max_chars": 5,
        }
    )
