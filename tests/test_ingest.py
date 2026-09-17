import pytest

from ingest import SENTENCE_PATTERN, chunk, chunk_by_sentence, chunk_corpus
from schema import Document

# Test cases for the chunk function


def test_chunk_empty_text_returns_empty_list(tokenizer):
    assert chunk("", tokenizer, size=10, overlap=2) == []


def test_chunk_raises_value_error_when_overlap_greater_than_or_equal_to_size(tokenizer):
    with pytest.raises(ValueError):
        chunk("This is a test.", tokenizer, size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk("This is a test.", tokenizer, size=5, overlap=10)


def test_text_shorter_than_size_returns_single_chunk(tokenizer):
    text = "This is a short text."
    assert chunk(text, tokenizer, size=50, overlap=10) == [text]


def test_the_last_chunk_is_not_repeated(tokenizer):
    text = "This is a test. This is another test. This is the last test."
    chunks = chunk(text, tokenizer, size=10, overlap=5)
    assert len(chunks) == 3
    assert chunks[-1] != chunks[-2]


def test_every_chunk_is_in_input_text(tokenizer):
    text = "This is a test. Here are more tests. And one more test."
    chunks = chunk(text, tokenizer, size=10, overlap=5)
    for chk in chunks:
        assert chk in text


def test_every_chunk_has_at_most_size_tokens(tokenizer):
    text = "This is a test, and here are more tests. Finally, one last test."
    chunks = chunk(text, tokenizer, size=10, overlap=5)
    for chk in chunks:
        assert len(tokenizer(chk, add_special_tokens=False)["input_ids"]) <= 12


# Test cases for the chunk_by_sentence function


def test_chunk_by_sentence_empty_text_returns_empty_list(tokenizer):
    assert chunk_by_sentence("", tokenizer, size=10, overlap=2) == []


def test_chunk_by_sentence_ends_at_sentence_boundaries(tokenizer):
    text = "This is a test. This is another test. This is the one more test. And this is the last test."
    chunks = chunk_by_sentence(text, tokenizer, size=10, overlap=3)
    assert len(chunks) > 1
    for chk in chunks:
        assert chk.endswith((".", "!", "?"))


def test_chunk_by_sentence_with_small_size_chunks_overlapping_sentences(tokenizer):
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_by_sentence(text, tokenizer, size=12, overlap=5)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        last_sentence_of_current_chunk = SENTENCE_PATTERN.findall(chunks[i])[-1]
        first_sentence_of_next_chunk = SENTENCE_PATTERN.findall(chunks[i + 1])[0]
        assert last_sentence_of_current_chunk == first_sentence_of_next_chunk


def test_chunk_by_sentence_sentence_longer_than_size_is_split(tokenizer):
    text = "This is a very long sentence that exceeds the chunk size limit."
    chunks = chunk_by_sentence(text, tokenizer, size=5, overlap=2)
    assert len(chunks) > 1


def test_chunk_by_sentence_splits_into_real_chunks(tokenizer):
    text = "This is a test sentence. After the test sentence, here is another test sentence."
    chunks = chunk_by_sentence(text, tokenizer, size=5, overlap=2)
    assert len(chunks) > 1
    for chk in chunks:
        assert chk in text


def test_chunk_by_sentence_no_chunk_is_longer_than_size(tokenizer):
    text = "This is a very long test sentence to check that no chunk is longer than size. And short sentnece. And more sentences. One more."
    chunks = chunk_by_sentence(text, tokenizer, size=5, overlap=2)
    assert len(chunks) > 1
    for chk in chunks:
        assert len(tokenizer(chk, add_special_tokens=False)["input_ids"]) <= 5


# Test cases for SENTENCE_PATTERN


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
)
def test_sentence_pattern_splits_correctly(text, expected):
    assert SENTENCE_PATTERN.findall(text) == expected


# Test cases for the chunk_corpus function


def test_chunk_corpus_empty_corpus_returns_empty_dict(tokenizer):
    assert chunk_corpus({}, tokenizer, chunker=chunk) == {}


@pytest.mark.parametrize("chunker", [chunk, chunk_by_sentence])
def test_chunk_corpus_chunk_id_looks_like_document_id_with_index(tokenizer, chunker):
    corpus = {
        "doc1": Document(
            doc_id="doc1",
            title="Test Document 1",
            text="This is the first test document.",
        ),
        "doc2": Document(
            doc_id="doc2",
            title="Test Document 2",
            text="This is another test document.",
        ),
    }

    chunked_corpus = chunk_corpus(corpus, tokenizer, chunker=chunker)

    for chk in chunked_corpus.values():
        assert chk.chunk_id == f"{chk.doc_id}#{chk.position}"
        assert chk.title == corpus[chk.doc_id].title
    doc_ids_in_chunks = {chk.doc_id for chk in chunked_corpus.values()}
    assert doc_ids_in_chunks == set(corpus.keys())
