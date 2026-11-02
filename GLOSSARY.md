# RAG from scratch

A retrieval-augmented generation pipeline built stage by stage, where every retrieval
stage is scored against the same relevance judgements.

## Corpus

**Document**:
One source text (title + body) as it appears in the corpus; the unit relevance is judged on.
_Avoid_: Article, abstract, passage

**Chunk**:
A contiguous piece of one Document's body; the unit that is indexed and retrieved.
_Avoid_: Passage, segment, split

**Chunker**:
A strategy for cutting a Document into Chunks (fixed-size or sentence-packed).
_Avoid_: Splitter, segmenter

## Evaluation

**Judgement**:
One statement that a given Document is relevant to a given query.
_Avoid_: Label, qrel (for a single row)

**Query type**:
Whether a hand-written query reuses its Document's words (keyword) or avoids them (paraphrase).

**Run**:
A retriever's output: for each query, a ranked list of Chunks, best first.
_Avoid_: Results, predictions

**Depth**:
How many Chunks a retriever returns per query in a Run.
_Avoid_: Top-k, limit

**Cutoff**:
How many Documents a metric scores, after a Run's Chunks are collapsed into Documents (the k in recall@k).
_Avoid_: Depth, top-k

**Long-document slice**:
The queries with at least one relevant Document too long to fit in a single Chunk;
the only queries on which Chunkers can differ.
_Avoid_: Multi-chunk queries

**Baseline**:
The winning dense-retrieval configuration of Milestone 3; every later stage is compared against it.
_Avoid_: Default, reference run
