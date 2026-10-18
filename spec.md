# Spec

The contracts between pipeline stages: data formats, ids, interfaces and metric
definitions. Anything marked *planned* isn't implemented yet; update this file when it is.

## 1. Pipeline

```
corpus.jsonl ──load──▶ {doc_id: Document} ──chunk_corpus──▶ {chunk_id: Chunk}
                                                                   │ index
queries.jsonl ──load──▶ {query_id: text} ──retriever──▶ run {query_id: [chunk_id, ...]}
                                                                   │
qrels/test.tsv ──load──▶ {query_id: {doc_id: score}} ──────▶ evaluate(run, qrels, EVAL_KS)
                                                                   │
                                                  {"recall@k" | "mrr@k" | "ndcg@k": mean}
```

Every retriever (dense, BM25, hybrid, reranked) produces a **run** in the same format,
so every stage is scored by the same `evaluate` call.

## 2. Ids

| Id | Type | Format | Example |
|---|---|---|---|
| `doc_id` | `str` | as in the source corpus; always a string, even when numeric | `"31715818"` |
| `chunk_id` | `str` | `'{doc_id}#{position}'`, position 0-based within the doc | `"31715818#2"` |
| `query_id` | `str` | as in the source; hand-written queries use `h1`…`h20` | `"1"`, `"h7"` |

- Collections are dicts keyed by id, never positional lists.
- To get a doc id from a chunk id, use `chunk_id.rsplit("#", 1)[0]`. It uses `rsplit` so a `#` inside a doc id survives.

## 3. On-disk data (BEIR format)

Each dataset directory:

```
<dir>/corpus.jsonl      {"_id", "title", "text"}           one doc per line
<dir>/queries.jsonl     {"_id", "text", "metadata": {...}}  one query per line
<dir>/qrels/test.tsv    header "query-id  corpus-id  score", tab-separated
```

| Dataset | Path (`config`) | Contents |
|---|---|---|
| SciFact | `SCIFACT_DIR` = `data/scifact/` | 5183 docs, 1109 queries (all splits); test qrels: 300 queries, 339 judgements, all score 1 |
| Hand-written | `HANDWRITTEN_DIR` = `data/handwritten/` | 20 queries + qrels only; **uses the SciFact corpus** |

Hand-written set details:
- One relevant doc per query, score 1.
- `metadata.type` is `"keyword"` (reuses the abstract's words) or `"paraphrase"` (avoids them), 10 of each.
- The questions were checked against competing docs, but not against every doc, so an unjudged relevant doc is possible.
- `load_queries` drops `metadata`. Reading the type will need a small loader (*planned*).

## 4. In-memory types (`src/schema.py`)

```python
Document(doc_id: str, title: str, text: str)
Chunk(chunk_id: str, doc_id: str, position: int, title: str, text: str)
```

- `Chunk.text` is the body only. The title is combined at **indexing time**, so title vs no-title stays an experiment switch.
- Indexed string for e5: `"passage: {title}. {text}"`, built by `dense.format_passage` (see §8). Query string: `f"query: {text}"`.

## 5. Loading (`src/ingest.py`)

| Function | Returns |
|---|---|
| `load_corpus(path)` | `{doc_id: Document}` |
| `load_queries(path)` | `{query_id: text}`, all splits |
| `load_query_relations(path)` | `{query_id: {doc_id: score}}` |
| `load_scifact_data(dir)` | `(corpus, queries, test qrels)` |
| `get_evaluable_queries(queries, qrels)` | only the queries that have qrels |
| `check_consistency(corpus, queries, qrels)` | `None`; raises `ValueError` if qrels reference an unknown query or doc |

## 6. Chunking (`src/ingest.py`)

Chunker contract: `(text: str, tokenizer, size: int, overlap: int) -> list[str]`

- Size and overlap are in **tokenizer tokens** (e5 tokenizer), not words or characters.
- Chunks are **slices of the original string** (token offsets or sentence spans), never `tokenizer.decode`, so the text is preserved exactly.
- `overlap >= size` raises `ValueError`.

| Chunker | Behaviour |
|---|---|
| `chunk` | Fixed windows of `size` tokens with `overlap`. May cut mid-word. |
| `chunk_by_sentence` | Greedy packing of whole sentences up to `size`. Trailing sentences fitting in `overlap` repeat in the next chunk. A sentence longer than `size` falls back to `chunk`. |

`chunk_corpus(corpus, tokenizer, chunker, size, overlap) -> {chunk_id: Chunk}`

Defaults (`config`):
- `CHUNK_SIZE = 400`, `CHUNK_OVERLAP = 50`, `TOKEN_LIMIT = 512`.
- 400 leaves room for special tokens, `passage: ` and titles (max 80 tokens). The indexed string is at most 485 tokens.

## 7. Runs and evaluation (`src/evaluate.py`)

**Run**: `dict[str, list[str]]`, `{query_id: [chunk_id, ...]}`, each list ranked best first.
- The list may contain several chunks of the same doc.
- Its length is the retrieval depth. It should be ≥ `max(EVAL_KS)`.

**`map_chunks_to_docs(ranked_chunk_ids) -> list[str]`**: strips `#position` and dedupes. Each doc keeps the rank of its best chunk.
`["1#0", "2#1", "1#1"]` → `["1", "2"]`.

**`evaluate(run, qrels, ks) -> dict[str, float]`**, for each query in `qrels`:
1. `ranked_doc_ids = map_chunks_to_docs(run.get(query_id, []))`. A query missing from the run scores 0.
2. `relevant = {doc_id with score > 0}`.
3. Score every metric at every k.

It returns the mean over qrels queries, keyed `"recall@k"`, `"mrr@k"`, `"ndcg@k"` and grouped by metric. Queries in the run but not in qrels are ignored. Empty qrels, or a query with no positive judgement, raise `ValueError`.

`EVAL_KS = (1, 5, 10, 20)`. 20 is the reranker's candidate pool, so recall@20 is the most reranking can achieve.

### Metrics

All metrics are per query, use binary relevance, and look only at the top k. `R` = the relevant set, `rank` is 1-based.

| Metric | Definition | Range and notes |
|---|---|---|
| `recall_at_k` | \|R ∩ top-k\| / \|R\| | [0, 1]. Can't reach 1 when \|R\| > k. |
| `reciprocal_rank` | 1 / rank of the first relevant doc in the top k, else 0 | [0, 1]. The mean over queries = MRR@k. |
| `ndcg_at_k` | DCG / IDCG, DCG = Σ over relevant docs in the top k of 1 / log₂(rank + 1) | [0, 1]. IDCG is computed on min(k, \|R\|) ideal hits, so it depends only on the judgements. |

All three raise `ValueError` if `R` is empty or `k < 1`.

Sanity baselines (`.venv/bin/python src/evaluate.py`):
- A random run ≈ 0 everywhere.
- A perfect run = 1, except SciFact recall@1 = 0.955 (23 test queries have more than 1 relevant doc).

## 8. Dense retrieval (`src/dense.py`)

| Function | Behaviour |
|---|---|
| `format_passage(chunk, with_title=True) -> str` | `PASSAGE_PREFIX` + stripped title + `". "` (just `" "` if the title already ends in `.`/`!`/`?`) + body unchanged. An empty title or `with_title=False` gives prefix + body. |
| `format_query(text) -> str` | `QUERY_PREFIX` + text. |
| `cache_path(model_name, chunker_name, size, overlap, with_title) -> Path` | `ARTIFACTS_DIR / "emb__{model with / → --}__{chunker}__{size}_{overlap}__{title\|notitle}.npz"`. Builds the path only. |
| `embed_chunks(chunks, model, path, with_title=True) -> (ids, vectors)` | `ids = list(chunks)`; `vectors` float32 `(len(ids), dim)`, unit-length, row i ↔ `ids[i]`. Raises `ValueError` on empty `chunks`. |

Embedding cache (`.npz`): keys `ids`, `vectors`, `fingerprint` (sha256 of the passage strings joined with `"\0"`). The cache is reused only if the ids (in order) **and** the fingerprint match. Otherwise the chunks are re-embedded and the file is overwritten. `model` is anything with `encode(texts, **kwargs)`; the tests pass a fake.

## 9. Planned interfaces

*Planned. Record the real signature here once written.*

- **Retrievers** (`dense.py`, `bm25.py`): take queries and a depth, return a run.
- **Hybrid** (`hybrid.py`): runs in, a fused run out (reciprocal rank fusion, `1 / (60 + rank)`).
- **Rerank** (`rerank.py`): a run plus chunk texts in, a reordered run (top ~20) out.
- **Generate** (`generate.py`): a query plus top chunks in, an answer citing `chunk_id`s out. There is an explicit "not found in documents" answer when the context doesn't support one.
- **Caches** in `artifacts/`: reranker scores, keyed like the embedding cache (§8), so switching chunker or model never reuses stale scores.
