# RAG from scratch

A retrieval-augmented generation system built **without orchestration frameworks**: no
LangChain, no LlamaIndex. Every stage (chunking, dense retrieval, BM25, hybrid fusion,
reranking, generation) is written by hand and **measured against an evaluation set**, so
each design choice is backed by numbers rather than defaults.

Everything runs locally on CPU: open models, a file-based vector database and a local
LLM. No paid APIs.

## Pipeline

```
corpus ──chunk──▶ chunks ──embed──▶ Milvus ─┐
                     └──────index──▶ BM25  ─┼─▶ fusion (RRF) ─▶ reranker ─▶ LLM answer
query ──────────────────────────────────────┘                               citing chunk ids
```

Every retriever returns a **run**, `{query_id: [chunk_id, ...]}` ranked best first. The
same `evaluate` function scores all of them, so stages are directly comparable.

## Status

| # | Milestone | State |
|---|---|---|
| 1 | Ingestion + chunking: fixed-size windows vs sentence packing | done |
| 2 | Evaluation: recall@k, MRR@k, nDCG@k on SciFact + hand-written queries | done |
| 3 | Dense retrieval baseline (multilingual-e5-small + Milvus Lite) | in progress |
| 4 | BM25 from scratch (lowercasing vs stemming) | planned |
| 5 | Hybrid search with reciprocal rank fusion | planned |
| 6 | Cross-encoder reranking | planned |
| 7 | Generation with citations and a "not found in documents" answer | planned |
| 8 | CLI / API + Docker | planned |

Phase 2 repeats the pipeline on Kazakh (a KazQAD subset).

## Results

SciFact test split (300 queries), k ∈ {1, 5, 10, 20}. One row is added per stage.

| Stage | recall@10 | MRR@10 | nDCG@10 | recall@20 |
|---|---|---|---|---|
| Dense (e5-small) | — | — | — | — |
| BM25 | — | — | — | — |
| Hybrid (RRF) | — | — | — | — |
| + Reranker | — | — | — | — |

No numbers yet: the dense baseline is being built.

## Evaluation

- **SciFact** ([BEIR](https://github.com/beir-cellar/beir)): 5183 scientific abstracts. The
  test qrels judge 300 queries; most have exactly one relevant abstract.
- **Hand-written set**: 20 questions over the same corpus, each labelled `keyword` (reuses
  the abstract's wording) or `paraphrase` (avoids it). Splitting by type shows where
  lexical matching (BM25) and semantic matching (dense) each win.
- **Metrics**: recall@k, MRR@k and nDCG@k with binary relevance. Retrievers rank chunks;
  chunk ids map back to documents (each document keeps its best chunk's rank) before
  scoring, because the judgements are per document.
- **Sanity baselines**: a random run scores ≈ 0 and a perfect run scores 1 (except
  SciFact recall@1 = 0.955, since 23 test queries have more than one relevant document).

Exact formats, ids and metric definitions are in [spec.md](spec.md).

## Stack

| Part | Choice |
|---|---|
| Embeddings | `intfloat/multilingual-e5-small` for experiments, `BAAI/bge-m3` for quality |
| Vector DB | Milvus Lite via `pymilvus` (a single local `.db` file) |
| Lexical | BM25, hand-written |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| LLM | `qwen3:4b` via Ollama |
| Other | numpy, pandas, transformers, sentence-transformers, pytest |

## Setup

Requires Python 3.12. Commands assume Linux or WSL.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

Download SciFact into `data/`:

```bash
mkdir -p data && cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip && cd ..
```

The hand-written queries go in `data/handwritten/` (`queries.jsonl` + `qrels/test.tsv`,
BEIR format).

Models are downloaded from the Hugging Face Hub on first use. Once they're cached, set
`HF_HUB_OFFLINE=1` to skip the network check.

## Usage

Commands run from the repository root:

```bash
.venv/bin/python -m pytest -q tests      # test suite
.venv/bin/python src/evaluate.py         # random/perfect sanity runs for the metrics
.venv/bin/python src/ingest.py           # corpus stats and chunk counts per chunker
```

## Layout

```
src/config.py     paths, model name, chunk size/overlap, evaluation cutoffs
src/schema.py     Document and Chunk dataclasses
src/ingest.py     BEIR loading, validation, corpus stats, both chunkers
src/evaluate.py   metrics, chunk→doc mapping, evaluate(run, qrels, ks)
src/dense.py      passage/query formatting, cached embeddings (Milvus search next)
src/bm25.py, hybrid.py, rerank.py, generate.py   later milestones
tests/            pytest suite (pyproject.toml puts src/ on the path)
spec.md           data formats, interfaces and metric definitions
```

## License

[MIT](LICENSE)
