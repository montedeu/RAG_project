"""Project-wide settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SCIFACT_DIR = PROJECT_ROOT / "data" / "scifact"
# Hand-written queries over the SciFact corpus; ids h1..h20, metadata.type keyword/paraphrase.
HANDWRITTEN_DIR = PROJECT_ROOT / "data" / "handwritten"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOKEN_LIMIT = 512

# Chunk budget leaves room for special tokens, "passage: " and the title (max 80 tokens).
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Cutoffs reported by evaluate(); 20 matches the reranker's candidate pool.
EVAL_KS = (1, 5, 10, 20)
