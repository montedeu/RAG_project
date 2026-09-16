"""Project-wide settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SCIFACT_DIR = PROJECT_ROOT / "data" / "scifact"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOKEN_LIMIT = 512

# Chunk budget leaves room for special tokens, "passage: " and the title (max 80 tokens).
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
