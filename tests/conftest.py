import pytest
from tokenizers import AutoTokenizer

from src.config import EMBEDDING_MODEL


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
