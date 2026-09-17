import pytest
from transformers import AutoTokenizer

from config import EMBEDDING_MODEL


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
