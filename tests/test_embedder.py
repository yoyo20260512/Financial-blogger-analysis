import shutil

import pytest

from src.processing.embedder import CHROMA_DIR, EmbeddingManager


@pytest.fixture(autouse=True)
def cleanup_chroma():
    """Clean ChromaDB between tests so each run starts fresh."""
    yield
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def test_embedder_init():
    em = EmbeddingManager()
    assert em.collection_name == "blogger_rag"


def test_add_and_search():
    em = EmbeddingManager()
    chunks = [
        {"text": "投资要长期持有优质公司", "metadata": {"blogger": "博士", "source": "test"}},
        {"text": "短线操作需要注意风险控制", "metadata": {"blogger": "梅森", "source": "test"}},
    ]
    em.add_chunks(chunks)
    results = em.search("长期投资", n_results=5)
    assert len(results) >= 1
