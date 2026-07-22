"""Test RAG retriever module"""

import shutil

import pytest

from src.processing.embedder import CHROMA_DIR, EmbeddingManager
from src.rag.retriever import retrieve, get_stats


@pytest.fixture(scope="module", autouse=True)
def seeded_chroma():
    """Seed ChromaDB once at module scope (retriever is read-only)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    em = EmbeddingManager()
    chunks = [
        {"text": "投资要长期持有优质公司", "metadata": {"blogger": "博士", "source": "test"}},
        {"text": "短线操作需要注意风险控制", "metadata": {"blogger": "梅森", "source": "test"}},
        {"text": "市场分析显示消费板块有潜力", "metadata": {"blogger": "博士", "source": "test"}},
    ]
    em.add_chunks(chunks)
    yield


def test_retrieve_returns_list():
    results = retrieve("投资策略", n=5)
    assert isinstance(results, list)


def test_retrieve_with_blogger_filter():
    results = retrieve("市场分析", blogger="博士", n=5)
    for r in results:
        assert r["metadata"]["blogger"] == "博士"


def test_retrieve_returns_correct_number():
    results = retrieve("投资", n=2)
    assert len(results) <= 2


def test_retrieve_result_structure():
    results = retrieve("投资", n=1)
    assert len(results) >= 1
    item = results[0]
    assert "text" in item
    assert "metadata" in item
    assert "distance" in item


def test_get_stats_returns_dict():
    stats = get_stats()
    assert isinstance(stats, dict)
    assert "total_chunks" in stats
    assert "by_blogger" in stats
    assert stats["total_chunks"] >= 3
