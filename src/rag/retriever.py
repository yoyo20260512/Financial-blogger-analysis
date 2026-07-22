"""
RAG 检索器 — 从 ChromaDB 检索相关文档块
"""
from typing import Optional

from src.processing.embedder import EmbeddingManager

_embedder = EmbeddingManager()


def retrieve(
    query: str,
    blogger: Optional[str] = None,
    n: int = 10,
) -> list[dict]:
    """
    检索与查询最相关的文档块

    参数:
        query: 用户查询
        blogger: 按博主过滤（可选）
        n: 返回结果数量

    返回:
        [{"text": "...", "metadata": {...}, "distance": 0.xx}, ...]
    """
    filter_dict = None
    if blogger:
        filter_dict = {"blogger": blogger}

    results = _embedder.search(query, n_results=n, filter_dict=filter_dict)
    return results


def get_stats() -> dict:
    """获取向量库统计"""
    return _embedder.get_collection_stats()
