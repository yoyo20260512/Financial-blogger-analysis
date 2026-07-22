"""
向量化 + ChromaDB 存储
"""
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "chroma_db"
# 使用已缓存的 sentence-transformers 中文模型（无需下载）
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


class EmbeddingManager:
    """Embedding 生成 + ChromaDB 向量存储管理"""

    def __init__(self, collection_name: str = "blogger_rag"):
        self.collection_name = collection_name
        self._model = None
        self._client = None
        self._collection = None

    def _get_model(self):
        """延迟加载 embedding 模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBED_MODEL)
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _get_collection(self):
        """获取 ChromaDB collection（延迟初始化）"""
        if self._collection is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[dict]):
        """
        向向量库添加文本块

        参数:
            chunks: [{"text": "...", "metadata": {...}}, ...]
        """
        if not chunks:
            return

        model = self._get_model()
        collection = self._get_collection()

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{m.get('blogger', 'unknown')}_{m.get('chunk_id', i)}_{i}"
               for i, m in enumerate(metadatas)]

        logger.info("Embedding %d chunks...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d chunks to ChromaDB", len(chunks))

    def search(self, query: str, n_results: int = 10, filter_dict: Optional[dict] = None) -> list[dict]:
        """
        检索相关文本块

        参数:
            query: 查询文本
            n_results: 返回结果数
            filter_dict: 过滤条件，如 {"blogger": "博士"}

        返回:
            [{"text": "...", "metadata": {...}, "distance": 0.xx}, ...]
        """
        model = self._get_model()
        collection = self._get_collection()

        query_embedding = model.encode([query]).tolist()[0]

        where = filter_dict if filter_dict else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            })
        return output

    def get_collection_stats(self) -> dict:
        """获取向量库统计信息"""
        collection = self._get_collection()
        count = collection.count()
        # 按博主分组统计
        blogger_counts = {}
        all_metadatas = collection.get(limit=count)["metadatas"]
        for m in all_metadatas:
            blogger = m.get("blogger", "unknown")
            blogger_counts[blogger] = blogger_counts.get(blogger, 0) + 1
        return {"total_chunks": count, "by_blogger": blogger_counts}
