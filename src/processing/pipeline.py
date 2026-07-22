"""
处理管道：读取 data/raw/ → 清洗 → 分块 → 向量化 → ChromaDB
"""
import json
import logging
import os
from pathlib import Path

from src.processing.cleaner import clean_text
from src.processing.chunker import chunk_text
from src.processing.embedder import EmbeddingManager

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
CHUNKS_INDEX = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks_index.json"


def run_processing_pipeline() -> dict:
    """全量处理 data/raw/ 中的所有文件"""
    if not RAW_DIR.exists():
        return {"status": "error", "message": "raw dir not found"}

    all_chunks = []
    stats = {"files": 0, "chunks": 0, "bloggers": {}}

    # 遍历 data/raw/{blogger}/*.txt
    for blogger_dir in sorted(RAW_DIR.iterdir()):
        if not blogger_dir.is_dir():
            continue
        blogger = blogger_dir.name
        blogger_chunks = 0
        blogger_files = 0

        for file_path in sorted(blogger_dir.glob("*.txt")):
            if not file_path.is_file():
                continue
            raw_text = file_path.read_text(encoding="utf-8")
            if len(raw_text.strip()) < 20:
                continue

            # 清洗
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            # 提取文件名中的时间信息（如果有）
            source_type = "baidupan"
            if blogger == "飞书":
                source_type = "feishu"

            metadata = {
                "blogger": blogger,
                "source": source_type,
                "filename": file_path.name,
            }

            # 分块
            chunks = chunk_text(cleaned, metadata)
            all_chunks.extend(chunks)
            blogger_chunks += len(chunks)
            blogger_files += 1

        stats["files"] += blogger_files
        stats["chunks"] += blogger_chunks
        stats["bloggers"][blogger] = {"files": blogger_files, "chunks": blogger_chunks}
        logger.info("  %s: %d files → %d chunks", blogger, blogger_files, blogger_chunks)

    if not all_chunks:
        return {"status": "ok", "message": "no data to process", **stats}

    # 写入 chunks_index.json（回溯用）
    CHUNKS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_INDEX.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Chunks index saved: %d chunks", len(all_chunks))

    # 向量化并存入 ChromaDB（先清空再重新写入，避免重复）
    embedder = EmbeddingManager()
    embedder.reset_collection()
    embedder.add_chunks(all_chunks)

    logger.info("Pipeline complete: %d chunks → ChromaDB", len(all_chunks))
    return {"status": "ok", "chunks_processed": len(all_chunks), **stats}
