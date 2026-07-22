"""
文本分块 — 按语义段落/字符数切分，支持重叠
"""
import logging
import re
import tiktoken

logger = logging.getLogger(__name__)

# 中文适合使用 cl100k_base 编码
ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    将文本切分为重叠的块

    参数:
        text: 清洗后的文本
        metadata: 元数据（博主、来源、类型、时间等）
        chunk_size: 每块的目标 token 数
        overlap: 相邻块重叠 token 数

    返回:
        [{"text": "...", "metadata": {...}, "chunk_id": 0}, ...]
    """
    if not text:
        return []

    # 按段落拆分
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0

    def flush():
        nonlocal current_chunk, current_tokens, chunk_id
        if current_chunk:
            chunk_text_joined = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text_joined,
                "metadata": {**metadata, "chunk_id": chunk_id},
                "tokens": len(ENCODER.encode(chunk_text_joined)),
            })
            chunk_id += 1
            # 重叠：保留最后一个段落的一部分
            overlap_texts = []
            overlap_tokens = 0
            for p in reversed(current_chunk):
                pt = len(ENCODER.encode(p))
                if overlap_tokens + pt <= overlap:
                    overlap_texts.insert(0, p)
                    overlap_tokens += pt
                else:
                    break
            current_chunk = overlap_texts
            current_tokens = overlap_tokens

    for para in paragraphs:
        para_tokens = len(ENCODER.encode(para))

        # 单个段落超过 chunk_size，需要强行拆分
        if para_tokens > chunk_size:
            flush()
            # 按句子拆分
            sentences = re.split(r"([。！？!\?])", para)
            temp = ""
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                if len(ENCODER.encode(temp + sentence)) > chunk_size and temp:
                    chunks.append({
                        "text": temp.strip(),
                        "metadata": {**metadata, "chunk_id": chunk_id},
                        "tokens": len(ENCODER.encode(temp.strip())),
                    })
                    chunk_id += 1
                    temp = sentence
                else:
                    temp += sentence
            if temp:
                current_chunk = [temp.strip()]
                current_tokens = len(ENCODER.encode(temp.strip()))
            continue

        # 加上新段落会超 chunk_size，刷新
        if current_tokens + para_tokens > chunk_size and current_chunk:
            flush()

        current_chunk.append(para)
        current_tokens += para_tokens

    # 最后一块
    flush()
    return chunks
