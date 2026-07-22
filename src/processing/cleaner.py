"""
文本清洗 — 中文文本去噪、规范化
"""
import re


def clean_text(text: str) -> str:
    """清洗原始文本"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格转半角
    text = text.replace("　", " ")
    # 移除 URL
    text = re.sub(r"https?://\S+", "", text)
    # 合并连续空格
    text = re.sub(r" +", " ", text)
    # 合并连续空行（最多保留 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除首尾空白
    text = text.strip()
    return text


def remove_boilerplate(text: str) -> str:
    """移除常见页眉页脚、版权声明"""
    patterns = [
        r"第\s*\d+\s*页[，,]\s*共\s*\d+\s*页",
        r"^\s*[-—]+\s*\d+\s*[-—]+\s*$",
        r"版权所有[©\s]*",
        r"免责声明.*?(?=\n|$)",
        r"仅供内部.*?(?=\n|$)",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)
    return clean_text(text)
