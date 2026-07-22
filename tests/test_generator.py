"""LLM Generator 测试"""
import os
import pytest
from src.rag.generator import answer_question


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="需要 OPENAI_API_KEY")
def test_answer_question_returns_str():
    """验证 answer_question 返回字符串"""
    result = answer_question("什么是价值投资？")
    assert isinstance(result, str)
