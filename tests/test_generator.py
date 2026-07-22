from src.rag.generator import answer_question, summarize_blogger_style


def test_answer_question_returns_str():
    result = answer_question("什么是价值投资？")
    assert isinstance(result, str)
    assert len(result) > 0
