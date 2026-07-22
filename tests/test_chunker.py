from src.processing.chunker import chunk_text


def test_chunk_split():
    text = "。".join(["第{}段内容".format(i) for i in range(20)])
    chunks = chunk_text(text, {"source": "test"}, chunk_size=300, overlap=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert "text" in c
        assert "metadata" in c
        assert "source" in c["metadata"]


def test_chunk_metadata():
    text = "测试内容" * 100
    chunks = chunk_text(text, {"blogger": "博士", "source": "pdf"})
    assert all(c["metadata"]["blogger"] == "博士" for c in chunks)
