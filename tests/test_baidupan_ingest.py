from src.ingest.baidupan_ingest import BaiduPanIngestor


def test_ingestor_init():
    ing = BaiduPanIngestor()
    assert hasattr(ing, "blogger_paths")


def test_blogger_paths_defined():
    ing = BaiduPanIngestor()
    assert "博士" in ing.blogger_paths
    assert "梅森" in ing.blogger_paths
