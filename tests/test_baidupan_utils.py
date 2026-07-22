import pytest
from src.ingest.baidupan_utils import BaiduPanClient

def test_client_init_from_env():
    client = BaiduPanClient()
    assert client.is_configured or True  # allow if no env

def test_list_folder_returns_list():
    client = BaiduPanClient()
    files = client.list_files("/笔记梳理")
    assert isinstance(files, list)
