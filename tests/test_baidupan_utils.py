"""BaiduPan 工具类测试"""
import pytest
from src.ingest.baidupan_utils import BaiduPanClient


def test_client_init_from_env():
    """验证客户端可从环境变量初始化（无token时不会报错）"""
    client = BaiduPanClient()
    # 只是验证初始化不抛异常
    assert isinstance(client.is_configured, bool)


def test_list_folder_returns_list():
    """验证 list_files 始终返回列表（空数据或无token时返回空列表）"""
    client = BaiduPanClient()
    files = client.list_files("/笔记梳理")
    assert isinstance(files, list)
