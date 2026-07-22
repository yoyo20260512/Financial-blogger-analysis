"""
飞书文档接入 — 读取指定 Wiki 文档内容
"""
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FEISHU_BASE = "https://open.feishu.cn/open-apis"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "飞书"


class FeishuAuth:
    """飞书 tenant_access_token 管理"""

    def __init__(self):
        self.app_id = os.getenv("FEISHU_APP_ID", "")
        self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self._token = None
        self._expire_at = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id) and bool(self.app_secret)

    def get_token(self) -> str:
        import time
        now = time.time()
        if self._token and now < self._expire_at - 60:
            return self._token
        resp = httpx.post(
            f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=15,
        )
        payload = resp.json()
        if payload.get("code", 0) != 0:
            raise RuntimeError(f"Feishu auth failed: {payload.get('msg')}")
        self._token = payload["tenant_access_token"]
        self._expire_at = now + payload.get("expire", 7200)
        return self._token


class FeishuDocIngestor:
    """飞书文档接入器"""

    # 目标文档 token
    WIKI_TOKEN = "KCt9wXwo8i8eTmkJ45tcmCUqnEh"

    def __init__(self):
        self.auth = FeishuAuth()

    @property
    def available(self) -> bool:
        return self.auth.is_configured

    def fetch_doc(self) -> str:
        """读取飞书 Wiki 文档内容并保存到本地"""
        if not self.available:
            logger.warning("Feishu not configured")
            return ""

        token = self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}

        # 尝试解析 wiki 节点，找到 doc 真实 token
        doc_token = self.WIKI_TOKEN
        try:
            resp = httpx.get(
                f"{FEISHU_BASE}/wiki/v2/spaces/get_node",
                headers=headers,
                params={"token": self.WIKI_TOKEN},
                timeout=15,
            )
            p = resp.json()
            if p.get("code") == 0:
                node = p.get("data", {}).get("node", {})
                obj_token = node.get("obj_token")
                obj_type = node.get("obj_type", "")
                if obj_token and obj_type == "docx":
                    doc_token = obj_token
        except Exception as e:
            logger.warning("Feishu wiki parse failed: %s", e)

        # 读取 docx 内容
        try:
            resp = httpx.get(
                f"{FEISHU_BASE}/docx/v1/documents/{doc_token}/raw_content",
                headers=headers,
                timeout=30,
            )
            p = resp.json()
            if p.get("code") == 0:
                content = p.get("data", {}).get("content", "")
                # 保存到本地
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                (RAW_DIR / "wiki_doc.txt").write_text(content, encoding="utf-8")
                logger.info("Feishu doc saved: %d chars", len(content))
                return content
        except Exception as e:
            logger.warning("Feishu doc read failed: %s", e)

        return ""
