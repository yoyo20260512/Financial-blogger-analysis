"""
百度网盘工具类 - 封装文件列表、下载、AI纪要接口
"""
import json, logging, os, time
from datetime import datetime
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PAN_BASE = "https://pan.baidu.com"
PCS_BASE = "https://pcs.baidu.com/rest/2.0/pcs/file"
APAAS_BASE = "https://pan.baidu.com"


class BaiduPanClient:
    """百度网盘客户端（认证 + 文件操作 + AI纪要）"""

    def __init__(self):
        self.api_key = os.getenv("BAIDUPAN_API_KEY", "")
        self.app_secret = os.getenv("BAIDUPAN_APP_SECRET", "")
        self._access_token = os.getenv("BAIDUPAN_ACCESS_TOKEN", "")
        self._refresh_token = os.getenv("BAIDUPAN_REFRESH_TOKEN", "")

    @property
    def is_configured(self) -> bool:
        return bool(self._access_token)

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def list_files(self, path: str = "/笔记梳理") -> list[dict]:
        """列出网盘目录下的文件"""
        if not self._access_token:
            logger.warning("BaiduPan: no access token")
            return []

        params = {
            "access_token": self._access_token,
            "method": "list",
            "dir": path,
            "order": "time",
            "desc": 1,
            "num": 200,
        }
        try:
            resp = httpx.get(f"{PAN_BASE}/rest/2.0/xpan/file", params=params, timeout=30)
            data = resp.json()
            if data.get("errno") != 0:
                logger.warning("BaiduPan list errno=%s", data.get("errno"))
                return []
            results = []
            for item in data.get("list", []):
                results.append({
                    "fs_id": item.get("fs_id", 0),
                    "filename": item.get("server_filename", ""),
                    "path": item.get("path", ""),
                    "size": item.get("size", 0),
                    "isdir": item.get("isdir", 0),
                    "category": item.get("category", 0),
                    # 1=视频, 2=音频, 3=图片, 4=文档
                    "mtime": item.get("server_mtime", 0),
                })
            return results
        except Exception as e:
            logger.error("BaiduPan list error: %s", e)
            return []

    def download_text(self, fsid: int) -> str:
        """下载文字文件内容（.txt/.md）"""
        if not self._access_token:
            return ""
        try:
            resp = httpx.get(
                f"{PCS_BASE}",
                params={"method": "download", "access_token": self._access_token, "fsid": fsid},
                timeout=60, follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            logger.warning("BaiduPan download error: %s", e)
        return ""

    def submit_transcribe(self, fsid: int, filename: str = "") -> dict:
        """提交AI纪要视频转写任务"""
        if not self._access_token:
            return {"status": "error", "message": "no token"}
        try:
            resp = httpx.post(
                f"{APAAS_BASE}/apaas/v1/api/mediainsight/taskcreate",
                params={"access_token": self._access_token},
                json={
                    "fsid": fsid,
                    "language": "zh",
                    "parameters": {
                        "ai_outline_enable": True,
                        "ai_outline": {"module_names": ["fullSummary", "segmentedSummary", "realRecord"]}
                    }
                },
                timeout=30,
            )
            data = resp.json()
            if data.get("errno") != 0 or not data.get("task_id"):
                return {"status": "error", "message": f"创建任务失败: {data.get('errno')}"}
            return {"status": "ok", "task_id": data["task_id"]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def query_transcribe(self, task_id: str) -> dict:
        """查询AI纪要转写结果"""
        if not self._access_token:
            return {"status": "error", "message": "no token"}
        try:
            resp = httpx.get(
                f"{APAAS_BASE}/apaas/v1/api/mediainsight/taskquery",
                params={"access_token": self._access_token, "task_id": task_id},
                timeout=30,
            )
            data = resp.json()
            status = data.get("status", 0)
            if status == 300:  # 完成
                result = data.get("result", {})
                record = result.get("realRecord", {})
                text = record.get("text", "") or json.dumps(record, ensure_ascii=False)
                summary = result.get("fullSummary", {}).get("text", "")
                return {"status": "ok", "text": text, "summary": summary, "done": True}
            elif status in (0, 100, 200):
                return {"status": "ok", "text": "", "summary": "", "done": False}
            else:
                return {"status": "error", "message": f"转写失败: status={status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
