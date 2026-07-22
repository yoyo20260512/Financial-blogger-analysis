"""
百度网盘数据接入 — 扫描目录、下载/转写、本地缓存
"""
import json
import logging
import os
import time
from pathlib import Path

from src.ingest.baidupan_utils import BaiduPanClient
from src.ingest.file_parser import parse_text

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


class BaiduPanIngestor:
    """百度网盘接入器：按博主扫描 → 下载 → 解析 → 存 data/raw/"""

    def __init__(self):
        self.client = BaiduPanClient()
        # 博主 → 网盘路径映射
        self.blogger_paths = {
            "博士": "/笔记梳理/博士",
            "梅森": "/笔记梳理/梅森",
            "爽姐": "/a投资/爽姐",
            "老姚": "/a投资/2026老姚前瞻班直播【持续更新至26年底】",
        }
        self.cache_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / "data" / "processed" / "baidupan_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, fsid: int) -> Path:
        return self.cache_dir / f"{fsid}.json"

    def _load_cache(self) -> dict:
        """加载已处理的文件缓存"""
        cache_file = self.cache_dir / "processed.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
        return {"processed_fsids": []}

    def _save_cache(self, cache: dict):
        cache_file = self.cache_dir / "processed.json"
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    def run(self) -> dict:
        """全量运行接入流程"""
        if not self.client.is_configured:
            logger.warning("BaiduPan not configured, skipping")
            return {"status": "skipped", "reason": "not configured"}

        cache = self._load_cache()
        processed = set(cache.get("processed_fsids", []))
        results = {"videos": 0, "documents": 0, "total": 0, "bloggers": {}}

        for blogger, pan_path in self.blogger_paths.items():
            blogger_dir = RAW_DIR / blogger
            blogger_dir.mkdir(parents=True, exist_ok=True)

            files = self.client.list_files(pan_path)
            blogger_count = {"videos": 0, "documents": 0}

            for f in files:
                fsid = f["fs_id"]
                filename = f["filename"]
                category = f.get("category", 0)
                ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

                if fsid in processed:
                    continue

                # 处理视频 — 提交AI纪要
                if category == 1 and ext in ("mp4", "mov", "avi", "mkv"):
                    result = self.client.submit_transcribe(fsid, filename)
                    if result.get("status") == "ok":
                        cache_entry = {
                            "fsid": fsid, "filename": filename,
                            "task_id": result.get("task_id", ""),
                            "blogger": blogger, "text": "", "summary": "",
                        }
                        (self.cache_dir / f"{fsid}.json").write_text(
                            json.dumps(cache_entry, ensure_ascii=False)
                        )
                        processed.add(fsid)
                        blogger_count["videos"] += 1
                        results["videos"] += 1
                        logger.info("  → 已提交AI纪要: %s", filename[:50])

                # 处理文档类 — 下载并解析
                elif category == 4 and ext in ("txt", "md", "pdf", "ppt", "pptx"):
                    content_bytes = None
                    try:
                        import httpx
                        resp = httpx.get(
                            "https://pcs.baidu.com/rest/2.0/pcs/file",
                            params={
                                "method": "download",
                                "access_token": self.client._access_token,
                                "fsid": fsid,
                            },
                            timeout=60, follow_redirects=True,
                        )
                        if resp.status_code == 200:
                            content_bytes = resp.content
                    except Exception as e:
                        logger.warning("Download error %s: %s", filename, e)

                    if content_bytes:
                        text = parse_text(filename, content_bytes)
                        if text:
                            # 存到 data/raw/{blogger}/
                            safe_name = filename.rsplit(".", 1)[0][:80].replace(" ", "_")
                            out_path = blogger_dir / f"{safe_name}.txt"
                            out_path.write_text(text, encoding="utf-8")
                            processed.add(fsid)
                            blogger_count["documents"] += 1
                            results["documents"] += 1
                            logger.info("  → 已保存: %s (%d chars)", filename, len(text))

            results["bloggers"][blogger] = blogger_count
            results["total"] += blogger_count["videos"] + blogger_count["documents"]

        cache["processed_fsids"] = list(processed)
        self._save_cache(cache)
        return {"status": "ok", **results}

    def check_pending_transcribes(self) -> list[dict]:
        """检查所有待处理的AI纪要任务"""
        completed = []
        cache = self._load_cache()
        for fsid_str in sorted(os.listdir(self.cache_dir)):
            if not fsid_str.endswith(".json") or fsid_str == "processed.json":
                continue
            cpath = self.cache_dir / fsid_str
            try:
                entry = json.loads(cpath.read_text())
            except Exception:
                continue
            task_id = entry.get("task_id", "")
            if task_id and not entry.get("text"):
                result = self.client.query_transcribe(task_id)
                if result.get("done") and result.get("text"):
                    entry["text"] = result["text"]
                    entry["summary"] = result.get("summary", "")
                    entry["task_id"] = ""
                    cpath.write_text(json.dumps(entry, ensure_ascii=False))
                    # 同时写入 data/raw/
                    blogger = entry.get("blogger", "unknown")
                    blogger_dir = RAW_DIR / blogger
                    blogger_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = entry.get("filename", str(fsid_str)).rsplit(".", 1)[0][:80].replace(" ", "_")
                    out_path = blogger_dir / f"{safe_name}.txt"
                    out_path.write_text(result["text"], encoding="utf-8")
                    completed.append({"fsid": entry["fsid"], "filename": entry["filename"], "text_length": len(result["text"])})
                    logger.info("  ✓ 转写完成: %s", entry.get("filename", ""))
        return completed
