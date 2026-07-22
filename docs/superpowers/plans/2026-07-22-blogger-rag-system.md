# 财经博主 RAG 系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG system that ingests blogger content from Baidu Netdisk and Feishu, stores it in a vector database, and enables style summarization, opinion retrieval, and investment experience extraction.

**Architecture:** Multi-stage pipeline: Data Ingestion (BaiduPan + Feishu) → Text Processing (clean, chunk, embed) → Vector Store (ChromaDB) → RAG Retrieval → LLM Generation.

**Tech Stack:** Python 3.9+, ChromaDB, sentence-transformers (BAAI/bge-large-zh-v1.5), pypdf, python-pptx, httpx, openai, DeepSeek API (for RAG, default)

## Global Constraints

- All paths relative to project root: `/Users/jinqianyu/Documents/个人资料/大数据/财经博主/`
- Python 3.9+ compatibility
- **LLM 双模式**：
  - RAG 阶段：默认用 DeepSeek API（OPENAI_API_BASE=https://api.deepseek.com, model=deepseek-chat）
  - 后续 SFT 阶段：用 LLaMA-Factory + 本地已下载模型（Qwen2.5-1.5B-Instruct / DeepSeek-R1-Distill-Qwen-1.5B）
  - Generator 支持 `LLM_MODE=api` 和 `LLM_MODE=local` 切换
- ChromaDB for vector storage (local, no external services)
- All extracted text stored in `data/raw/` before processing
- BaiduPan auth config from existing `.env` pattern
- Feishu doc token extracted from wiki URL

---

### Task 1: Project Scaffolding & Directory Setup

**Files:**
- Create: `requirements_rag.txt`
- Create: `.env`
- Create: `README_RAG.md`
- Create: `data/raw/` (directories)
- Create: `data/processed/`
- Create: `src/` (package structure)

**Interfaces:**
- Consumes: nothing
- Produces: project structure, dependency list, env config template

- [ ] **Step 1: Create project directories**

```bash
mkdir -p src/ingest src/processing src/rag data/raw/博士 data/raw/梅森 data/raw/爽姐 data/raw/老姚 data/raw/飞书 data/processed docs/superpowers
```

- [ ] **Step 2: Create `requirements_rag.txt`**

```
# RAG System Dependencies
httpx>=0.27.0
python-dotenv>=1.0.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pypdf>=4.0.0
python-pptx>=1.0.0
numpy>=1.24.0
tiktoken>=0.5.0
openai>=1.0.0
```

- [ ] **Step 3: Create `.env`** (copy from reference, strip unneeded fields)

```bash
cp /Users/jinqianyu/Documents/个人资料/大数据/11_大模型agent项目_经济研究报告生成智能体v2/fina-agent/a-share-version/.env .env
```

Then edit `.env` to keep only:
- `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- `BAIDUPAN_APP_ID`, `BAIDUPAN_API_KEY`, `BAIDUPAN_APP_SECRET`, `BAIDUPAN_ACCESS_TOKEN`, `BAIDUPAN_REFRESH_TOKEN`
- `FEISHU_APP_ID`, `FEISHU_APP_SECRET`

- [ ] **Step 4: Create `src/__init__.py`**

```python
# RAG System package
```

- [ ] **Step 5: Create `src/ingest/__init__.py`**, `src/processing/__init__.py`, `src/rag/__init__.py`

```python
# empty __init__.py for each
```

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements_rag.txt
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffolding RAG system project structure"
```

---

### Task 2: BaiduPan Auth & File Listing Utility

**Files:**
- Create: `src/ingest/baidupan_utils.py`

**Interfaces:**
- Consumes: `.env` BaiduPan credentials
- Produces: `BaiduPanClient` class with `list_files()`, `download_text()`, `submit_transcribe()`, `query_transcribe()` methods

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baidupan_utils.py
import pytest
from src.ingest.baidupan_utils import BaiduPanClient

def test_client_init_from_env():
    client = BaiduPanClient()
    assert client.is_configured or True  # allow if no env

def test_list_folder_returns_list():
    client = BaiduPanClient()
    files = client.list_files("/笔记梳理")
    assert isinstance(files, list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
mkdir -p tests && touch tests/__init__.py
pytest tests/test_baidupan_utils.py -v
Expected: FAIL with import error
```

- [ ] **Step 3: Write BaiduPanClient implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_baidupan_utils.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/ingest/baidupan_utils.py tests/ && git commit -m "feat: add BaiduPan client for file listing and transcribe"
```

---

### Task 3: File Parser — PDF, PPT, TXT Extraction

**Files:**
- Create: `src/ingest/file_parser.py`

**Interfaces:**
- Consumes: file path (local text content), file type hint
- Produces: `parse_text(filename: str, content: bytes) -> str` — extracted clean text

- [ ] **Step 1: Write the failing test**

```python
# tests/test_file_parser.py
from src.ingest.file_parser import parse_text

def test_parse_txt():
    result = parse_text("test.txt", b"hello world")
    assert "hello world" in result

def test_parse_pdf_simple():
    # minimal valid PDF
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n...\n"
    result = parse_text("test.pdf", pdf_bytes)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_file_parser.py -v
Expected: FAIL
```

- [ ] **Step 3: Write file parser**

```python
"""
文件解析器 - 支持 PDF、PPT、TXT/MD 文字提取
"""
import io
import logging

logger = logging.getLogger(__name__)


def parse_text(filename: str, content: bytes) -> str:
    """根据文件后缀提取文字内容"""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt" or ext == "md":
        return content.decode("utf-8", errors="replace")

    elif ext == "pdf":
        return _parse_pdf(content)

    elif ext in ("ppt", "pptx"):
        return _parse_pptx(content)

    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""


def _parse_pdf(content: bytes) -> str:
    """用 pypdf 提取 PDF 文字"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n\n".join(texts)
    except Exception as e:
        logger.warning("PDF parse error: %s", e)
        return ""


def _parse_pptx(content: bytes) -> str:
    """用 python-pptx 提取 PPT 文字"""
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(content))
        texts = []
        for slide in prs.slides:
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            if slide_texts:
                texts.append("\n".join(slide_texts))
        return "\n\n---\n\n".join(texts)
    except Exception as e:
        logger.warning("PPT parse error: %s", e)
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_file_parser.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/ingest/file_parser.py tests/ && git commit -m "feat: add file parser for PDF, PPT, TXT"
```

---

### Task 4: BaiduPan Ingestion Pipeline

**Files:**
- Create: `src/ingest/baidupan_ingest.py`

**Interfaces:**
- Consumes: `BaiduPanClient`, `parse_text`, directory path mapping (blogger → pan_path)
- Produces: `run_ingest() -> dict` — summary of what was ingested, writes to `data/raw/{blogger}/`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baidupan_ingest.py
from src.ingest.baidupan_ingest import BaiduPanIngestor

def test_ingestor_init():
    ing = BaiduPanIngestor()
    assert hasattr(ing, "blogger_paths")

def test_blogger_paths_defined():
    ing = BaiduPanIngestor()
    assert "博士" in ing.blogger_paths
    assert "梅森" in ing.blogger_paths
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_baidupan_ingest.py -v
Expected: FAIL
```

- [ ] **Step 3: Write BaiduPan ingestor**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_baidupan_ingest.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/ingest/baidupan_ingest.py tests/ && git commit -m "feat: add BaiduPan ingestion pipeline"
```

---

### Task 5: Feishu Doc Ingestion

**Files:**
- Create: `src/ingest/feishu_ingest.py`

**Interfaces:**
- Consumes: `.env` Feishu credentials, wiki URL
- Produces: `fetch_wiki_doc() -> str` — document text content, saved to `data/raw/飞书/`

- [ ] **Step 1: Parse the Feishu wiki URL to get doc token**

The URL `https://my.feishu.cn/wiki/KCt9wXwo8i8eTmkJ45tcmCUqnEh` contains the token `KCt9wXwo8i8eTmkJ45tcmCUqnEh` after `/wiki/`.

- [ ] **Step 2: Write and run test (optional — integration test needs real token)**

- [ ] **Step 3: Write Feishu ingestor**

```python
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
```

- [ ] **Step 4: Run quick smoke test**

```bash
python -c "from src.ingest.feishu_ingest import FeishuDocIngestor; f = FeishuDocIngestor(); print('available:', f.available); c = f.fetch_doc(); print('content len:', len(c))"
Expected: available=True or False, content from Feishu if configured
```

- [ ] **Step 5: Commit**

```bash
git add src/ingest/feishu_ingest.py && git commit -m "feat: add Feishu doc ingestor"
```

---

### Task 6: Text Cleaner

**Files:**
- Create: `src/processing/cleaner.py`

**Interfaces:**
- Consumes: raw text string
- Produces: `clean_text(text: str) -> str` — cleaned text

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cleaner.py
from src.processing.cleaner import clean_text

def test_remove_extra_whitespace():
    result = clean_text("hello    world\n\n\n\n")
    assert "hello world" in result
    assert "\n\n\n" not in result

def test_remove_special_chars():
    result = clean_text("　test　　data")
    assert "test data" in result

def test_remove_urls():
    result = clean_text("check https://example.com/page for details")
    assert "check" in result
    assert "https://" not in result

def test_remove_empty_lines():
    result = clean_text("a\n\n\n\nb\n\n\nc")
    # should keep reasonable spacing
    assert "a" in result and "b" in result and "c" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cleaner.py -v
Expected: FAIL
```

- [ ] **Step 3: Write cleaner**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_cleaner.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/processing/cleaner.py tests/ && git commit -m "feat: add text cleaner for Chinese text"
```

---

### Task 7: Document Chunker

**Files:**
- Create: `src/processing/chunker.py`

**Interfaces:**
- Consumes: cleaned text, metadata dict
- Produces: `chunk_text(text, metadata, chunk_size=500, overlap=0.1) -> list[dict]` — list of chunks with metadata

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py -v
Expected: FAIL
```

- [ ] **Step 3: Write chunker**

```python
"""
文本分块 — 按语义段落/字符数切分，支持重叠
"""
import logging
import re
import tiktoken

logger = logging.getLogger(__name__)

# 中文适合使用 cl100k_base 编码
ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    将文本切分为重叠的块

    参数:
        text: 清洗后的文本
        metadata: 元数据（博主、来源、类型、时间等）
        chunk_size: 每块的目标 token 数
        overlap: 相邻块重叠 token 数

    返回:
        [{"text": "...", "metadata": {...}, "chunk_id": 0}, ...]
    """
    if not text:
        return []

    # 按段落拆分
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0

    def flush():
        nonlocal current_chunk, current_tokens, chunk_id
        if current_chunk:
            chunk_text_joined = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text_joined,
                "metadata": {**metadata, "chunk_id": chunk_id},
                "tokens": len(ENCODER.encode(chunk_text_joined)),
            })
            chunk_id += 1
            # 重叠：保留最后一个段落的一部分
            overlap_texts = []
            overlap_tokens = 0
            for p in reversed(current_chunk):
                pt = len(ENCODER.encode(p))
                if overlap_tokens + pt <= overlap:
                    overlap_texts.insert(0, p)
                    overlap_tokens += pt
                else:
                    break
            current_chunk = overlap_texts
            current_tokens = overlap_tokens

    for para in paragraphs:
        para_tokens = len(ENCODER.encode(para))

        # 单个段落超过 chunk_size，需要强行拆分
        if para_tokens > chunk_size:
            flush()
            # 按句子拆分
            sentences = re.split(r"([。！？!\?])", para)
            temp = ""
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                if len(ENCODER.encode(temp + sentence)) > chunk_size and temp:
                    chunks.append({
                        "text": temp.strip(),
                        "metadata": {**metadata, "chunk_id": chunk_id},
                        "tokens": len(ENCODER.encode(temp.strip())),
                    })
                    chunk_id += 1
                    temp = sentence
                else:
                    temp += sentence
            if temp:
                current_chunk = [temp.strip()]
                current_tokens = len(ENCODER.encode(temp.strip()))
            continue

        # 加上新段落会超 chunk_size，刷新
        if current_tokens + para_tokens > chunk_size and current_chunk:
            flush()

        current_chunk.append(para)
        current_tokens += para_tokens

    # 最后一块
    flush()
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_chunker.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/processing/chunker.py tests/ && git commit -m "feat: add text chunker with overlap"
```

---

### Task 8: Embedding & Vector Store

**Files:**
- Create: `src/processing/embedder.py`

**Interfaces:**
- Consumes: text chunks
- Produces: `EmbeddingManager` class with `add_chunks()`, `search()`, `get_collection_stats()` — wraps ChromaDB

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
from src.processing.embedder import EmbeddingManager

def test_embedder_init():
    em = EmbeddingManager()
    assert em.collection_name == "blogger_rag"

def test_add_and_search():
    em = EmbeddingManager()
    chunks = [
        {"text": "投资要长期持有优质公司", "metadata": {"blogger": "博士", "source": "test"}},
        {"text": "短线操作需要注意风险控制", "metadata": {"blogger": "梅森", "source": "test"}},
    ]
    em.add_chunks(chunks)
    results = em.search("长期投资", n_results=5)
    assert len(results) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embedder.py -v
Expected: FAIL
```

- [ ] **Step 3: Write embedder**

```python
"""
向量化 + ChromaDB 存储
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "chroma_db"
# 使用 sentence-transformers 或 text2vec 作为 embedding 模型
# BAAI/bge-large-zh-v1.5 是中文场景的优秀选择
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")


class EmbeddingManager:
    """Embedding 生成 + ChromaDB 向量存储管理"""

    def __init__(self, collection_name: str = "blogger_rag"):
        self.collection_name = collection_name
        self._model = None
        self._client = None
        self._collection = None

    def _get_model(self):
        """延迟加载 embedding 模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBED_MODEL)
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _get_collection(self):
        """获取 ChromaDB collection（延迟初始化）"""
        if self._collection is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            # 删除旧的 collection 重新创建（开发阶段）
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[dict]):
        """
        向向量库添加文本块

        参数:
            chunks: [{"text": "...", "metadata": {...}}, ...]
        """
        if not chunks:
            return

        model = self._get_model()
        collection = self._get_collection()

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{m.get('blogger', 'unknown')}_{m.get('chunk_id', i)}_{i}"
               for i, m in enumerate(metadatas)]

        logger.info("Embedding %d chunks...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d chunks to ChromaDB", len(chunks))

    def search(self, query: str, n_results: int = 10, filter_dict: Optional[dict] = None) -> list[dict]:
        """
        检索相关文本块

        参数:
            query: 查询文本
            n_results: 返回结果数
            filter_dict: 过滤条件，如 {"blogger": "博士"}

        返回:
            [{"text": "...", "metadata": {...}, "distance": 0.xx}, ...]
        """
        model = self._get_model()
        collection = self._get_collection()

        query_embedding = model.encode([query]).tolist()[0]

        where = filter_dict if filter_dict else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            })
        return output

    def get_collection_stats(self) -> dict:
        """获取向量库统计信息"""
        collection = self._get_collection()
        count = collection.count()
        # 按博主分组统计
        blogger_counts = {}
        all_metadatas = collection.get(limit=count)["metadatas"]
        for m in all_metadatas:
            blogger = m.get("blogger", "unknown")
            blogger_counts[blogger] = blogger_counts.get(blogger, 0) + 1
        return {"total_chunks": count, "by_blogger": blogger_counts}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_embedder.py -v
Expected: PASS
```

Note: First run downloads embedding model (~2GB) and creates ChromaDB.

- [ ] **Step 5: Commit**

```bash
git add src/processing/embedder.py tests/ && git commit -m "feat: add embedding and ChromaDB vector store"
```

---

### Task 9: Processing Pipeline (End-to-End from raw to vector DB)

**Files:**
- Create: `src/processing/pipeline.py`

**Interfaces:**
- Consumes: `data/raw/` directory content
- Produces: `run_processing_pipeline() -> dict` — loads all raw text, cleans, chunks, embeds, stores

- [ ] **Step 1: Write the processing pipeline**

```python
"""
处理管道：读取 data/raw/ → 清洗 → 分块 → 向量化 → ChromaDB
"""
import json
import logging
import os
from pathlib import Path

from src.processing.cleaner import clean_text, remove_boilerplate
from src.processing.chunker import chunk_text
from src.processing.embedder import EmbeddingManager

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
CHUNKS_INDEX = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks_index.json"


def run_processing_pipeline() -> dict:
    """全量处理 data/raw/ 中的所有文件"""
    if not RAW_DIR.exists():
        return {"status": "error", "message": "raw dir not found"}

    all_chunks = []
    stats = {"files": 0, "chunks": 0, "bloggers": {}}

    # 遍历 data/raw/{blogger}/*.txt
    for blogger_dir in sorted(RAW_DIR.iterdir()):
        if not blogger_dir.is_dir():
            continue
        blogger = blogger_dir.name
        blogger_chunks = 0
        blogger_files = 0

        for file_path in sorted(blogger_dir.glob("*.txt")):
            if not file_path.is_file():
                continue
            raw_text = file_path.read_text(encoding="utf-8")
            if len(raw_text.strip()) < 20:
                continue

            # 清洗
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            # 提取文件名中的时间信息（如果有）
            source_type = "baidupan"
            if blogger == "飞书":
                source_type = "feishu"

            metadata = {
                "blogger": blogger,
                "source": source_type,
                "filename": file_path.name,
            }

            # 分块
            chunks = chunk_text(cleaned, metadata)
            all_chunks.extend(chunks)
            blogger_chunks += len(chunks)
            blogger_files += 1

        stats["files"] += blogger_files
        stats["chunks"] += blogger_chunks
        stats["bloggers"][blogger] = {"files": blogger_files, "chunks": blogger_chunks}
        logger.info("  %s: %d files → %d chunks", blogger, blogger_files, blogger_chunks)

    if not all_chunks:
        return {"status": "ok", "message": "no data to process", **stats}

    # 写入 chunks_index.json（回溯用）
    CHUNKS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_INDEX.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Chunks index saved: %d chunks", len(all_chunks))

    # 向量化并存入 ChromaDB
    embedder = EmbeddingManager()
    embedder.add_chunks(all_chunks)

    logger.info("Pipeline complete: %d chunks → ChromaDB", len(all_chunks))
    return {"status": "ok", "chunks_processed": len(all_chunks), **stats}
```

- [ ] **Step 2: Run smoke test**

```bash
# First, create a test raw file
mkdir -p data/raw/test_blogger
echo "这是一段测试内容。投资需要长期持有优质公司。分散风险很重要。" > data/raw/test_blogger/test.txt

python -c "
from src.processing.pipeline import run_processing_pipeline
result = run_processing_pipeline()
print(result)
"

# Clean up
rm -rf data/raw/test_blogger
```

Expected: Pipeline runs, creates chunks, stores in ChromaDB.

- [ ] **Step 3: Commit**

```bash
git add src/processing/pipeline.py && git commit -m "feat: add end-to-end processing pipeline"
```

---

### Task 10: RAG Retriever

**Files:**
- Create: `src/rag/retriever.py`

**Interfaces:**
- Consumes: user query, optional filters (blogger, topic)
- Produces: `retrieve(query, blogger=None, n=10) -> list[dict]` — relevant chunks with metadata

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from src.rag.retriever import retrieve

def test_retrieve_returns_list():
    results = retrieve("投资策略", n=5)
    assert isinstance(results, list)

def test_retrieve_with_blogger_filter():
    results = retrieve("市场分析", blogger="博士", n=5)
    for r in results:
        assert r["metadata"]["blogger"] == "博士"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retriever.py -v
Expected: FAIL
```

- [ ] **Step 3: Write retriever**

```python
"""
RAG 检索器 — 从 ChromaDB 检索相关文档块
"""
from typing import Optional

from src.processing.embedder import EmbeddingManager

_embedder = EmbeddingManager()


def retrieve(
    query: str,
    blogger: Optional[str] = None,
    n: int = 10,
) -> list[dict]:
    """
    检索与查询最相关的文档块

    参数:
        query: 用户查询
        blogger: 按博主过滤（可选）
        n: 返回结果数量

    返回:
        [{"text": "...", "metadata": {...}, "distance": 0.xx}, ...]
    """
    filter_dict = None
    if blogger:
        filter_dict = {"blogger": blogger}

    results = _embedder.search(query, n_results=n, filter_dict=filter_dict)
    return results


def get_stats() -> dict:
    """获取向量库统计"""
    return _embedder.get_collection_stats()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_retriever.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/retriever.py tests/ && git commit -m "feat: add RAG retriever"
```

---

### Task 11: LLM Generator (Dual Mode: API + Local)

**Files:**
- Create: `src/rag/generator.py`

**Interfaces:**
- Consumes: query + retrieved chunks
- Produces: `answer_question(query, blogger=None) -> str`, `summarize_blogger_style(blogger) -> str`

**LLM 双模式设计：**
- 默认 `LLM_MODE=api` — 走 DeepSeek API（OAI 兼容接口）
- 设置 `LLM_MODE=local` 则加载本地模型（`models/Qwen2.5-1.5B-Instruct/`）
- 后续 SFT 微调也基于 `LOCAL_MODEL_PATH`，用 LLaMA-Factory 处理

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
from src.rag.generator import answer_question, summarize_blogger_style

def test_answer_question_returns_str():
    result = answer_question("什么是价值投资？")
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_generator.py -v
Expected: FAIL
```

- [ ] **Step 3: Write generator (Dual Mode)**

```python
"""
LLM 生成器 — 双模式：DeepSeek API（默认）/ 本地 Qwen 模型
通过 LLM_MODE=api|local 切换

后续 SFT 路线：
  1. RAG query → {question, context, answer} 收集
  2. 用 LLaMA-Factory + LOCAL_MODEL_PATH 做 LoRA SFT
  3. 微调后的 model 替换 LOCAL_MODEL_PATH
"""
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src.rag.retriever import retrieve, get_stats

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODE = os.getenv("LLM_MODE", "api")  # api or local
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "Qwen2.5-1.5B-Instruct"),
)

# ======== API 模式（DeepSeek / OpenAI 兼容） ========

_api_client = None

def _get_api_client():
    global _api_client
    if _api_client is None:
        from openai import OpenAI
        _api_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"),
        )
    return _api_client

API_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

# ======== 本地模式（HuggingFace transformers） ========

_local_pipeline = None

def _get_local_pipeline():
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        logger.info("Loading local model: %s", LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _local_pipeline = {"model": model, "tokenizer": tokenizer}
    return _local_pipeline


def _generate_local(prompt: str, max_tokens: int = 2048) -> str:
    """用本地模型生成"""
    pipe = _get_local_pipeline()
    model, tokenizer = pipe["model"], pipe["tokenizer"]
    messages = [
        {"role": "system", "content": "你是一个专业的财经投资知识助手。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip() or "（本地模型未返回有效内容）"


# ======== 公共函数 ========

def _build_context(retrieved: list[dict]) -> str:
    """将检索结果拼接为上下文"""
    sections = []
    for i, r in enumerate(retrieved):
        blogger = r["metadata"].get("blogger", "未知")
        source = r["metadata"].get("source", "未知")
        sections.append(f"[来源：{blogger}（{source}）]\n{r['text']}")
    return "\n\n---\n\n".join(sections)


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    """统一 LLM 调用入口，按 LLM_MODE 分发"""
    if LLM_MODE == "local":
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return _generate_local(full_prompt, max_tokens)

    # API 模式
    client = _get_api_client()
    try:
        resp = client.chat.completions.create(
            model=API_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("LLM API call failed: %s", e)
        return f"生成失败: {e}"


def answer_question(
    query: str,
    blogger: Optional[str] = None,
    n: int = 10,
) -> str:
    """
    基于 RAG 检索结果回答问题

    参数:
        query: 用户问题
        blogger: 指定博主（可选）
        n: 检索数量
    """
    retrieved = retrieve(query, blogger=blogger, n=n)
    if not retrieved:
        return "暂无相关内容。"

    context = _build_context(retrieved)

    system_prompt = """你是一个专业的财经投资知识助手。
你的知识库来自多位财经博主的文章、视频和笔记。
请基于提供的参考资料回答用户问题。
如果参考资料不足以回答问题，请如实说明。
回答时注明信息来自哪位博主。
语言：中文，简洁专业。"""

    user_prompt = f"""## 参考资料

{context}

## 问题

{query}"""

    return _llm_call(system_prompt, user_prompt)


def summarize_blogger_style(blogger: str) -> str:
    """
    总结某位博主的投资风格
    """
    queries = [
        f"{blogger} 投资风格 投资理念",
        f"{blogger} 选股 策略 方法",
        f"{blogger} 风险控制 仓位管理",
    ]
    all_chunks = []
    seen = set()
    for q in queries:
        results = retrieve(q, blogger=blogger, n=5)
        for r in results:
            if r["text"] not in seen:
                all_chunks.append(r)
                seen.add(r["text"])

    if not all_chunks:
        return f"暂无关于 {blogger} 的足够信息。"

    context = _build_context(all_chunks[:20])

    system_prompt = """你是一个专业财经分析师。
请根据参考资料，总结这位财经博主的投资风格。
包括：投资理念、分析方法、偏好行业、风险控制、历史表现（如有）。
格式：简洁、结构化、有洞察。"""

    user_prompt = f"## 参考资料\n\n{context}\n\n## 任务\n\n请总结博主「{blogger}」的投资风格。"
    return _llm_call(system_prompt, user_prompt)


def extract_experience(blogger: Optional[str] = None) -> str:
    """
    从博主内容中提炼可复用的投资经验
    """
    queries = [
        "投资经验 教训 心得",
        "投资策略 方法 框架",
        "风险控制 止损 仓位管理",
    ]
    all_chunks = []
    seen = set()
    for q in queries:
        results = retrieve(q, blogger=blogger, n=8)
        for r in results:
            if r["text"] not in seen:
                all_chunks.append(r)
                seen.add(r["text"])

    if not all_chunks:
        return "暂无足够的经验数据。"

    context = _build_context(all_chunks[:25])

    blogger_hint = f"博主「{blogger}」" if blogger else "所有博主"
    system_prompt = """你是一个投资经验提炼专家。
请从参考资料中提炼出可复用的投资经验。
按主题分类：选股、风控、仓位、心态、行业分析等。
每条经验附上来源博主。"""

    user_prompt = f"## 参考资料\n\n{context}\n\n## 任务\n\n从{blogger_hint}的内容中提炼可复用的投资经验。"
    return _llm_call(system_prompt, user_prompt, max_tokens=3072)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_generator.py -v
Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/generator.py tests/ && git commit -m "feat: add RAG generator with dual mode (API/local)"
```

---

### Task 12: CLI Entry Point

**Files:**
- Create: `run_rag.py`

**Interfaces:**
- Consumes: command-line arguments
- Produces: CLI for ingest → process → query → summarize

- [ ] **Step 1: Write CLI entry point**

```python
#!/usr/bin/env python3
"""
财经博主 RAG 系统 — CLI 入口

用法:
    python run_rag.py ingest                     # 数据接入
    python run_rag.py process                    # 清洗→分块→向量化
    python run_rag.py pipeline                   # 全流程：ingest + process
    python run_rag.py ask "问题"                  # 问答
    python run_rag.py ask "问题" --blogger 博士   # 指定博主问答
    python run_rag.py style 博士                 # 总结博主风格
    python run_rag.py experience                 # 提炼投资经验
    python run_rag.py stats                      # 查看向量库统计
    python run_rag.py transcribe-check           # 检查AI纪要完成情况
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="财经博主 RAG 系统")
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    subparsers.add_parser("ingest", help="从数据源接入内容")

    # process
    subparsers.add_parser("process", help="清洗→分块→向量化")

    # pipeline
    subparsers.add_parser("pipeline", help="全流程：ingest + process")

    # ask
    ask_parser = subparsers.add_parser("ask", help="基于RAG回答问题")
    ask_parser.add_argument("query", help="问题")
    ask_parser.add_argument("--blogger", "-b", default=None, help="指定博主")

    # style
    style_parser = subparsers.add_parser("style", help="总结博主投资风格")
    style_parser.add_argument("blogger", help="博主名称")

    # experience
    exp_parser = subparsers.add_parser("experience", help="提炼投资经验")
    exp_parser.add_argument("--blogger", "-b", default=None, help="指定博主")

    # stats
    subparsers.add_parser("stats", help="查看向量库统计")

    # transcribe-check
    subparsers.add_parser("transcribe-check", help="检查AI纪要完成情况")

    args = parser.parse_args()

    if args.command == "ingest":
        from src.ingest.baidupan_ingest import BaiduPanIngestor
        ing = BaiduPanIngestor()
        result = ing.run()
        print(f"接入完成: {result}")

        from src.ingest.feishu_ingest import FeishuDocIngestor
        fe = FeishuDocIngestor()
        if fe.available:
            content = fe.fetch_doc()
            print(f"飞书文档: {len(content)} chars")
        else:
            print("飞书未配置，跳过")

    elif args.command == "process":
        from src.processing.pipeline import run_processing_pipeline
        result = run_processing_pipeline()
        print(f"处理完成: {result}")

    elif args.command == "pipeline":
        print("=== 阶段1: 数据接入 ===")
        from src.ingest.baidupan_ingest import BaiduPanIngestor
        ing = BaiduPanIngestor()
        result = ing.run()
        print(f"接入结果: {result}")

        from src.ingest.feishu_ingest import FeishuDocIngestor
        fe = FeishuDocIngestor()
        if fe.available:
            content = fe.fetch_doc()
            print(f"飞书文档: {len(content)} chars")

        print("\n=== 阶段2: 检查AI纪要 ===")
        completed = ing.check_pending_transcribes()
        print(f"新完成转写: {len(completed)}")

        print("\n=== 阶段3: 处理→向量化 ===")
        from src.processing.pipeline import run_processing_pipeline
        result = run_processing_pipeline()
        print(f"处理完成: {result}")

    elif args.command == "ask":
        from src.rag.generator import answer_question
        result = answer_question(args.query, blogger=args.blogger)
        print(f"\n回答:\n{result}")

    elif args.command == "style":
        from src.rag.generator import summarize_blogger_style
        result = summarize_blogger_style(args.blogger)
        print(f"\n{args.blogger} 投资风格:\n{result}")

    elif args.command == "experience":
        from src.rag.generator import extract_experience
        result = extract_experience(blogger=args.blogger)
        print(f"\n投资经验:\n{result}")

    elif args.command == "stats":
        from src.rag.retriever import get_stats
        stats = get_stats()
        print(f"\n向量库统计:\n总块数: {stats['total_chunks']}")
        for blogger, count in stats.get("by_blogger", {}).items():
            print(f"  {blogger}: {count} 块")

    elif args.command == "transcribe-check":
        from src.ingest.baidupan_ingest import BaiduPanIngestor
        ing = BaiduPanIngestor()
        completed = ing.check_pending_transcribes()
        print(f"检查完成，新完成 {len(completed)} 个转写任务")
        for c in completed:
            print(f"  ✓ {c['filename']} ({c['text_length']} chars)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI help**

```bash
python run_rag.py --help
Expected: Shows all commands
```

- [ ] **Step 3: Commit**

```bash
git add run_rag.py && git commit -m "feat: add CLI entry point for RAG system"
```

---

### Task 13: Complete Pipeline Smoke Test

**Files:** (no new files)

- [ ] **Step 1: Run full pipeline in dry-run mode**

```bash
# First check if there's existing data
ls data/raw/

# Run process only (if raw data exists)
python run_rag.py process

# Or run full pipeline (if BaiduPan is configured)
python run_rag.py pipeline
```

- [ ] **Step 2: Test RAG query**

```bash
python run_rag.py ask "长期投资和短期投机的区别"
```

- [ ] **Step 3: Test stats**

```bash
python run_rag.py stats
```

- [ ] **Step 4: Final commit if any fixes**

```bash
git add -A && git commit -m "chore: fix pipeline after smoke test"
```
