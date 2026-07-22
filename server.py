"""
财经博主 RAG 系统 — FastAPI Web 服务

启动: python server.py --host 0.0.0.0 --port 8000
公网分享: python server.py --host 0.0.0.0 --port 8000 --share
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.rag.generator import (
    answer_question,
    compare_bloggers,
    summarize_blogger_style,
    extract_experience,
)
from src.rag.retriever import get_stats

app = FastAPI(title="财经博主 RAG 分析系统", version="1.0.0")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── API ──────────────────────────────────────────

@app.get("/api/bloggers")
async def api_bloggers():
    """获取所有博主列表"""
    stats = get_stats()
    bloggers = list(stats.get("by_blogger", {}).keys())
    return {"bloggers": bloggers or ["博士", "梅森", "爽姐", "老姚"]}


@app.get("/api/stats")
async def api_stats():
    """获取向量库统计"""
    try:
        stats = get_stats()
        return stats
    except Exception as e:
        return {"total_chunks": 0, "by_blogger": {}, "error": str(e)}


@app.get("/api/ask")
async def api_ask(query: str = Query(...), blogger: str = Query(None)):
    """RAG 问答"""
    try:
        result = answer_question(query, blogger=blogger)
        return {"result": result}
    except Exception as e:
        return {"result": f"出错了: {e}"}


@app.get("/api/compare")
async def api_compare(query: str = Query(...), bloggers: str = Query("博士,梅森")):
    """对比博主观点"""
    try:
        blogger_list = [b.strip() for b in bloggers.split(",")]
        result = compare_bloggers(query, blogger_list)
        return {"result": result}
    except Exception as e:
        return {"result": f"出错了: {e}"}


@app.get("/api/style")
async def api_style(blogger: str = Query(...)):
    """总结博主风格"""
    try:
        result = summarize_blogger_style(blogger)
        return {"result": result}
    except Exception as e:
        return {"result": f"出错了: {e}"}


@app.get("/api/experience")
async def api_experience(blogger: str = Query(None)):
    """提炼投资经验"""
    try:
        result = extract_experience(blogger=blogger)
        return {"result": result}
    except Exception as e:
        return {"result": f"出错了: {e}"}


# ── 首页 ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>请先创建 static/index.html</h1>")


# ── 启动 ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="财经博主 RAG Web 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--share", action="store_true", help="打印 share 命令")
    args = parser.parse_args()

    if args.share:
        print(f"\n🌐 局域网访问: http://{args.host}:{args.port}")
        print("🌍 公网分享 (serveo):")
        print(f"   ssh -o ServerAliveInterval=60 -R 80:localhost:{args.port} serveo.net")
        print("🌍 公网分享 (ngrok):")
        print(f"   ngrok http {args.port}")
        print()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
