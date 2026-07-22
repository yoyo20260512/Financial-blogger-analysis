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
