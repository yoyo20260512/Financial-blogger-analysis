"""
LLM 生成器 — 双模式：DeepSeek API（默认）/ 本地 Qwen 模型
通过 LLM_MODE=api|local 切换

后续 SFT 路线：
  1. RAG query → {question, context, answer} 收集
  2. 用 LLaMA-Factory + LOCAL_MODEL_PATH 做 LoRA SFT
  3. 微调后的 model 替换 LOCAL_MODEL_PATH
"""
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src.rag.retriever import retrieve

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODE = os.getenv("LLM_MODE", "api")  # api or local
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "Qwen2.5-1.5B-Instruct",
    ),
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
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
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
