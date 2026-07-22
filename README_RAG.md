# 财经博主 RAG 系统

基于 RAG（检索增强生成）的财经博主内容知识库系统，支持从百度网盘和飞书文档中自动拉取博主内容，构建可查询的知识库。

## 功能概述

- **数据接入**：从百度网盘、飞书文档自动拉取博主内容（PDF、Word、PPT 等）
- **文档处理**：解析多种文档格式，进行分块、清洗、预处理
- **向量存储**：使用 ChromaDB 存储文档嵌入向量，支持语义检索
- **RAG 问答**：基于检索结果 + LLM 生成回答

## 目录结构

```
.
├── src/
│   ├── ingest/       # 数据接入：百度网盘、飞书 API 客户端
│   ├── processing/   # 文档解析、分块、向量化
│   └── rag/          # 检索 + 生成管线
├── data/
│   ├── raw/          # 原始文档，按博主分类
│   │   ├── 博士/
│   │   ├── 梅森/
│   │   ├── 爽姐/
│   │   ├── 老姚/
│   │   └── 飞书/
│   └── processed/    # 处理后数据（向量库、分块缓存）
├── .env              # 环境变量（API Key 等）
├── requirements_rag.txt  # 依赖清单
└── README_RAG.md     # 本文档
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements_rag.txt

# 配置环境变量
cp .env.example .env  # 编辑填入你的 API Key

# 使用入口
python -m src.ingest.pipeline    # 启动数据接入管线
python -m src.rag.query          # 启动问答交互
```

## 依赖

- `chromadb` - 向量数据库
- `sentence-transformers` - 文本嵌入模型
- `pypdf`, `python-pptx` - 文档解析
- `openai` / `httpx` - LLM 接口
- `python-dotenv` - 环境变量管理
