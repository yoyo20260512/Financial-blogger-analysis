# 财经博主 RAG 系统设计文档

## 概述

构建一个财经博主投资知识 RAG 系统，从百度网盘（视频/PPT/PDF/文本）和飞书文档采集博主内容，构建向量知识库，支持博主风格总结、观点检索和经验提炼。后期可用积累的数据做 SFT 微调。

## 数据源

### 百度网盘
- `笔记梳理/{博主名}/` — 按博主分类存储的视频、PPT、PDF、TXT/MD
- `a投资/爽姐/` — 爽姐已有内容
- `a投资/2026老姚前瞻班直播【持续更新至26年底】/` — 老姚持续更新内容
- 百度会员 AI纪要 API 做视频转文字
- PyMuPDF/pdfplumber 解析 PDF，python-pptx 解析 PPT

### 飞书
- 指定 Wiki 文档：`https://my.feishu.cn/wiki/KCt9wXwo8i8eTmkJ45tcmCUqnEh`

### IMA 知识库（暂不接入，预留）

## 架构

```
数据采集 → 文本清洗 → 分块 → Embedding → 向量存储 → RAG 检索 → LLM 生成
         ↘ 元数据标注 ↗
```

## 存储

- `data/raw/{博主}/{类型}/` — 原始提取文本
- `data/processed/` — 分块后索引
- ChromaDB 本地向量数据库

## 数据流程

1. 百度网盘：扫描目录 → 视频提交 AI纪要 / PDF 文字提取 / PPT 文字提取 / TXT 直接读 → 存 data/raw/
2. 飞书：读取指定 Wiki 文档内容 → 存 data/raw/飞书/
3. 统一处理：清洗 → 分块（~500 tokens，重叠 10-20%）→ 标注元数据（博主/来源/类型/时间）→ Embedding → ChromaDB
4. RAG 检索：按博主/主题/股票检索 → LLM 总结风格 / 回答

## 后续扩展

- SFT 训练数据生成（从 RAG 检索结果构建 instruction-output 对）
- DeepSeek API：RAG 阶段默认使用
- Qwen2.5-1.5B-Instruct：本地推理 + 后续 LLaMA-Factory LoRA SFT 微调
- 微调后的模型替换本地模型路径继续使用
