# 智能科研助理

基于 RAG + Agent 的论文知识库问答系统。当前完成第一阶段：文档导入、分块、本地 Embedding、Chroma 持久化、Top-K 检索、DeepSeek RAG 生成与引用溯源。

## 快速开始

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，配置 `DEEPSEEK_API_KEY`。本项目的 `.env` 已被 `.gitignore` 忽略，不应提交。本地 Embedding 默认使用 `BAAI/bge-small-zh-v1.5`，首次运行会由 `sentence-transformers` 下载模型到用户缓存。

```powershell
python scripts\verify_api.py
streamlit run app.py
```

也可以先用命令行建立索引：

```powershell
python scripts\index_document.py "D:\path\to\paper.pdf"
```

## 第一阶段实现

- `loaders.py`：PDF（PyMuPDF）、DOCX（python-docx）、TXT/Markdown 加载，保留文档、页码和路径元数据。
- `chunking.py`：页码感知的递归字符分块，可配置 `chunk_size` 与 `chunk_overlap`。
- `embeddings.py`：本地 SentenceTransformer 适配器，不将文档送往远程 Embedding API。
- `vector_store.py`：Chroma PersistentClient，以文档哈希与 Chunk ID 实现可重复导入。
- `llm.py`：OpenAI 兼容 DeepSeek 客户端，支持非流式/流式、`tools/tool_choice`、重试、超时与异常降级。
- `rag.py`：检索、动态上下文、相关度阈值、引用溯源和空结果处理。

## 配置与隐私

DeepSeek 只接收用于生成回答的检索上下文；原始文档、Embedding 与 Chroma 索引在本地处理。使用云端模型时请根据你的合规要求判断是否可以传输文档片段。日志和终端输出不会打印 API Key。

## 第一阶段开发记录

1. 已读取《南京农业大学课程实践》和《项目交付模板》，按模块一与模块二的第一阶段范围实现。
2. Python 3.12.7 环境已检查，DeepSeek `deepseek-flash` 最小请求已成功返回有效模型标识。
3. 完成多格式加载、递归分块、本地 Embedding、Chroma 持久化、Top-K 检索和 DeepSeek RAG 生成。
4. Chroma 使用 `1.5.9`；旧版 `0.6.x` 在当前 Windows + Python 3.12 需要本地 MSVC 编译 `chroma-hnswlib`，因此转用包含预编译轮子的新版。
5. 仍未实现第二阶段的 BM25、RRF、Reranker 和 ReAct Agent；也未进行第四阶段的正式评测集。
