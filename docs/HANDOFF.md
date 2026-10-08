# PaperMind 开发交接文档

## 1. 项目概览

PaperMind 是一个面向科研论文的本地知识库 RAG + Agent 系统。文档在本地解析和向量化，DeepSeek `deepseek-flash` 只接收受限的检索上下文或 Agent 对话；Embedding、BM25、RRF、Reranker 和 Chroma 索引均在本地。

当前 HEAD 基于第一阶段 `e9143ee` 完成第二阶段主体和补充验收修复。不要在验收前开始第三阶段前端重构或第四阶段正式评测。

## 2. 架构和调用流程

```text
app.py / scripts
  -> config.load_settings()
  -> ResearchAssistantService
       -> LoaderRegistry -> chunking -> ChromaVectorStore
       -> HybridRetriever(BM25 + vector + RRF + optional BGE CrossEncoder)
       -> RAGEngine (prompt/context/citation/cache/fallback/log)
       -> ResearchAgent -> ToolRegistry -> 8 tools
```

关键入口：

- `src/research_assistant/service.py`：组装索引、RAG 和 Agent。
- `src/research_assistant/retrieval.py`：BM25、RRF、BGE Reranker、检索模式。
- `src/research_assistant/rag.py`：检索、上下文限制、引用、语义缓存、流式输出和降级。
- `src/research_assistant/agent.py`：ReAct 循环、记忆、并行/超时/重复调用和工具日志。
- `src/research_assistant/tools.py`：JSON Schema 工具注册、论文信息、检索、比较、摘要、关键词、时间、计算器、联网搜索。
- `src/research_assistant/llm.py`：OpenAI 兼容 DeepSeek 调用、重试和安全错误。
- `src/research_assistant/loaders.py` / `chunking.py`：文档解析和三种分块。

## 3. 本地目录和配置

```text
course_materials/       本地课程 Word 材料，不上传
models/bge-reranker-base/本地 Reranker 权重，不上传
data/uploads/           用户上传论文，不上传
data/index/             Chroma、清单、SQLite 记忆/缓存，不上传
data/logs/              RAG/工具 JSONL 日志，不上传
src/                    应用代码
tests/                  自动化测试
scripts/                验证、索引、Agent 和模型安装脚本
docs/                   阶段记录、状态、审查和交接文档
```

配置在 `config.yaml`；密钥和本机开关在 `.env`。使用 `.env.example` 创建配置，禁止把真实密钥写入代码、日志、测试或 Git。

Reranker 固定模型为 `BAAI/bge-reranker-base`。优先从 `models/bge-reranker-base` 加载；安装脚本会复用 Hugging Face 缓存，失败时尝试 `HF_ENDPOINT=https://hf-mirror.com`，再给出 ModelScope 手动安装提示：

```powershell
python scripts\download_reranker.py
```

## 4. 当前状态、Bug 和注意事项

- Reranker 已在 CPU 上离线加载并改变真实论文排序；CPU 较慢，不要在应用启动时隐式下载。
- `paper_metadata` 是 PDF 首页启发式解析，年份/DOI 可能为空；工具返回 `metadata_note`，不要把启发式结果当权威书目。
- Agent 的工具选择由 DeepSeek 决定；中文请求应使用 UTF-8 终端（PowerShell 可先 `$env:PYTHONUTF8='1'`）。不要用硬编码关键词替代模型路由。
- 文档和工具结果是不可信数据，系统 Prompt 已要求忽略其中的指令注入内容；未来新增工具必须保持同样边界。
- `data/`、`models/` 和 `course_materials/` 都被 `.gitignore` 排除。推送前检查 `git diff --cached` 和 `git ls-files`。

## 5. 推荐阅读顺序

1. `README.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/第二阶段代码审查报告.md`
4. `src/research_assistant/config.py`、`service.py`
5. `retrieval.py`、`rag.py`、`agent.py`、`tools.py`
6. `tests/test_stage1.py`、`tests/test_advanced.py`
7. `scripts/download_reranker.py`、`scripts/verify_agent_scenarios.py`

## 6. 可执行命令

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\verify_api.py
python scripts\download_reranker.py
python scripts\index_document.py "D:\papers\paper.pdf"
python -X utf8 scripts\verify_agent_scenarios.py --scenario calculator
pytest -q
python -m compileall -q src app.py scripts
python -m pip check
streamlit run app.py
```

首次接手先运行只读检查 `git status`、`git log -5`、`git remote -v`，确认没有未授权远程和未提交敏感文件，再开始修改。
