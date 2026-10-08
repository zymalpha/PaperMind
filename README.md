# 智能科研助理

面向科研论文的本地知识库问答与 ReAct Agent。当前完成第一、二阶段：本地文档解析与向量索引、高级混合检索、可溯源 RAG 生成和多工具 Agent。

## 安装与运行

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中设置自己的 `DEEPSEEK_API_KEY`。密钥仅从环境读取；`.env`、本地索引、上传文件和运行日志均不应提交。默认使用 DeepSeek `deepseek-flash`，Base URL 为 `https://api.deepseek.com`。Embedding 使用本地 `BAAI/bge-small-zh-v1.5`；首次运行需下载模型。联网搜索默认关闭，可在 `.env` 设置 `WEB_SEARCH_ENABLED=true` 后启用。

```powershell
python scripts\verify_api.py
python scripts\index_document.py "D:\papers\paper.pdf"
streamlit run app.py
```

命令行运行 Agent（同一 session ID 可续接会话）：

```powershell
python scripts\run_agent.py "总结知识库中论文的研究方法" --session-id seminar
```

## 第二阶段能力

- 分块策略：固定长度、递归边界、句段语义分块；文档元信息包含来源文件、页码和 chunk ID。
- 检索模式：`vector`、`hybrid`、`hybrid_rerank`。混合检索由本地 BM25 和手写 RRF 融合；重排使用本地 `BAAI/bge-reranker-base`。首次启用重排需下载模型。加载失败会记录告警并退回混合检索。
- 索引管理：按内容哈希去重；同名上传文档变化时替换旧索引；支持强制重建、清单查询和删除。
- RAG：受限上下文、证据引用、SQLite 语义缓存、低相关性与模型失败降级；请求日志保存查询哈希、chunk、耗时和 Token 用量，不记录问题原文或 API Key。
- Agent：ReAct 多轮调用，JSON Schema 参数校验，最多 8 个已注册工具：知识库检索、论文元信息、论文对比、关键词提取、论文摘要、时间、计算器及可配置联网搜索。支持独立会话、近期记忆、旧轮次摘要、并行调用、超时和重复调用防护。工具执行明细写入 `data/logs/agent_tools.jsonl`，敏感字段及类似 API Key 的字符串会脱敏。

检索模式在 `config.yaml` 的 `retrieval.mode` 配置。默认启用混合检索+重排；如果本机暂时无法下载重排模型，可设为 `hybrid`。本地日志与 SQLite 缓存都位于 `data/`，不会纳入 Git。

## 目录

```text
src/research_assistant/  应用代码
tests/                   自动化测试
scripts/                 API 验证、文档索引和 Agent 命令行入口
docs/                    分阶段开发记录
data/uploads/            本地上传文件（忽略）
data/index/               Chroma、清单、缓存和 Agent 记忆（忽略）
data/logs/                请求及工具执行日志（忽略）
app.py                   现有 Streamlit 启动入口
config.yaml              非密钥配置
```

## 验证与阶段边界

运行 `pytest -q`、`python -m compileall -q src app.py scripts` 和 `python -m pip check` 验证。第一阶段和第二阶段的记录分别位于 `docs/第一阶段开发记录.md` 与 `docs/第二阶段开发记录.md`。第三阶段界面重构与第四阶段正式评测、报告及 PPT 均尚未开始。
