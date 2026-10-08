# PaperMind

PaperMind 是南京农业大学生产实习项目：面向科研论文的可溯源 RAG + Agent 智能科研助手。论文在本地解析、分块、Embedding 和索引；DeepSeek 负责回答生成与 Agent 决策；回答保留文件名、页码和 chunk 证据。

当前已完成第一、第二阶段，并在第三阶段增加了现代化 React Web 工作台。原有 `app.py` Streamlit 入口仍然保留。

## 功能

- PDF、DOCX、TXT、Markdown 解析和增量索引
- 向量检索、BM25 + RRF 混合检索、混合检索 + BGE Reranker
- 上下文裁剪、引用溯源、语义缓存和低相关性降级
- ReAct Agent：知识库检索、元信息提取、论文对比、关键词、摘要、时间、计算器、联网搜索八个工具
- React 对话工作台：流式回答、Markdown/GFM、代码、数学公式、历史会话、复制、停止、深色主题
- 论文知识库管理：批量上传、进度、索引、重索引、删除
- 引用来源和 Agent 工具执行轨迹展示
- FastAPI 健康、设置、会话、文档和 SSE 聊天接口

## 架构

```text
web/ (React + TypeScript + Vite)
        │ REST / Server-Sent Events
server/ (FastAPI 适配层、会话 SQLite、上传安全校验)
        │
src/research_assistant/ (已验证的 RAG + Agent 核心)
  Loader → Chunker → Chroma + BM25/RRF/Reranker → RAG/Agent → DeepSeek
```

## 目录

```text
web/                    React 前端
server/                 FastAPI 接口层和 Web 会话存储
src/research_assistant/ RAG、Agent、工具和模型调用核心
tests/                  Python 单元/API 测试
scripts/                API、模型、索引和 Agent 验证脚本
docs/                   开发记录、状态、交接和审查报告
app.py                  原 Streamlit 入口（保留）
start_web.bat           Windows 一键启动
stop_web.bat            Windows 停止服务
data/                   本地上传、索引、缓存、日志（忽略）
models/                 本地 Embedding/Reranker 权重（忽略）
course_materials/       课程材料（本地保存，忽略）
```

## 安装与配置

需要 Python 3.11+、Node.js 18+ 和 npm。

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
cd web
npm install
cd ..
```

在 `.env` 中设置真实密钥（只保存在本机）：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
WEB_SEARCH_ENABLED=false
```

默认本地模型为 `BAAI/bge-small-zh-v1.5` 和 `BAAI/bge-reranker-base`。Reranker 权重固定放在 `models/bge-reranker-base/`，可运行 `python scripts/download_reranker.py` 下载；权重不会提交到 Git。

## 启动

Windows 推荐运行 `start_web.bat`：脚本会检查 Python/Node、安装前端依赖、启动 FastAPI（`http://127.0.0.1:8000`）、启动 Vite（`http://127.0.0.1:5173`）并打开浏览器。关闭时运行 `stop_web.bat`。

手动启动：

```powershell
# 终端 1
$env:PYTHONPATH="$PWD\\src"
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000

# 终端 2
cd web
npm run dev
```

保留的 Streamlit 入口：`streamlit run app.py`。

## API 速览

- `GET /api/health`、`GET/PATCH /api/settings`
- `GET/POST /api/sessions`、`GET/PATCH/DELETE /api/sessions/{id}`
- `POST /api/chat/stream`：SSE 事件 `session/token/citations/tool/final/error/done`
- `GET /api/documents`、`POST /api/documents/upload`
- `POST /api/documents/{id}/reindex`、`DELETE /api/documents/{id}`

前端不会直接访问 DeepSeek。API Key 不会返回浏览器，也不会写入日志。

## 测试

```powershell
pytest -q
python -m compileall -q src server app.py scripts
python -m pip check
cd web; npm run build
```

最近一次本地结果：Python 测试 `19 passed`，前端 Vite production build 通过。真实 DeepSeek 生成和真实论文上传需要在配置本地密钥与模型后执行，避免测试阶段产生无意义费用。

阶段状态、限制和后续计划见 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)，接手流程见 [docs/HANDOFF.md](docs/HANDOFF.md)。
