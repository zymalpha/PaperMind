# PaperMind 开发交接文档

## 项目定位

PaperMind 是本地论文知识库 RAG + DeepSeek Agent 系统。`src/research_assistant/` 是核心能力，`server/` 只做 HTTP 适配和 Web 持久化，`web/` 是 React 客户端；不要为了改 UI 重写核心检索或 Agent。

## 调用关系

```text
web/src/api.ts
  ├─ REST: sessions / documents / settings / health
  └─ SSE: POST /api/chat/stream
       ↓
server/main.py (懒加载 ResearchAssistantService)
       ├─ RAGEngine.stream_answer → citations + tokens
       └─ ResearchAgent.run       → answer + tool_trace
       ↓
src/research_assistant/service.py
  LoaderRegistry → Chunker → ChromaVectorStore
  HybridRetriever(BM25/RRF/BGE) → RAGEngine / ResearchAgent
```

关键入口：

- `src/research_assistant/config.py`：从 `config.yaml` 和 `.env` 加载配置。
- `src/research_assistant/service.py`：组装向量库、Embedding、Reranker、RAG 和 Agent。
- `src/research_assistant/retrieval.py`：三种检索模式和 Reranker 降级。
- `src/research_assistant/rag.py`：上下文、引用、缓存、流式生成和降级。
- `src/research_assistant/agent.py`：ReAct 循环、SQLite 记忆和工具日志。
- `server/main.py`：FastAPI 接口；服务对象是懒加载的，启动健康检查不会触发模型下载。
- `web/src/App.tsx`：页面状态、侧栏、聊天、知识库和设置界面。

## 本地目录与配置

- `.env`：真实 `DEEPSEEK_API_KEY`，只在本机，禁止提交。
- `models/bge-reranker-base/`：BGE CrossEncoder 权重，禁止提交。
- `data/uploads/`：上传论文；`data/index/`：Chroma、清单、缓存、SQLite；`data/logs/`：JSONL 日志。均被 `.gitignore` 排除。
- `course_materials/`：课程材料，本地保存，不上传。
- 前端端口 5173，FastAPI 端口 8000；可用 `PAPERMIND_API_PORT` 改 API 端口（批处理脚本默认 8000）。

## 当前状态与注意事项

- 第一、二阶段功能已验证，第三阶段 API 和前端基础功能已实现。
- API Key 不由任何接口返回；日志只写哈希、chunk、耗时、工具摘要。
- RAG SSE token 实时发送；Agent 目前是同步循环，工具轨迹在循环完成后推送，这是已知限制。
- `PATCH /api/settings` 只允许 `retrieval_mode`、`top_k`、`candidate_k`，会在下次请求懒加载服务时生效。
- 上传只接受 `.pdf/.docx/.txt/.md/.markdown`，单文件 50 MB；服务端使用 `Path.name` 防止路径穿越。
- `ResearchAgent` 的 memory SQLite 与 Web conversation SQLite 是两套存储，修改其中一套时要注意一致性。

## 推荐阅读顺序

1. `README.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/第三阶段开发记录.md`
4. `server/main.py`、`server/storage.py`
5. `web/src/api.ts`、`web/src/App.tsx`、`web/src/styles.css`
6. `src/research_assistant/service.py`、`retrieval.py`、`rag.py`、`agent.py`
7. `tests/test_web_api.py`、`tests/test_advanced.py`

## 启动与测试命令

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
cd web; npm install; npm run dev
```

Windows 可运行根目录 `start_web.bat`，停止运行 `stop_web.bat`。

```powershell
pytest -q
python -m compileall -q src server app.py scripts
python -m pip check
cd web; npm run build
```

如需验证 DeepSeek API：`python scripts/verify_api.py`。如需准备 Reranker：`python scripts/download_reranker.py`。先运行 `git status --short`，确认 `.env`、`data/`、`models/`、`course_materials/` 不在提交中。
