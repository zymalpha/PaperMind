# PaperMind 项目状态

更新时间：2026-10-08
当前阶段：第三阶段（Web 前端与系统集成）
最近一次本地提交：本轮第三阶段补充验收提交（以 `git log -1 --oneline` 为准）。

## 阶段总览

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 第一阶段：本地 RAG 基础 | 已实现、已验证 | 多格式解析、本地 Embedding、Chroma、DeepSeek 生成与引用 |
| 第二阶段：高级 RAG + Agent | 已实现、已验证 | 三种分块、BM25/RRF、BGE Reranker、缓存、ReAct 与 8 个工具 |
| 第二阶段补充验收 | 已完成 | Reranker 本地加载、中文 Agent、代码审查和 GitHub 同步 |
| 第三阶段：Web 集成 | 已实现、基础已验证 | FastAPI、React/Vite 工作台、文档管理、会话与 SSE 接口 |
| 第四阶段：正式评测与交付材料 | 未开始 | 本轮明确不开展 |

## 第三阶段完成内容

### 已实现

- `server/main.py`：健康、设置、会话、文档上传/重索引/删除和 SSE 聊天接口。
- `server/storage.py`：SQLite 持久化 Web 会话、消息、引用和工具轨迹。
- `web/`：React + TypeScript + Vite 前端，浅色优先、深色模式、响应式侧栏和聊天工作区。
- 对话：新建/切换/重命名/删除、刷新恢复、Markdown/GFM/代码/数学公式、复制、停止请求、检索模式标识。
- 知识库：批量上传进度、真实索引状态、重索引和删除；后端限制扩展名、文件名路径和 50 MB 单文件大小。
- 来源和 Agent 过程：展示实际 RAG 引用及实际 `tool_trace`，默认折叠工具详情。
- `start_web.bat` / `stop_web.bat`：Windows 本地双服务启动和停止。
- 文档与依赖：README、交接文档、开发记录、审查报告、FastAPI 依赖和前端锁文件。

### 已验证

- Python 自动化测试：`22 passed`（包括 FastAPI 健康/会话生命周期、多轮 SSE、Agent 工具引用和持久化测试）。
- `python -m compileall -q src server app.py scripts`：通过。
- `python -m pip check`：通过（依赖安装后执行）。
- `web/npm run build`：通过，Vite production bundle 生成成功。
- `web/npm audit --omit=dev`：通过，0 vulnerabilities；KaTeX 插件依赖已通过 npm overrides 统一到 0.19.0。
- TestClient 验证 `GET /api/health`、`GET /api/settings`、会话 CRUD 和模拟 SSE 引用持久化。
- 服务配置确认：DeepSeek `deepseek-flash`、`https://api.deepseek.com`；API Key 仅在本地 `.env`。

### 未验证/限制

- 本轮没有重复消耗 DeepSeek 费用做完整真实生成回归；需要用户在本机启动服务后进行一次真实论文问答验收。
- Agent SSE 会在 Agent 同步循环完成后发送工具轨迹；RAG token 是实时发送的，客户端断开会关闭 RAG 上游流。底层 `ResearchAgent.run()` 仍是同步 API，Agent 请求目前不能真正取消。
- 当前未加入浏览器自动化（Playwright/Selenium）截图验收；已完成前端 production build 和 API 级验证。
- 本轮未完成浏览器实际交互和真实论文 + DeepSeek 流式问答端到端验收；自动化测试使用隔离的 fake engine，不产生模型费用。
- Web 会话 SQLite 与 Agent memory 分开存储，删除 Web 会话不会清理 Agent memory 历史文件中的同名 session。
- 真实本地模型加载依赖 `models/` 和 Hugging Face 缓存；缺失时服务会返回明确错误，不会伪造 Reranker 结果。

## 已知技术债务

1. 完善可取消的异步 Agent 执行器，并让工具事件在执行过程中实时推送。
2. 为文档任务增加后台队列和进度查询，避免超大 PDF 占用请求线程。
3. 增加受控文档预览接口和更细粒度的用户权限（当前仅本机 CORS）。
4. 进一步拆分前端 bundle（当前约 791 KB，KaTeX/Markdown 使产物大于 Vite 默认警告阈值）。

## 后续计划

- 第三阶段验收后：修复用户发现的 UI/API 问题，补充真实论文端到端记录。
- 第四阶段：另行设计检索准确率、引用正确率、延迟、Token 和成本评测集；当前不提前制作评测报告或 PPT。
