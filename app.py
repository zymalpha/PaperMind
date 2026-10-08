from __future__ import annotations

import sys
import time
from html import escape
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.llm import LLMError  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


st.set_page_config(page_title="智能科研助理", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; color: #17212b; }
    [data-testid="stHeader"] { background: rgba(247,248,250,.92); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3e7eb; }
    .block-container { max-width: 1280px; padding-top: 1.6rem; }
    h1, h2, h3 { letter-spacing: 0; color: #17212b; }
    .status-strip { padding: .7rem 0; border-top: 1px solid #e3e7eb; border-bottom: 1px solid #e3e7eb; }
    .source { border-left: 3px solid #138a72; padding: .2rem .8rem; margin: .5rem 0; }
    .muted { color: #65717d; font-size: .86rem; }
    .stButton button { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_service() -> ResearchAssistantService:
    return ResearchAssistantService(load_settings())


def render_sources(citations) -> None:
    if not citations:
        return
    with st.expander(f"引用来源 · {len(citations)}", expanded=True):
        for citation in citations:
            st.markdown(
                f"<div class='source'><strong>[{citation.index}] {escape(citation.source_name)}</strong> "
                f"<span class='muted'>第 {citation.page} 页 · 相关度 {citation.score:.2f}</span><br>"
                f"{escape(citation.excerpt)}</div>",
                unsafe_allow_html=True,
            )


settings = load_settings()
st.title("智能科研助理")
st.caption("基于私有文献检索的可溯源论文问答")

try:
    service = get_service()
except Exception as exc:
    st.error(f"系统初始化失败：{exc}")
    st.info("首次使用本地 BGE 模型需要联网下载。请确认依赖已安装后刷新页面。")
    st.stop()

documents = service.list_documents()
with st.container():
    status_columns = st.columns(4)
    status_columns[0].metric("已导入文档", len(documents))
    status_columns[1].metric("索引 Chunk", service.vector_store.count)
    status_columns[2].metric("生成模型", settings.llm_model)
    status_columns[3].metric("向量库", "Chroma · 已连接")

with st.sidebar:
    st.header("文档知识库")
    uploads = st.file_uploader(
        "导入论文或资料",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("开始解析并建立索引", type="primary", use_container_width=True, disabled=not uploads):
        progress = st.progress(0, text="准备处理文档")
        for index, upload in enumerate(uploads or [], start=1):
            safe_name = Path(upload.name).name
            target = settings.upload_dir / safe_name
            if target.exists():
                target = settings.upload_dir / f"{target.stem}_{int(time.time())}{target.suffix}"
            target.write_bytes(upload.getbuffer())
            progress.progress((index - 1) / len(uploads), text=f"正在索引 {safe_name}")
            try:
                document, created = service.index_file(target, copy_to_uploads=False)
                if created:
                    st.success(f"{document.source_name}：{document.chunk_count} 个 Chunk")
                else:
                    st.info(f"{safe_name}：内容已存在，未重复索引")
            except Exception as exc:
                st.error(f"{safe_name}：{exc}")
        progress.progress(1.0, text="处理完成")
        st.cache_resource.clear()
        st.rerun()

    st.subheader("已索引文档")
    if not documents:
        st.caption("暂无文档")
    for document in documents:
        st.markdown(f"**{document.source_name}**")
        st.caption(f"{document.chunk_count} 个 Chunk · ID {document.document_id[:8]}")

    st.divider()
    st.caption(f"Embedding: {settings.embedding_model}")
    st.caption("文档在本地解析与向量化；检索上下文会发送给 DeepSeek 用于生成回答。")

if "messages" not in st.session_state:
    st.session_state.messages = []

left, right = st.columns([3, 1], gap="large")
with left:
    st.subheader("论文问答")
    if not st.session_state.messages:
        st.info("导入文档后，可询问研究方法、实验设置、结论与局限性。")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                render_sources(message["citations"])

    question = st.chat_input("请输入与已导入文献相关的问题", disabled=service.vector_store.count == 0)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                stream, citations = service.create_rag_engine().stream_answer(question)
                answer = st.write_stream(stream)
                render_sources(citations)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )
            except (LLMError, ValueError, RuntimeError) as exc:
                message = f"无法完成回答：{exc}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})

with right:
    st.subheader("检索设置")
    st.text_input("Top-K", value=str(settings.top_k), disabled=True)
    st.text_input("Chunk Size", value=str(settings.chunk_size), disabled=True)
    st.text_input("Chunk Overlap", value=str(settings.chunk_overlap), disabled=True)
    st.caption("第一阶段参数由 config.yaml 管理。")
    if st.button("清空当前对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
