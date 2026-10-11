import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { Bot, Check, ChevronDown, ChevronRight, Copy, FileText, RefreshCw, Sparkles } from "lucide-react";
import type { ChatMessage, Citation, ToolTrace } from "./api";

export default function MessageBubble({ message, notify, onRetry }: {
  message: ChatMessage;
  notify: (message: string) => void;
  onRetry: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      notify("已复制回答");
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      notify("复制失败，请检查浏览器权限");
    }
  };
  return <article className={`message ${message.role}`}>
    <div className="message-avatar">{message.role === "user" ? "你" : <Sparkles size={15} />}</div>
    <div className="message-body">
      <div className="message-meta"><span>{message.role === "user" ? "你" : "PaperMind"}</span>
        {message.role === "assistant" && message.content && <>
          <button className="copy-button" onClick={() => void copy()}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? "已复制" : "复制"}</button>
          <button className="copy-button" onClick={onRetry}><RefreshCw size={13} />重新生成</button>
        </>}
      </div>
      {message.role === "assistant" && !message.content && !message.error
        ? <div className="typing"><i /><i /><i /></div>
        : message.content ? <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex, rehypeHighlight]}>{message.content}</ReactMarkdown></div> : null}
      {message.error && <div className="message-error" role="alert">{message.error}</div>}
      {message.tools?.length ? <ToolTracePanel tools={message.tools} /> : null}
      {message.citations?.length ? <Sources citations={message.citations} /> : null}
    </div>
  </article>;
}

function Sources({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  return <div className="sources"><button className="sources-heading" onClick={() => setOpen(!open)}>
    <span><FileText size={15} />来源依据 <em>{citations.length}</em></span>{open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
  </button>{open && <div className="source-list">{citations.map((citation) => <details className="source-item" key={citation.chunk_id}>
    <summary><span className="source-index">[{citation.index}]</span><span className="source-name">{citation.source_name}</span><span className="source-page">第 {citation.page} 页</span></summary>
    <p>{citation.excerpt}</p><small>相关度 {citation.score.toFixed(2)} · chunk {citation.chunk_id.slice(0, 10)}</small>
  </details>)}</div>}</div>;
}

function ToolTracePanel({ tools }: { tools: ToolTrace[] }) {
  const [open, setOpen] = useState(false);
  const failed = tools.some((tool) => !tool.ok);
  return <div className="tool-trace"><button className="tool-trace-heading" onClick={() => setOpen(!open)}>
    <span><Bot size={15} />工具执行 <em>{tools.length}</em></span><span className={`trace-status ${failed ? "trace-failed" : ""}`}>{failed ? "含失败步骤" : "已完成"} {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}</span>
  </button>{open && <div className="trace-list">{tools.map((tool, index) => {
    const summary = tool.error ?? (tool.result === undefined ? "" : JSON.stringify(tool.result));
    return <div className="trace-row" key={`${tool.tool}-${index}`}><span className={`trace-number ${tool.ok ? "success" : "failure"}`}>{tool.ok ? <Check size={13} /> : "!"}</span>
      <div><strong>{tool.tool}</strong><small>{tool.status} · {tool.elapsed_ms ?? 0} ms</small>{summary && <p>{summary.length > 220 ? `${summary.slice(0, 220)}…` : summary}</p>}</div>
    </div>;
  })}</div>}</div>;
}
