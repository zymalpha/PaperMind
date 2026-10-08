export type Citation = { index: number; source_name: string; page: number; chunk_id: string; score: number; excerpt: string };
export type ToolTrace = { step?: number; tool: string; status: string; ok?: boolean; elapsed_ms?: number; result?: unknown; error?: string };
export type ChatMessage = { role: "user" | "assistant"; content: string; citations?: Citation[]; tools?: ToolTrace[]; created_at?: number };
export type Session = { session_id: string; title: string; created_at: number; updated_at: number };
export type Document = { document_id: string; source_name: string; chunk_count: number; indexed_at: string; chunk_strategy: string; status: string };
export type Settings = { model: string; base_url: string; api_configured: boolean; retrieval_mode: "vector" | "hybrid" | "hybrid_rerank"; top_k: number; candidate_k: number; embedding_model: string; reranker_model: string; reranker_enabled: boolean; web_search_enabled: boolean };

const API = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }, ...init });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const data = await response.json(); message = data.detail?.error ?? data.detail ?? message; } catch { /* response was not JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Record<string, unknown>>("/api/health"),
  settings: () => request<Settings>("/api/settings"),
  updateSettings: (payload: Partial<Pick<Settings, "retrieval_mode" | "top_k" | "candidate_k">>) => request<Settings>("/api/settings", { method: "PATCH", body: JSON.stringify(payload) }),
  sessions: () => request<{ sessions: Session[] }>("/api/sessions"),
  createSession: (title = "新对话") => request<Session>("/api/sessions", { method: "POST", body: JSON.stringify({ title }) }),
  session: (id: string) => request<Session & { messages: ChatMessage[] }>(`/api/sessions/${encodeURIComponent(id)}`),
  renameSession: (id: string, title: string) => request<Session>(`/api/sessions/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteSession: (id: string) => request<{ deleted: boolean }>(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  documents: () => request<{ documents: Document[]; count: number }>("/api/documents"),
  upload: (files: File[], onProgress?: (value: number) => void) => {
    const body = new FormData(); files.forEach((file) => body.append("files", file));
    return new Promise<{ results: Array<{ ok: boolean; created?: boolean; document?: Document; filename?: string; error?: string }> }>((resolve, reject) => {
      const xhr = new XMLHttpRequest(); xhr.open("POST", `${API}/api/documents/upload`);
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(event.loaded / event.total); };
      xhr.onload = () => { if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText)); else reject(new Error(xhr.responseText || "上传失败")); };
      xhr.onerror = () => reject(new Error("网络中断，上传失败")); xhr.send(body);
    });
  },
  reindex: (id: string) => request<{ document: Document }>(`/api/documents/${encodeURIComponent(id)}/reindex`, { method: "POST" }),
  deleteDocument: (id: string) => request<{ deleted: boolean }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
};

export type StreamHandlers = { onSession?: (id: string) => void; onToken?: (token: string) => void; onCitations?: (citations: Citation[]) => void; onTool?: (tool: ToolTrace) => void; onFinal?: (data: { answer: string; citations: Citation[]; tools: ToolTrace[]; elapsed_ms?: number }) => void; onError?: (message: string) => void };

export async function streamChat(payload: { question: string; session_id?: string; document_id?: string; retrieval_mode?: Settings["retrieval_mode"]; use_agent?: boolean }, handlers: StreamHandlers, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API}/api/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify(payload), signal });
  if (!response.ok || !response.body) throw new Error(`聊天请求失败（${response.status}）`);
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buffer += decoder.decode(value, { stream: true }); const frames = buffer.split("\n\n"); buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const event = frame.match(/^event:\s*(.+)$/m)?.[1]?.trim(); const raw = frame.match(/^data:\s*(.+)$/m)?.[1] ?? "{}";
      let data: any; try { data = JSON.parse(raw); } catch { continue; }
      if (event === "session") handlers.onSession?.(data.session_id); else if (event === "token") handlers.onToken?.(data); else if (event === "citations") handlers.onCitations?.(data); else if (event === "tool") handlers.onTool?.(data); else if (event === "final") handlers.onFinal?.(data); else if (event === "error") handlers.onError?.(data.message ?? "模型调用失败");
    }
  }
}
