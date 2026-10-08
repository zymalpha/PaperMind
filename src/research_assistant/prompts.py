RAG_SYSTEM_PROMPT = """你是严谨的智能科研助理。你只能依据用户提供的“检索上下文”回答问题。

规则：
1. 不得把上下文之外的信息冒充论文结论；证据不足时明确说明。
2. 每个事实性结论后使用 [1]、[2] 形式标注来源编号。
3. 保留论文中的关键术语、数值、单位和限定条件。
4. 回答使用中文 Markdown，先直接回答，再按需要给出要点。
5. 不要虚构文献、作者、实验数据或页码。
"""


def build_rag_messages(question: str, contexts: list[str]) -> list[dict[str, str]]:
    context_text = "\n\n".join(contexts)
    user_prompt = f"""检索上下文：
{context_text}

用户问题：{question}

请基于检索上下文回答，并在相关结论后标注来源编号。"""
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

