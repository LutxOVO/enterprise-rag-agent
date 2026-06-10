from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings


SYSTEM_PROMPT = """你是企业知识库 RAG Agent。
请优先依据【知识库上下文】回答问题。
如果上下文不足，请明确说明“当前知识库中没有足够信息”，不要编造事实。
回答要简洁、准确，并尽量给出可执行建议。"""


def build_prompt(question: str, context: str, history_text: str) -> str:
    """把历史对话、检索上下文和用户问题拼成最终 Prompt。"""
    return f"""【对话历史】
{history_text or "暂无"}

【知识库上下文】
{context or "未检索到相关内容"}

【用户问题】
{question}

请基于以上信息回答。"""


def _build_chat_model(streaming: bool = False):
    """根据配置创建 OpenAI-compatible Chat 模型；无可用 Key 时返回 None。"""
    provider = settings.chat_provider.lower()
    if provider in {"qwen", "dashscope"}:
        if not settings.resolved_qwen_api_key:
            raise RuntimeError("Qwen LLM requires DASHSCOPE_API_KEY or QWEN_API_KEY in .env.")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.resolved_qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.2,
            streaming=streaming,
        )

    if provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.2,
            streaming=streaming,
        )

    return None


async def generate_answer(question: str, context: str, history_text: str) -> str:
    """非流式回答：优先调用大模型，未配置模型时返回本地 Demo 回答。"""
    prompt = build_prompt(question, context, history_text)
    llm = _build_chat_model(streaming=False)
    if llm:
        response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        return str(response.content)

    if context:
        return (
            "【Demo 本地回答】我已从知识库检索到相关内容。"
            f"根据片段信息，问题“{question}”可以参考以下内容：\n\n{context[:900]}"
        )
    return "【Demo 本地回答】当前知识库中没有足够信息，请先上传相关 txt、md 或 pdf 文档。"


async def stream_answer(question: str, context: str, history_text: str) -> AsyncGenerator[str, None]:
    """流式回答：逐段 yield 文本，FastAPI 会包装成 SSE 返回给前端。"""
    prompt = build_prompt(question, context, history_text)
    llm = _build_chat_model(streaming=True)
    if llm:
        async for chunk in llm.astream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]):
            if chunk.content:
                yield str(chunk.content)
        return

    answer = await generate_answer(question, context, history_text)
    for char in answer:
        yield char
