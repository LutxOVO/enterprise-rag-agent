from collections.abc import AsyncGenerator

from app.core.config import settings
from app.rag.llm import generate_answer, stream_answer
from app.rag.vector_store import ChromaVectorStore, SearchResult
from app.schemas import SourceChunk
from app.storage.database import get_messages, save_message


class RagService:
    """普通 2-Step RAG：先检索，再把上下文拼进 Prompt 调用模型。"""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store or ChromaVectorStore()

    async def ask(self, question: str, thread_id: str, top_k: int | None = None) -> tuple[str, list[SourceChunk]]:
        results = self.retrieve(question, top_k)
        context = self.format_context(results)
        history_text = self._format_history(thread_id)
        answer = await generate_answer(question, context, history_text)

        self._save_turn(thread_id, question, answer)
        return answer, self._to_sources(results)

    async def ask_stream(self, question: str, thread_id: str, top_k: int | None = None) -> AsyncGenerator[str, None]:
        results = self.retrieve(question, top_k)
        context = self.format_context(results)
        history_text = self._format_history(thread_id)
        collected: list[str] = []

        async for chunk in stream_answer(question, context, history_text):
            collected.append(chunk)
            yield chunk

        self._save_turn(thread_id, question, "".join(collected))

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """统一的知识库检索入口，默认使用配置里的 top_k。"""
        return self.vector_store.similarity_search(question, k=top_k or settings.top_k)

    def format_context(self, results: list[SearchResult]) -> str:
        """把检索结果整理成大模型容易阅读的上下文文本。"""
        lines = []
        for index, result in enumerate(results, start=1):
            meta = result.document.metadata
            lines.append(
                f"[{index}] 来源: {meta.get('filename')} / chunk {meta.get('chunk_index')}\n"
                f"{result.document.page_content}"
            )
        return "\n\n".join(lines)

    def _format_history(self, thread_id: str) -> str:
        """读取最近对话历史，作为简单 memory 拼进 Prompt。"""
        messages = get_messages(thread_id, limit=8)
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    def _to_sources(self, results: list[SearchResult]) -> list[SourceChunk]:
        """把内部检索结果转换成接口响应里的 sources。"""
        sources: list[SourceChunk] = []
        for result in results:
            meta = result.document.metadata
            sources.append(
                SourceChunk(
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=round(result.score, 4),
                    content_preview=result.document.page_content[:180],
                )
            )
        return sources

    def _save_turn(self, thread_id: str, question: str, answer: str) -> None:
        """保存一轮用户问题和模型回答。"""
        save_message(thread_id, "user", question)
        save_message(thread_id, "assistant", answer)
