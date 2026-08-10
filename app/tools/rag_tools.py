from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore
from app.storage.database import count_documents, list_documents


class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="Question or keywords used to search the knowledge base")
    top_k: int = Field(2, ge=1, le=5)


def _preview(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def _search_knowledge_base_impl(query: str, top_k: int = 2) -> str:
    """检索知识库，并返回短片段给 Agent。"""
    results = ChromaVectorStore().similarity_search(query, k=min(top_k, 5))
    if not results:
        return "知识库为空，或没有检索到相关内容。"

    lines = ["知识库检索结果如下。若需要完整自然语言回答，请调用 /api/rag/ask。"]
    for index, result in enumerate(results, start=1):
        meta = result.document.metadata
        lines.append(
            f"[{index}] 来源: {meta.get('filename')} | chunk={meta.get('chunk_index')} | "
            f"score={result.score:.4f}\n{_preview(result.document.page_content)}"
        )
    return "\n\n".join(lines)


@tool(args_schema=KnowledgeSearchInput)
def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """检索已上传的知识库 chunk，并返回相关片段。"""
    return _search_knowledge_base_impl(query, top_k)


@tool
def query_system_status() -> str:
    """查询系统状态，例如文档数、chunk 数和 Chroma 路径。"""
    return (
        f"应用: {settings.app_name}\n"
        f"文档数: {count_documents()}\n"
        f"Chunk 数: {ChromaVectorStore().count_chunks()}\n"
        f"Embedding Provider: {settings.embedding_provider}\n"
        f"Chroma 目录: {settings.chroma_persist_dir}"
    )


@tool
def query_document_list() -> str:
    """查询当前已经上传的文档列表。"""
    documents = list_documents()
    if not documents:
        return "当前还没有上传文档。"

    return "\n".join(
        f"- {doc['filename']} ({doc['file_type']}), chunks={doc['chunk_count']}, id={doc['document_id']}"
        for doc in documents
    )
