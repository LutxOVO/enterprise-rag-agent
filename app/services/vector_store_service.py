import shutil
from typing import Any

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore
from app.schemas import ClearVectorStoreResponse, StatusResponse, VectorChunkInfo, VectorChunkListResponse
from app.storage.database import clear_documents, clear_messages, clear_upload_tracking, count_documents


def get_system_status() -> StatusResponse:
    """汇总系统状态：SQLite 统计文档数，Chroma 统计 chunk 数。"""
    vector_store = ChromaVectorStore()
    return StatusResponse(
        app_name=settings.app_name,
        document_count=count_documents(),
        chunk_count=vector_store.count_chunks(),
        embedding_provider=settings.embedding_provider,
        vector_store_path=str(settings.chroma_persist_dir),
        database_path=str(settings.sqlite_db_path),
    )


def list_vector_chunks(limit: int = 20, offset: int = 0) -> VectorChunkListResponse:
    """把 Chroma 原始记录转换成前端/Swagger 更容易看的结构。"""
    vector_store = ChromaVectorStore()
    total, records = vector_store.list_chunks(limit=limit, offset=offset)
    chunks = [_to_chunk_info(index, record) for index, record in enumerate(records, start=offset)]
    return VectorChunkListResponse(total=total, limit=limit, offset=offset, chunks=chunks)


def clear_knowledge_base(delete_files: bool = True) -> ClearVectorStoreResponse:
    """
    清空本地知识库。

    这里同时清理三类数据：
    1. Chroma 中的 chunk 向量；
    2. SQLite 中的文档元数据和聊天历史；
    3. 可选删除上传文件和 MinerU 解析输出。
    """
    vector_store = ChromaVectorStore()
    removed_chunks = vector_store.clear()
    removed_documents = clear_documents()
    removed_messages = clear_messages()
    clear_upload_tracking()

    if delete_files:
        _reset_local_dirs()

    return ClearVectorStoreResponse(
        cleared=True,
        removed_chunks=removed_chunks,
        removed_documents=removed_documents,
        removed_messages=removed_messages,
        deleted_files=delete_files,
        message="Vector store and related metadata were cleared.",
    )


def _to_chunk_info(index: int, record: dict[str, Any]) -> VectorChunkInfo:
    metadata = record.get("metadata", {})
    content = record.get("content", "")
    vector = record.get("vector", [])
    return VectorChunkInfo(
        id=index,
        document_id=metadata.get("document_id", ""),
        filename=metadata.get("filename", ""),
        file_type=metadata.get("file_type", ""),
        chunk_index=int(metadata.get("chunk_index", 0)),
        content=content,
        content_length=len(content),
        vector_dimension=len(vector),
    )


def _reset_local_dirs() -> None:
    """删除上传文件和 MinerU 输出目录后重新创建空目录。"""
    for path in (settings.upload_dir, settings.mineru_output_dir):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
