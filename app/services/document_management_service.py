import asyncio
import hashlib
import shutil
import time
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore
from app.schemas import DeleteDocumentResponse, ReindexDocumentResponse
from app.services.document_service import DocumentService, SavedUpload, VECTOR_WRITE_LOCK
from app.storage.database import (
    delete_document_record,
    get_document,
    reserve_document_fingerprint,
    set_document_fingerprint_status,
    upsert_document_fingerprint,
)


def delete_document(document_id: str, delete_files: bool = True) -> DeleteDocumentResponse:
    """删除一个文档的向量、PostgreSQL 记录以及可选的本地源文件。"""
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    # 先删 Chroma，再删 PostgreSQL，避免留下“数据库说存在、向量却还在”的残留。
    with VECTOR_WRITE_LOCK:
        removed_chunks = ChromaVectorStore().delete_by_document_id(document_id)
    deleted_record = delete_document_record(document_id)
    if deleted_record is None:
        raise HTTPException(status_code=500, detail="Document metadata disappeared during deletion.")

    if delete_files:
        _delete_document_files(Path(document["file_path"]))

    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
        removed_chunks=removed_chunks,
        deleted_files=delete_files,
        message="Document, vectors, fingerprint and local files were deleted.",
    )


async def reindex_document(document_id: str) -> ReindexDocumentResponse:
    """复用原文件重新解析和入库，成功后清理同一文档的旧 chunk。"""
    document = await asyncio.to_thread(get_document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    source_path = Path(document["file_path"])
    if not source_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="The original source file is no longer available; upload it again.",
        )

    started = time.perf_counter()
    file_hash = await asyncio.to_thread(_sha256_file, source_path)
    reservation = await asyncio.to_thread(
        reserve_document_fingerprint,
        file_hash,
        document_id,
        document["filename"],
        source_path,
    )
    if not reservation["reserved"] and reservation.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="This document is already being reindexed by another task.",
        )
    if str(reservation["document_id"]) != document_id:
        raise HTTPException(
            status_code=409,
            detail=f"The current file content is already indexed as document {reservation['document_id']}.",
        )
    # 文件内容可能已经变化；把当前 hash 作为该文档新的幂等版本登记下来。
    await asyncio.to_thread(
        upsert_document_fingerprint,
        file_hash,
        document_id,
        document["filename"],
        source_path,
        status="processing",
    )

    vector_store = ChromaVectorStore()
    old_vector_ids = await asyncio.to_thread(vector_store.get_ids_by_document_id, document_id)
    saved_upload = SavedUpload(
        document_id=document_id,
        filename=document["filename"],
        file_type=document["file_type"],
        suffix=source_path.suffix.lower(),
        path=source_path,
        file_hash=file_hash,
        size_bytes=source_path.stat().st_size,
    )

    try:
        # 保留旧向量直到新内容成功写入，重建失败时原知识库仍可回答问题。
        service = DocumentService(vector_store=vector_store)
        result = await asyncio.to_thread(service.ingest_saved_file, saved_upload)
        await asyncio.to_thread(_delete_vector_ids, vector_store, old_vector_ids)
        await asyncio.to_thread(
            set_document_fingerprint_status,
            file_hash,
            "indexed",
            source_path,
        )
    except HTTPException:
        await asyncio.to_thread(
            set_document_fingerprint_status,
            file_hash,
            "failed",
            source_path,
        )
        raise
    except Exception as exc:
        await asyncio.to_thread(
            set_document_fingerprint_status,
            file_hash,
            "failed",
            source_path,
        )
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}") from exc

    return ReindexDocumentResponse(
        document_id=document_id,
        filename=result["filename"],
        status="indexed",
        chunk_count=int(result["chunk_count"]),
        duration_ms=int((time.perf_counter() - started) * 1000),
        message="The original file was parsed and indexed again.",
    )


def _delete_vector_ids(vector_store: ChromaVectorStore, vector_ids: list[str]) -> None:
    """在写锁内删除旧版本向量，避免阻塞异步事件循环。"""
    with VECTOR_WRITE_LOCK:
        vector_store.delete_by_ids(vector_ids)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(settings.upload_read_chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _delete_document_files(source_path: Path) -> None:
    """只允许删除项目数据目录里的路径，避免数据库异常值造成越界删除。"""
    _unlink_inside(source_path, settings.upload_dir)
    output_dir = settings.mineru_output_dir / source_path.stem
    _remove_tree_inside(output_dir, settings.mineru_output_dir)


def _unlink_inside(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return
    resolved_path.unlink(missing_ok=True)


def _remove_tree_inside(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return
    if resolved_path.exists():
        shutil.rmtree(resolved_path, ignore_errors=True)
