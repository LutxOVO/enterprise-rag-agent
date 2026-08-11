import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.schemas import (
    BatchUploadItemResponse,
    BatchUploadResponse,
    UploadBatchSummary,
    UploadResponse,
)
from app.services.document_service import DocumentService, SavedUpload
from app.storage.database import (
    create_upload_batch,
    create_upload_batch_item,
    get_document,
    get_upload_batch,
    get_upload_batch_item,
    list_upload_batches,
    refresh_upload_batch,
    reserve_document_fingerprint,
    set_document_fingerprint_status,
    update_upload_batch_item,
)
from app.workflows.document_batch import build_document_batch_upload_graph


def _error_text(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    message = str(detail if detail is not None else exc)
    return message[:500] or type(exc).__name__


def _item_response(item: dict[str, Any]) -> BatchUploadItemResponse:
    return BatchUploadItemResponse(
        item_id=item["item_id"],
        filename=item["filename"],
        file_type=item["file_type"],
        status=item["status"],
        document_id=item.get("document_id"),
        duplicate_of_document_id=item.get("duplicate_of_document_id"),
        chunk_count=int(item.get("chunk_count") or 0),
        duration_ms=int(item.get("duration_ms") or 0),
        retry_count=int(item.get("retry_count") or 0),
        error_stage=item.get("error_stage"),
        error=item.get("error_message"),
    )


def build_batch_response(batch_id: str) -> BatchUploadResponse:
    batch = get_upload_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Upload batch not found: {batch_id}")

    items = [_item_response(item) for item in batch["items"]]
    results = [
        UploadResponse(
            document_id=item.document_id or "",
            filename=item.filename,
            chunk_count=item.chunk_count,
            status="indexed",
        )
        for item in items
        if item.status == "indexed"
    ]
    errors = [
        {
            "item_id": item.item_id,
            "filename": item.filename,
            "error_stage": item.error_stage or "processing",
            "error": item.error or "Unknown upload error.",
        }
        for item in items
        if item.status == "failed"
    ]
    return BatchUploadResponse(
        batch_id=batch["batch_id"],
        status=batch["status"],
        total=int(batch["total"]),
        succeeded=int(batch["succeeded"]),
        failed=int(batch["failed"]),
        skipped=int(batch["skipped"]),
        duration_ms=int(batch["duration_ms"]),
        results=results,
        errors=errors,
        graph_path=batch["graph_path"],
        items=items,
    )


def list_batch_summaries(limit: int = 20) -> list[UploadBatchSummary]:
    return [UploadBatchSummary(**batch) for batch in list_upload_batches(limit)]


async def run_batch_upload(files: list[UploadFile]) -> BatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one file.")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=400,
            detail=f"A batch can contain at most {settings.max_batch_files} files.",
        )

    batch_id = uuid.uuid4().hex
    create_upload_batch(batch_id, len(files))
    document_service = DocumentService()
    prepared_uploads: list[dict[str, Any]] = []

    # 预处理阶段先把文件保存并登记指纹，再把真正可处理的文件交给 worker。
    for file in files:
        item_id = uuid.uuid4().hex
        original_name = Path(file.filename or "unknown").name or "unknown"
        suffix = Path(original_name).suffix.lower()
        file_type = suffix.lstrip(".") or "unknown"
        create_upload_batch_item(item_id, batch_id, original_name, file_type)
        saved_upload: SavedUpload | None = None
        started = asyncio.get_running_loop().time()

        try:
            update_upload_batch_item(item_id, status="saving")
            saved_upload = await document_service.save_upload_file(file, settings.upload_dir)
            update_upload_batch_item(
                item_id,
                document_id=saved_upload.document_id,
                file_hash=saved_upload.file_hash,
                file_path=str(saved_upload.path),
            )
            reservation = reserve_document_fingerprint(
                saved_upload.file_hash,
                saved_upload.document_id,
                saved_upload.filename,
                saved_upload.path,
            )

            if not reservation["reserved"]:
                existing_id = str(reservation["document_id"])
                existing = get_document(existing_id) or {}
                document_service.cleanup_saved_upload(saved_upload)
                update_upload_batch_item(
                    item_id,
                    document_id=existing_id,
                    status="duplicate",
                    duplicate_of_document_id=existing_id,
                    file_path=None,
                    chunk_count=int(existing.get("chunk_count") or 0),
                    duration_ms=int((asyncio.get_running_loop().time() - started) * 1000),
                    error_stage=None,
                    error_message=None,
                )
                continue

            # 失败指纹重试时沿用原 document_id，避免同一个内容产生多个逻辑文档。
            saved_upload.document_id = str(reservation["document_id"])
            prepared_uploads.append({"item_id": item_id, "saved_upload": saved_upload})
            update_upload_batch_item(item_id, status="queued", document_id=saved_upload.document_id)
        except Exception as exc:
            if saved_upload is not None:
                document_service.cleanup_saved_upload(saved_upload)
                set_document_fingerprint_status(saved_upload.file_hash, "failed", saved_upload.path)
            update_upload_batch_item(
                item_id,
                status="failed",
                error_stage="validation" if isinstance(exc, HTTPException) else "saving",
                error_message=_error_text(exc),
                duration_ms=int((asyncio.get_running_loop().time() - started) * 1000),
            )

    graph_path = ["orchestrator_start", "preflight"]
    if prepared_uploads:
        try:
            graph = build_document_batch_upload_graph()
            result = await graph.ainvoke(
                {
                    "batch_id": batch_id,
                    "prepared_uploads": prepared_uploads,
                    "results": [],
                    "errors": [],
                    "graph_path": [],
                    "semaphore": asyncio.Semaphore(max(1, settings.upload_concurrency)),
                }
            )
            graph_path = result.get("graph_path", graph_path)
        except Exception as exc:
            # 工作流本身异常时，只标记仍处于非终态的文件，已经成功的文件不回滚。
            graph_path = [*graph_path, f"workflow_failed:{type(exc).__name__}", "END"]
            for prepared in prepared_uploads:
                item = get_upload_batch_item(prepared["item_id"])
                if item and item["status"] not in {"indexed", "duplicate", "failed"}:
                    saved_upload = prepared["saved_upload"]
                    set_document_fingerprint_status(saved_upload.file_hash, "failed", saved_upload.path)
                    update_upload_batch_item(
                        prepared["item_id"],
                        status="failed",
                        error_stage="workflow",
                        error_message=_error_text(exc),
                    )

    refresh_upload_batch(batch_id, graph_path=[*graph_path, "END"] if graph_path[-1] != "END" else graph_path)
    return build_batch_response(batch_id)


async def retry_batch_item(batch_id: str, item_id: str) -> BatchUploadResponse:
    batch = get_upload_batch(batch_id)
    item = get_upload_batch_item(item_id)
    if batch is None or item is None or item["batch_id"] != batch_id:
        raise HTTPException(status_code=404, detail="Upload batch item not found.")
    if item["status"] != "failed":
        raise HTTPException(status_code=400, detail="Only failed upload items can be retried.")
    if int(item["retry_count"] or 0) >= settings.max_upload_retries:
        raise HTTPException(
            status_code=400,
            detail=f"Retry limit reached ({settings.max_upload_retries}).",
        )

    file_path = Path(item["file_path"] or "")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="The saved source file is no longer available.")

    file_hash = item.get("file_hash")
    if not file_hash:
        hasher = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(settings.upload_read_chunk_size), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

    document_id = item.get("document_id") or uuid.uuid4().hex
    reservation = reserve_document_fingerprint(
        file_hash,
        document_id,
        item["filename"],
        file_path,
    )
    if not reservation["reserved"]:
        raise HTTPException(status_code=409, detail="This document is already being processed by another task.")

    # 上次失败可能发生在 Chroma 已写入、PostgreSQL 状态未成功更新之后；重试前先做补偿清理。
    DocumentService().remove_document_vectors(str(reservation["document_id"]))

    retry_count = int(item["retry_count"] or 0) + 1
    update_upload_batch_item(
        item_id,
        document_id=str(reservation["document_id"]),
        file_hash=file_hash,
        status="queued",
        error_stage=None,
        error_message=None,
        retry_count=retry_count,
    )
    saved_upload = SavedUpload(
        document_id=str(reservation["document_id"]),
        filename=item["filename"],
        file_type=item["file_type"],
        suffix=Path(item["filename"]).suffix.lower(),
        path=file_path,
        file_hash=file_hash,
        size_bytes=file_path.stat().st_size,
    )

    graph = build_document_batch_upload_graph()
    try:
        result = await graph.ainvoke(
            {
                "batch_id": batch_id,
                "prepared_uploads": [{"item_id": item_id, "saved_upload": saved_upload}],
                "results": [],
                "errors": [],
                "graph_path": ["retry_start"],
                "semaphore": asyncio.Semaphore(1),
            }
        )
        graph_path = result.get("graph_path", ["retry", "END"])
    except Exception as exc:
        set_document_fingerprint_status(file_hash, "failed", file_path)
        update_upload_batch_item(
            item_id,
            status="failed",
            error_stage="workflow",
            error_message=_error_text(exc),
        )
        graph_path = ["retry_start", f"workflow_failed:{type(exc).__name__}", "END"]
    refresh_upload_batch(batch_id, graph_path=graph_path)
    return build_batch_response(batch_id)
