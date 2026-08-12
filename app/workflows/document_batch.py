import asyncio
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.services.document_service import DocumentService, SavedUpload
from app.storage.database import (
    set_document_fingerprint_status,
    update_upload_batch_item,
)


class PreparedUpload(TypedDict):
    item_id: str
    saved_upload: SavedUpload


class BatchUploadState(TypedDict):
    batch_id: str
    prepared_uploads: list[PreparedUpload]
    results: Annotated[list[dict[str, Any]], add]
    errors: Annotated[list[dict[str, str]], add]
    graph_path: Annotated[list[str], add]
    semaphore: asyncio.Semaphore


class UploadWorkerState(TypedDict):
    batch_id: str
    item_id: str
    saved_upload: SavedUpload
    semaphore: asyncio.Semaphore


def dispatch_upload_workers(state: BatchUploadState) -> list[Send]:
    return [
        Send(
            "upload_worker",
            {
                "batch_id": state["batch_id"],
                "item_id": prepared["item_id"],
                "saved_upload": prepared["saved_upload"],
                "semaphore": state["semaphore"],
            },
        )
        for prepared in state["prepared_uploads"]
    ]


def _error_text(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    message = str(detail if detail is not None else exc)
    return message[:500] or type(exc).__name__


async def upload_worker_node(state: UploadWorkerState) -> dict[str, Any]:
    saved_upload = state["saved_upload"]
    item_id = state["item_id"]
    stage_holder = {"stage": "processing"}
    started = asyncio.get_running_loop().time()

    def notify_stage(stage: str) -> None:
        stage_holder["stage"] = stage
        update_upload_batch_item(item_id, status=stage)

    try:
        async with state["semaphore"]:
            service = DocumentService()
            result = await asyncio.to_thread(
                service.ingest_saved_file,
                saved_upload,
                notify_stage,
            )
        await asyncio.to_thread(
            set_document_fingerprint_status,
            saved_upload.file_hash,
            "indexed",
            saved_upload.path,
        )
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        await asyncio.to_thread(
            update_upload_batch_item,
            item_id,
            document_id=saved_upload.document_id,
            status="indexed",
            chunk_count=result["chunk_count"],
            duration_ms=duration_ms,
            error_stage=None,
            error_message=None,
        )
        return {
            "results": [{**result, "status": "indexed"}],
            "errors": [],
            "graph_path": [f"upload_worker:{saved_upload.filename}"],
        }
    except Exception as exc:
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        error_message = _error_text(exc)
        error_stage = stage_holder["stage"]
        await asyncio.to_thread(
            set_document_fingerprint_status,
            saved_upload.file_hash,
            "failed",
            saved_upload.path,
        )
        await asyncio.to_thread(
            update_upload_batch_item,
            item_id,
            document_id=saved_upload.document_id,
            status="failed",
            duration_ms=duration_ms,
            error_stage=error_stage,
            error_message=error_message,
        )
        return {
            "results": [],
            "errors": [
                {
                    "item_id": item_id,
                    "filename": saved_upload.filename,
                    "error_stage": error_stage,
                    "error": error_message,
                }
            ],
            "graph_path": [f"upload_worker_failed:{saved_upload.filename}"],
        }


def summarize_uploads_node(state: BatchUploadState) -> BatchUploadState:
    graph_path = [
        "orchestrator_start",
        "dispatch_upload_workers",
        *state.get("graph_path", []),
        "summarize_uploads",
        "END",
    ]
    return {**state, "graph_path": graph_path}


def build_document_batch_upload_graph():
    graph = StateGraph(BatchUploadState)
    graph.add_node("upload_worker", upload_worker_node)
    graph.add_node("summarize_uploads", summarize_uploads_node)

    graph.set_conditional_entry_point(dispatch_upload_workers)
    graph.add_edge("upload_worker", "summarize_uploads")
    graph.add_edge("summarize_uploads", END)
    return graph.compile()
