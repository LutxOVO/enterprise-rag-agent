import json

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.checkpoint import agent_checkpoint_runtime
from app.agent.graph import model_is_configured
from app.core.config import settings
from app.schemas import (
    AgentRunRequest,
    AgentRequest,
    AgentResponse,
    AgentResumeRequest,
    AgentStateResponse,
    AgentThreadDeleteResponse,
    AskRequest,
    AskResponse,
    BatchUploadResponse,
    ClearVectorStoreResponse,
    DeleteDocumentResponse,
    DocumentInfo,
    RagasBuildSamplesResponse,
    RagasEvaluationRequest,
    RagasRunResponse,
    ReindexDocumentResponse,
    StatusResponse,
    UploadResponse,
    UploadBatchSummary,
    VectorChunkListResponse,
)
from app.services.agent_service import (
    AgentServiceError,
    agent_service,
)
from app.services.document_service import DocumentService
from app.services.document_management_service import delete_document, reindex_document
from app.services.evaluation_service import (
    build_samples_file,
    default_result_path,
    default_samples_path,
    resolve_project_path,
    run_evaluation_file,
)
from app.services.rag_service import RagService
from app.services.upload_batch_service import (
    build_batch_response,
    list_batch_summaries,
    retry_batch_item,
    run_batch_upload,
)
from app.services.vector_store_service import clear_knowledge_base, get_system_status, list_vector_chunks
from app.storage.database import check_database_connection, list_documents


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    try:
        check_database_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PostgreSQL is unavailable: {exc}") from exc
    return {
        "status": "ok",
        "app": settings.app_name,
        "agent": {
            "checkpointer_ready": agent_checkpoint_runtime.ready,
            "model_configured": model_is_configured(),
            "ready": agent_service.ready,
        },
    }


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> dict:
    """上传文档，完成解析、切分、向量化和元数据保存。"""
    return await DocumentService().ingest_upload(file, settings.upload_dir)


@router.post("/documents/upload-batch", response_model=BatchUploadResponse)
async def upload_documents_batch(files: list[UploadFile] = File(...)) -> BatchUploadResponse:
    """批量上传：校验、去重、LangGraph worker、状态记录和失败隔离。"""
    return await run_batch_upload(files)


@router.get("/documents/upload-batches", response_model=list[UploadBatchSummary])
def upload_batch_history(limit: int = Query(20, ge=1, le=100)) -> list[UploadBatchSummary]:
    """返回最近批次摘要，便于前端和排查页面查看历史。"""
    return list_batch_summaries(limit)


@router.get("/documents/upload-batches/{batch_id}", response_model=BatchUploadResponse)
def upload_batch_detail(batch_id: str) -> BatchUploadResponse:
    """返回一个批次及其全部文件明细。"""
    return build_batch_response(batch_id)


@router.post(
    "/documents/upload-batches/{batch_id}/items/{item_id}/retry",
    response_model=BatchUploadResponse,
)
async def retry_upload_batch_item(batch_id: str, item_id: str) -> BatchUploadResponse:
    """只重试失败文件，成功和重复文件不会被再次处理。"""
    return await retry_batch_item(batch_id, item_id)


@router.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[dict]:
    """从 PostgreSQL 返回已上传文档列表。"""
    return list_documents()


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document_route(
    document_id: str,
    delete_files: bool = Query(True, description="同时删除原文件和 MinerU 输出"),
) -> DeleteDocumentResponse:
    """删除单个文档的向量、PostgreSQL 元数据和可选本地文件。"""
    return delete_document(document_id, delete_files=delete_files)


@router.post("/documents/{document_id}/reindex", response_model=ReindexDocumentResponse)
async def reindex_document_route(document_id: str) -> ReindexDocumentResponse:
    """复用原文件重新解析、切分和写入向量库。"""
    return await reindex_document(document_id)


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    """返回文档数、chunk 数、Embedding Provider 和本地存储路径。"""
    return get_system_status()


@router.get("/vector-store/chunks", response_model=VectorChunkListResponse)
def vector_store_chunks(
    limit: int = Query(20, ge=1, le=100, description="Maximum chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
) -> VectorChunkListResponse:
    """分页查看 Chroma 中的 chunk，便于调试切分和检索效果。"""
    return list_vector_chunks(limit=limit, offset=offset)


@router.delete("/vector-store/clear", response_model=ClearVectorStoreResponse)
def clear_vector_store(
    confirm: bool = Query(False, description="Must be true to clear the vector store"),
    delete_files: bool = Query(True, description="Also delete uploaded files and MinerU outputs"),
) -> ClearVectorStoreResponse:
    """清空 Chroma、PostgreSQL 元数据和可选的本地上传文件。"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the vector store.")

    return clear_knowledge_base(delete_files=delete_files)


@router.post("/rag/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    answer, sources, timings_ms = await RagService().ask_with_details(
        request.question,
        request.thread_id,
        request.top_k,
        request.retrieval_strategy,
        request.document_id,
        request.filename,
    )
    return AskResponse(
        thread_id=request.thread_id,
        answer=answer,
        sources=sources,
        timings_ms=timings_ms,
    )


@router.post("/rag/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    async def event_generator():
        async for chunk in RagService().ask_stream(
            request.question,
            request.thread_id,
            request.top_k,
            request.retrieval_strategy,
            request.document_id,
            request.filename,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _raise_agent_http(exc: AgentServiceError) -> None:
    detail: dict[str, object] = {"message": exc.detail, **exc.payload}
    raise HTTPException(status_code=exc.status_code, detail=detail) from exc


def _encode_sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _agent_sse(events):
    try:
        async for event in events:
            yield _encode_sse(event)
    except AgentServiceError as exc:
        yield _encode_sse({"event": "error", "message": exc.detail, **exc.payload})
        yield _encode_sse({"event": "done", "status": "failed"})


@router.post("/agent/runs/stream")
async def agent_run_stream(request: AgentRunRequest) -> StreamingResponse:
    try:
        reserved_lock = await agent_service.reserve_new_run(request.thread_id)
    except AgentServiceError as exc:
        _raise_agent_http(exc)
    return StreamingResponse(
        _agent_sse(
            agent_service.run_stream(
                request.input,
                request.thread_id,
                web_search_enabled=request.web_search_enabled,
                reserved_lock=reserved_lock,
            )
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/threads/{thread_id}/resume/stream")
async def agent_resume_stream(thread_id: str, request: AgentResumeRequest) -> StreamingResponse:
    try:
        reserved_lock = await agent_service.reserve_resume(
            thread_id,
            request.approval_id,
            request.decision,
        )
    except AgentServiceError as exc:
        _raise_agent_http(exc)
    return StreamingResponse(
        _agent_sse(
            agent_service.resume_stream(
                thread_id,
                request.approval_id,
                request.decision,
                request.reason,
                reserved_lock=reserved_lock,
            )
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/threads/{thread_id}/state", response_model=AgentStateResponse)
async def agent_thread_state(thread_id: str) -> AgentStateResponse:
    try:
        state = await agent_service.get_state(thread_id)
    except AgentServiceError as exc:
        _raise_agent_http(exc)
    return AgentStateResponse(**state)


@router.delete("/agent/threads/{thread_id}", response_model=AgentThreadDeleteResponse)
async def delete_agent_thread(thread_id: str) -> AgentThreadDeleteResponse:
    try:
        result = await agent_service.delete_thread(thread_id)
    except AgentServiceError as exc:
        _raise_agent_http(exc)
    return AgentThreadDeleteResponse(**result)


@router.post("/agent/invoke", response_model=AgentResponse)
async def agent_invoke(request: AgentRequest) -> AgentResponse:
    try:
        result = await agent_service.invoke(
            request.input,
            request.thread_id,
            web_search_enabled=request.web_search_enabled,
        )
    except AgentServiceError as exc:
        _raise_agent_http(exc)
    return AgentResponse(
        thread_id=request.thread_id,
        output=result["output"],
        tool_used=result["tool_used"],
        extra={
            "route": result["route"],
            "route_reason": result.get("route_reason", ""),
            "graph_path": result.get("graph_path", []),
            "status": result.get("status", "completed"),
            "tool_trace": result.get("tool_trace", []),
            "sources": result.get("sources", []),
        },
    )


@router.post("/agent/dynamic-rag", response_model=AgentResponse)
def agent_dynamic_rag(request: AgentRequest) -> AgentResponse:
    """Answer with a LangGraph Dynamic RAG workflow: rewrite -> HyDE -> retrieve -> generate."""
    from app.services.agent_service import AgentService

    result = AgentService().invoke_dynamic_rag(request.input, request.thread_id)
    return AgentResponse(
        thread_id=request.thread_id,
        output=result["output"],
        tool_used=result["tool_used"],
        extra={
            "route": result["route"],
            "used_rag_tool": result.get("used_rag_tool"),
            "rag_tool_name": result.get("rag_tool_name"),
            "graph_path": result.get("graph_path"),
            "original_query": result.get("original_query"),
            "rewritten_query": result.get("rewritten_query"),
            "hyde_answer": result.get("hyde_answer"),
            "retrieval_query": result.get("retrieval_query"),
            "context_sufficient": result.get("context_sufficient"),
            "context_evaluation_reason": result.get("context_evaluation_reason"),
            "top_k": result.get("top_k"),
            "retrieved_chunks": result.get("retrieved_chunks"),
        },
    )


@router.post("/evaluation/ragas/build-samples", response_model=RagasBuildSamplesResponse)
async def ragas_build_samples(request: RagasEvaluationRequest) -> RagasBuildSamplesResponse:
    """Build RAGAS evaluation samples without calling the DeepSeek judge model."""
    try:
        result = await build_samples_file(
            dataset_path=resolve_project_path(request.dataset_path, None),
            samples_path=resolve_project_path(
                request.samples_output_path,
                default_samples_path(request.retrieval_mode, request.retrieval_strategy),
            ),
            top_k=request.top_k,
            max_samples=request.max_samples,
            force_rebuild=request.force_rebuild,
            retrieval_mode=request.retrieval_mode,
            retrieval_strategy=request.retrieval_strategy,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return RagasBuildSamplesResponse(**result)


@router.post("/evaluation/ragas/run", response_model=RagasRunResponse)
async def ragas_run(request: RagasEvaluationRequest) -> RagasRunResponse:
    """
    Run a simple synchronous RAGAS evaluation.

    This endpoint may be slow because it generates RAG answers and calls DeepSeek
    several times as the judge model.
    """
    try:
        result = await run_evaluation_file(
            dataset_path=resolve_project_path(request.dataset_path, None),
            samples_path=resolve_project_path(
                request.samples_output_path,
                default_samples_path(request.retrieval_mode, request.retrieval_strategy),
            ),
            output_path=resolve_project_path(
                request.result_output_path,
                default_result_path(request.retrieval_mode, request.retrieval_strategy),
            ),
            top_k=request.top_k,
            max_samples=request.max_samples,
            force_rebuild=request.force_rebuild,
            retrieval_mode=request.retrieval_mode,
            retrieval_strategy=request.retrieval_strategy,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return RagasRunResponse(**result)
