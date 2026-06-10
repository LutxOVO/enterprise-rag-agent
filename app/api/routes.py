from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    ClearVectorStoreResponse,
    DocumentInfo,
    RagasBuildSamplesResponse,
    RagasEvaluationRequest,
    RagasRunResponse,
    StatusResponse,
    UploadResponse,
    VectorChunkListResponse,
)
from app.services.agent_service import AgentService
from app.services.document_service import DocumentService
from app.services.evaluation_service import build_samples_file, resolve_project_path, run_evaluation_file
from app.services.rag_service import RagService
from app.services.vector_store_service import clear_knowledge_base, get_system_status, list_vector_chunks
from app.storage.database import list_documents


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> dict:
    """上传文档，完成解析、切分、向量化和元数据保存。"""
    return await DocumentService().ingest_upload(file, settings.upload_dir)


@router.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[dict]:
    """从 SQLite 返回已上传文档列表。"""
    return list_documents()


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
    """清空 Chroma、SQLite 元数据和可选的本地上传文件。"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the vector store.")

    return clear_knowledge_base(delete_files=delete_files)


@router.post("/rag/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    answer, sources = await RagService().ask(request.question, request.thread_id, request.top_k)
    return AskResponse(thread_id=request.thread_id, answer=answer, sources=sources)


@router.post("/rag/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    async def event_generator():
        async for chunk in RagService().ask_stream(request.question, request.thread_id, request.top_k):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/agent/invoke", response_model=AgentResponse)
def agent_invoke(request: AgentRequest) -> AgentResponse:
    result = AgentService().invoke(request.input, request.thread_id)
    return AgentResponse(
        thread_id=request.thread_id,
        output=result["output"],
        tool_used=result["tool_used"],
        extra={"route": result["route"]},
    )


@router.post("/agent/dynamic-rag", response_model=AgentResponse)
def agent_dynamic_rag(request: AgentRequest) -> AgentResponse:
    """Answer with a LangChain dynamic_prompt middleware that injects retrieved context."""
    result = AgentService().invoke_dynamic_rag(request.input, request.thread_id)
    return AgentResponse(
        thread_id=request.thread_id,
        output=result["output"],
        tool_used=result["tool_used"],
        extra={
            "route": result["route"],
            "original_query": result.get("original_query"),
            "rewritten_query": result.get("rewritten_query"),
            "hyde_answer": result.get("hyde_answer"),
            "retrieval_query": result.get("retrieval_query"),
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
            samples_path=resolve_project_path(request.samples_output_path, None),
            top_k=request.top_k,
            max_samples=request.max_samples,
            force_rebuild=request.force_rebuild,
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
            samples_path=resolve_project_path(request.samples_output_path, None),
            output_path=resolve_project_path(request.result_output_path, None),
            top_k=request.top_k,
            max_samples=request.max_samples,
            force_rebuild=request.force_rebuild,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return RagasRunResponse(**result)
