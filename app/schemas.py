from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"
    duplicate_of_document_id: str | None = None


class BatchUploadItemResponse(BaseModel):
    item_id: str
    filename: str
    file_type: str
    status: str
    document_id: str | None = None
    duplicate_of_document_id: str | None = None
    chunk_count: int = 0
    duration_ms: int = 0
    retry_count: int = 0
    error_stage: str | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    batch_id: str
    status: str
    total: int
    succeeded: int
    failed: int
    skipped: int = 0
    duration_ms: int = 0
    results: list[UploadResponse]
    errors: list[dict[str, str]]
    graph_path: list[str]
    items: list[BatchUploadItemResponse] = Field(default_factory=list)


class UploadBatchSummary(BaseModel):
    batch_id: str
    status: str
    total: int
    succeeded: int
    failed: int
    skipped: int
    duration_ms: int
    created_at: str
    completed_at: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: str


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
    removed_chunks: int
    deleted_files: bool
    message: str


class ReindexDocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    duration_ms: int
    message: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    thread_id: str = Field("default", description="会话 ID，用来区分不同对话历史")
    top_k: int | None = Field(None, ge=1, le=10, description="检索最相似的前 K 个 chunk")
    retrieval_strategy: Literal["dense", "hybrid"] = Field(
        "dense",
        description="dense 为向量检索，hybrid 为 dense + BM25 的 RRF 混合检索",
    )
    document_id: str | None = Field(None, description="只检索指定文档 ID")
    filename: str | None = Field(None, description="只检索指定文件名")


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    content_preview: str
    retrieval_strategy: Literal["dense", "hybrid"] = "dense"
    dense_rank: int | None = None
    bm25_rank: int | None = None


class AskResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[SourceChunk]
    timings_ms: dict[str, int] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    input: str = Field(..., min_length=1, description="用户输入")
    thread_id: str = Field("default", min_length=1, max_length=128, description="会话 ID")
    web_search_enabled: bool = Field(
        False,
        description="是否允许在知识库证据不足时使用 Tavily 联网搜索",
    )


class AgentRunRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=10000, description="用户任务")
    thread_id: str = Field(..., min_length=1, max_length=128, description="可恢复的 Agent 线程 ID")
    web_search_enabled: bool = Field(
        False,
        description="是否允许在知识库证据不足时使用 Tavily 联网搜索",
    )


class AgentResumeRequest(BaseModel):
    approval_id: str = Field(..., min_length=1, max_length=128)
    decision: Literal["approve", "reject"]
    reason: str | None = Field(None, max_length=1000)


class AgentStateResponse(BaseModel):
    thread_id: str
    status: str
    run_id: str | None = None
    pending_approval: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    approval_status: str = "none"
    last_answer: str = ""
    last_error: str = ""
    web_search_enabled: bool = False


class AgentThreadDeleteResponse(BaseModel):
    thread_id: str
    deleted: bool
    message: str


class AgentResponse(BaseModel):
    thread_id: str
    output: str
    tool_used: str
    extra: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    app_name: str
    document_count: int
    chunk_count: int
    embedding_provider: str
    vector_store_path: str
    database_path: str


class VectorChunkInfo(BaseModel):
    id: int
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    content: str
    content_length: int
    vector_dimension: int


class VectorChunkListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    chunks: list[VectorChunkInfo]


class ClearVectorStoreResponse(BaseModel):
    cleared: bool
    removed_chunks: int
    removed_documents: int
    removed_messages: int
    deleted_files: bool
    message: str


class RagasEvaluationRequest(BaseModel):
    dataset_path: str = Field("eval_data/ai_agents_in_depth_eval_dataset.json", description="评估数据集 JSON 路径")
    retrieval_mode: Literal["baseline", "hyde_rewrite"] = Field(
        "baseline",
        description="baseline 为原问题检索，hyde_rewrite 为 Query Rewrite + HyDE",
    )
    retrieval_strategy: Literal["dense", "hybrid"] = Field(
        "dense",
        description="评估使用的检索器：dense 或 dense + BM25 的 hybrid",
    )
    samples_output_path: str | None = Field(None, description="RAGAS 样本缓存输出路径")
    result_output_path: str | None = Field(None, description="RAGAS 评估结果输出路径")
    top_k: int = Field(3, ge=1, le=10, description="每个问题检索的 chunk 数量")
    max_samples: int | None = Field(None, ge=1, description="限制评估样本数，调试时建议先填 1")
    force_rebuild: bool = Field(False, description="是否忽略缓存，重新生成回答和检索上下文")


class RagasBuildSamplesResponse(BaseModel):
    sample_count: int
    dataset_path: str
    samples_path: str
    top_k: int
    max_samples: int | None = None
    retrieval_mode: Literal["baseline", "hyde_rewrite"]
    retrieval_strategy: Literal["dense", "hybrid"] = "dense"
    cache_used: bool
    cache_status: str
    retrieval_metrics: dict[str, float | None] = Field(default_factory=dict)


class RagasRunResponse(BaseModel):
    sample_count: int
    dataset_path: str
    samples_path: str
    result_path: str
    csv_path: str
    top_k: int
    max_samples: int | None = None
    retrieval_mode: Literal["baseline", "hyde_rewrite"]
    retrieval_strategy: Literal["dense", "hybrid"] = "dense"
    cache_used: bool
    cache_status: str
    metrics: dict[str, float]
    metric_null_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_metrics: dict[str, float | None] = Field(default_factory=dict)
    summary_path: str | None = None
