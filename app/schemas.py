from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    thread_id: str = Field("default", description="会话 ID，用来区分不同对话历史")
    top_k: int | None = Field(None, ge=1, le=10, description="检索最相似的前 K 个 chunk")


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    content_preview: str


class AskResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[SourceChunk]


class AgentRequest(BaseModel):
    input: str = Field(..., min_length=1, description="用户输入")
    thread_id: str = Field("default", description="会话 ID")


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
    dataset_path: str = Field("eval_data/ragas_eval_dataset.json", description="评估数据集 JSON 路径")
    samples_output_path: str = Field("eval_outputs/ragas_samples.json", description="RAGAS 样本缓存输出路径")
    result_output_path: str = Field("eval_outputs/ragas_result.json", description="RAGAS 评估结果输出路径")
    top_k: int = Field(3, ge=1, le=10, description="每个问题检索的 chunk 数量")
    max_samples: int | None = Field(None, ge=1, description="限制评估样本数，调试时建议先填 1")
    force_rebuild: bool = Field(False, description="是否忽略缓存，重新生成回答和检索上下文")


class RagasBuildSamplesResponse(BaseModel):
    sample_count: int
    dataset_path: str
    samples_path: str
    top_k: int
    max_samples: int | None = None
    cache_used: bool
    cache_status: str


class RagasRunResponse(BaseModel):
    sample_count: int
    dataset_path: str
    samples_path: str
    result_path: str
    csv_path: str
    top_k: int
    max_samples: int | None = None
    cache_used: bool
    cache_status: str
    metrics: dict[str, float]
