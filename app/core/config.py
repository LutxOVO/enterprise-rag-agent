import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理项目配置，优先从 .env 读取。"""

    app_name: str = "Enterprise Knowledge Base RAG Agent"
    data_dir: Path = Path("data")

    # PostgreSQL 是项目唯一的运行时业务数据库；没有配置时启动会给出明确错误。
    database_url: str = ""
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    database_pool_recycle: int = 1800
    database_connect_timeout: int = 5
    database_statement_timeout_ms: int = 60000
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = ""
    postgres_port: int = 5432
    # Compose 使用该字段做宿主机端口插值；应用本身仍固定监听容器内 8000。
    rag_port: int = 8000

    embedding_provider: str = "qwen"
    chat_provider: str = "deepseek"

    qwen_api_key: str = ""
    # 只为兼容旧版 .env；新配置请统一使用 QWEN_API_KEY。
    dashscope_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_embedding_dimensions: int = 1024

    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "BAAI/bge-m3"
    siliconflow_embedding_dimensions: int = 1024

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    # Agent 默认开启思考模式；兼容层只把 reasoning_content 用于后续请求，不会暴露给前端。
    deepseek_thinking_type: str = "enabled"

    # Tavily 只在工作台开关打开且知识库证据不足时使用；密钥只从 .env 读取。
    tavily_api_key: str = ""
    tavily_max_results: int = 5
    tavily_search_depth: str = "basic"

    pdf_parser: str = "mineru_api"
    mineru_api_base_url: str = "https://mineru.net/api/v4"
    mineru_api_token: str = ""
    mineru_model_version: str = "vlm"
    mineru_language: str = "ch"
    mineru_enable_formula: bool = True
    mineru_enable_table: bool = True
    mineru_is_ocr: bool = True
    mineru_poll_interval_seconds: float = 2.0
    mineru_timeout_seconds: int = 600
    # MinerU 单次请求允许处理的最大 PDF 页数；超出后由项目自动拆分。
    mineru_max_pages_per_request: int = 200
    mineru_fallback_to_pypdf: bool = False

    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    embedding_batch_size: int = 10
    knowledge_min_sources: int = 2
    knowledge_min_confidence: float = 0.45
    # 混合检索的 RRF 参数：候选越多越可能覆盖关键词命中的片段，但计算也会略增。
    hybrid_rrf_k: int = 60
    hybrid_candidate_multiplier: int = 3

    # 批量上传的边界和并发配置。默认值适合本地学习项目，避免一次请求占用过多资源。
    max_batch_files: int = 10
    max_upload_file_size_mb: int = 50
    upload_read_chunk_size: int = 1024 * 1024
    upload_concurrency: int = 2
    max_upload_retries: int = 3

    # Agent 工具循环边界，避免模型在异常工具结果下无限调用。
    agent_max_tool_calls: int = 6
    agent_max_context_chars: int = 12000
    # 写操作审批的有效期；过期后必须重新发起一轮任务。
    agent_approval_ttl_seconds: int = 1800

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_persist_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def mineru_output_dir(self) -> Path:
        return self.data_dir / "mineru_output"

    @property
    def max_upload_file_size_bytes(self) -> int:
        return self.max_upload_file_size_mb * 1024 * 1024

    @property
    def resolved_qwen_api_key(self) -> str:
        return self.qwen_api_key or self.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")

    @property
    def resolved_deepseek_api_key(self) -> str:
        return self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def resolved_siliconflow_api_key(self) -> str:
        return self.siliconflow_api_key or os.getenv("SILICONFLOW_API_KEY", "")

    @property
    def resolved_tavily_api_key(self) -> str:
        return self.tavily_api_key or os.getenv("TAVILY_API_KEY", "")

    def ensure_dirs(self) -> None:
        """启动前确保本地数据目录存在。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
