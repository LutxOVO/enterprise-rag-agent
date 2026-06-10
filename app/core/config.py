import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理项目配置，优先从 .env 读取。"""

    app_name: str = "Enterprise Knowledge Base RAG Agent"
    data_dir: Path = Path("data")

    embedding_provider: str = "qwen"
    chat_provider: str = "qwen"

    qwen_api_key: str = ""
    dashscope_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_embedding_dimensions: int = 1024

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"

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
    mineru_fallback_to_pypdf: bool = False

    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    embedding_batch_size: int = 10

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
    def sqlite_db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def resolved_qwen_api_key(self) -> str:
        return self.qwen_api_key or self.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")

    @property
    def resolved_deepseek_api_key(self) -> str:
        return self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")

    def ensure_dirs(self) -> None:
        """启动前确保本地数据目录存在。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
