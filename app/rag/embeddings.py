from langchain_core.embeddings import Embeddings

from app.core.config import settings


def get_embeddings() -> Embeddings:
    """根据配置选择 Embedding 服务，把文本转换为向量。"""
    provider = settings.embedding_provider.lower()
    if provider == "qwen":
        if not settings.resolved_qwen_api_key:
            raise RuntimeError("Qwen Embedding requires QWEN_API_KEY in .env.")

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.qwen_embedding_model,
            api_key=settings.resolved_qwen_api_key,
            base_url=settings.qwen_base_url,
            dimensions=settings.qwen_embedding_dimensions,
            check_embedding_ctx_length=False,
        )

    if provider == "siliconflow":
        if not settings.resolved_siliconflow_api_key:
            raise RuntimeError("SiliconFlow Embedding requires SILICONFLOW_API_KEY in .env.")

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.siliconflow_embedding_model,
            api_key=settings.resolved_siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            dimensions=settings.siliconflow_embedding_dimensions,
            check_embedding_ctx_length=False,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI Embedding requires OPENAI_API_KEY in .env.")

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER: '{settings.embedding_provider}'. "
        "Please set EMBEDDING_PROVIDER to 'qwen', 'siliconflow' or 'openai' "
        "and configure the corresponding API Key in .env."
    )
