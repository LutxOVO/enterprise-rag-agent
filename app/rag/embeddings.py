import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.core.config import settings


class LocalHashEmbeddings(Embeddings):
    """
    本地兜底 Embedding，用于没有 API Key 时跑通 Demo 和测试。

    真实项目流程建议使用 EMBEDDING_PROVIDER=qwen。
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def get_embeddings() -> Embeddings:
    """根据配置选择 Qwen、OpenAI 或本地 Hash Embedding。"""
    provider = settings.embedding_provider.lower()
    if provider in {"qwen", "dashscope"}:
        if not settings.resolved_qwen_api_key:
            raise RuntimeError("Qwen Embedding requires DASHSCOPE_API_KEY or QWEN_API_KEY in .env.")

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.qwen_embedding_model,
            api_key=settings.resolved_qwen_api_key,
            base_url=settings.qwen_base_url,
            dimensions=settings.qwen_embedding_dimensions,
            check_embedding_ctx_length=False,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    return LocalHashEmbeddings()
