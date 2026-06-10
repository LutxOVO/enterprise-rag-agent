import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.embeddings import get_embeddings


@dataclass
class SearchResult:
    document: Document
    score: float


class ChromaVectorStore:
    """Chroma 的轻量封装，项目里所有向量写入和检索都走这里。"""

    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = self._get_collection_name()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> None:
        """把 chunk 先批量 Embedding，再写入 Chroma。"""
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = self._embed_in_batches(texts)

        self.chroma._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

    def similarity_search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """使用 Chroma retriever 检索最相关的 chunk，并额外带回 score。"""
        top_k = k or settings.top_k
        retriever = self.as_retriever(k=top_k)
        documents = retriever.invoke(query)

        # retriever 返回 Document，但不带 score；这里再查一次分数，方便前端调试和面试讲解。
        scored_documents = self.chroma.similarity_search_with_score(query, k=top_k)
        score_map = {doc.page_content: score for doc, score in scored_documents}

        return [
            SearchResult(document=doc, score=float(score_map.get(doc.page_content, 0.0)))
            for doc in documents
        ]

    def as_retriever(self, k: int | None = None):
        """把向量库包装成 Retriever，供普通 RAG 和 Dynamic Prompt 共用。"""
        return self.chroma.as_retriever(search_kwargs={"k": k or settings.top_k})

    def count_chunks(self) -> int:
        data = self.chroma._collection.get(include=[])
        return len(data.get("ids") or [])

    def list_chunks(self, limit: int = 20, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
        """分页返回 Chroma 中的 chunk，给“检索调试”页面使用。"""
        total = self.count_chunks()
        data = self.chroma._collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        embeddings = data.get("embeddings")
        if embeddings is None:
            embeddings = []

        records = []
        for content, metadata, vector in zip(documents, metadatas, embeddings):
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            records.append(
                {
                    "content": content,
                    "metadata": metadata or {},
                    "vector": vector or [],
                }
            )
        return total, records

    def clear(self) -> int:
        """删除当前 collection 的所有 chunk，并返回删除数量。"""
        data = self.chroma._collection.get(include=[])
        ids = data.get("ids") or []
        if ids:
            self.chroma._collection.delete(ids=ids)
        return len(ids)

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Qwen embedding API 单批最多 10 条文本，所以这里手动分批。"""
        vectors: list[list[float]] = []
        batch_size = settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents(batch))
        return vectors

    def _get_collection_name(self) -> str:
        provider = settings.embedding_provider.lower()
        if provider in {"qwen", "dashscope"}:
            return f"knowledge_base_qwen_{settings.qwen_embedding_dimensions}"
        if provider == "openai":
            return "knowledge_base_openai"
        return "knowledge_base_local"
