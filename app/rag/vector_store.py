import re
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
    retrieval_strategy: str = "dense"
    dense_rank: int | None = None
    bm25_rank: int | None = None
    # Hybrid 的 score 是 RRF 排名分数，不代表语义相似度；保留 Chroma 原始 cosine distance
    # 供可信度门控使用。distance 越小，表示向量语义越接近。
    dense_distance: float | None = None


@dataclass
class SearchRetriever:
    """一个足够简单的 Retriever 适配器，支持 hybrid 检索。"""

    vector_store: "ChromaVectorStore"
    k: int
    retrieval_strategy: str
    document_id: str | None = None
    filename: str | None = None

    def invoke(self, query: str) -> list[Document]:
        results = self.vector_store.search(
            query,
            k=self.k,
            retrieval_strategy=self.retrieval_strategy,
            document_id=self.document_id,
            filename=self.filename,
        )
        return [result.document for result in results]


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

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        document_id: str | None = None,
        filename: str | None = None,
    ) -> list[SearchResult]:
        """保留旧方法名，默认执行 dense similarity search。"""
        return self.search(query, k=k, retrieval_strategy="dense", document_id=document_id, filename=filename)

    def search(
        self,
        query: str,
        k: int | None = None,
        retrieval_strategy: str = "dense",
        document_id: str | None = None,
        filename: str | None = None,
    ) -> list[SearchResult]:
        """统一检索入口：支持纯向量 dense 和 BM25 + dense 的 hybrid。"""
        top_k = max(1, k or settings.top_k)
        strategy = retrieval_strategy.lower()
        metadata_filter = self._build_metadata_filter(document_id, filename)

        if strategy == "dense":
            return self._dense_search(query, top_k, metadata_filter)
        if strategy == "hybrid":
            return self._hybrid_search(query, top_k, metadata_filter)
        raise ValueError("retrieval_strategy must be 'dense' or 'hybrid'.")

    def as_retriever(
        self,
        k: int | None = None,
        retrieval_strategy: str = "dense",
        document_id: str | None = None,
        filename: str | None = None,
    ):
        """把向量库包装成 Retriever；hybrid 使用本地适配器复用同一检索入口。"""
        top_k = max(1, k or settings.top_k)
        strategy = retrieval_strategy.lower()
        if strategy == "dense":
            search_kwargs: dict[str, Any] = {"k": top_k}
            metadata_filter = self._build_metadata_filter(document_id, filename)
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
            return self.chroma.as_retriever(search_kwargs=search_kwargs)
        if strategy == "hybrid":
            return SearchRetriever(self, top_k, strategy, document_id=document_id, filename=filename)
        raise ValueError("retrieval_strategy must be 'dense' or 'hybrid'.")

    def _dense_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        # Chroma 的 cosine score 是距离，数值越小越相似；这里原样保留，便于调试。
        kwargs: dict[str, Any] = {"k": top_k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        scored_documents = self.chroma.similarity_search_with_score(query, **kwargs)
        return [
            SearchResult(
                document=document,
                score=float(score),
                retrieval_strategy="dense",
                dense_rank=rank,
                dense_distance=float(score),
            )
            for rank, (document, score) in enumerate(scored_documents, start=1)
        ]

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """用 Reciprocal Rank Fusion 合并 dense 和 BM25 的排序。"""
        candidate_k = max(top_k, top_k * settings.hybrid_candidate_multiplier)
        dense_results = self._dense_search(query, candidate_k, metadata_filter)
        all_documents = [
            document
            for document in self._get_all_documents()
            if self._matches_metadata_filter(document.metadata, metadata_filter)
        ]
        if not all_documents:
            return []

        from rank_bm25 import BM25Okapi

        tokenized_documents = [self._tokenize(document.page_content) for document in all_documents]
        bm25 = BM25Okapi(tokenized_documents)
        bm25_scores = bm25.get_scores(self._tokenize(query))
        bm25_order = sorted(
            range(len(all_documents)),
            key=lambda index: float(bm25_scores[index]),
            reverse=True,
        )[:candidate_k]

        dense_by_key = {self._document_key(result.document): result for result in dense_results}
        dense_ranks = {
            self._document_key(result.document): rank
            for rank, result in enumerate(dense_results, start=1)
        }
        bm25_by_key = {
            self._document_key(all_documents[index]): all_documents[index]
            for index in bm25_order
        }
        bm25_ranks = {
            self._document_key(all_documents[index]): rank
            for rank, index in enumerate(bm25_order, start=1)
        }

        candidate_keys = set(dense_by_key) | set(bm25_by_key)
        fused: list[SearchResult] = []
        for key in candidate_keys:
            dense_rank = dense_ranks.get(key)
            bm25_rank = bm25_ranks.get(key)
            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += 1.0 / (settings.hybrid_rrf_k + dense_rank)
            if bm25_rank is not None:
                rrf_score += 1.0 / (settings.hybrid_rrf_k + bm25_rank)

            document = dense_by_key[key].document if key in dense_by_key else bm25_by_key[key]
            fused.append(
                SearchResult(
                    document=document,
                    score=rrf_score,
                    retrieval_strategy="hybrid",
                    dense_rank=dense_rank,
                    bm25_rank=bm25_rank,
                    dense_distance=(
                        dense_by_key[key].dense_distance
                        if key in dense_by_key
                        else None
                    ),
                )
            )

        fused.sort(
            key=lambda result: (
                -result.score,
                result.dense_rank or 10**9,
                result.bm25_rank or 10**9,
            )
        )
        return fused[:top_k]

    def _get_all_documents(self) -> list[Document]:
        data = self.chroma._collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            Document(page_content=content or "", metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
        ]

    @staticmethod
    def _document_key(document: Document) -> tuple[str, str, str]:
        metadata = document.metadata
        return (
            str(metadata.get("document_id", "")),
            str(metadata.get("chunk_index", "")),
            document.page_content,
        )

    @staticmethod
    def _build_metadata_filter(
        document_id: str | None,
        filename: str | None,
    ) -> dict[str, Any] | None:
        conditions = []
        if document_id:
            conditions.append({"document_id": document_id})
        if filename:
            conditions.append({"filename": filename})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _matches_metadata_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        if "$and" in metadata_filter:
            return all(ChromaVectorStore._matches_metadata_filter(metadata, item) for item in metadata_filter["$and"])
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """无额外中文分词依赖：英文按词、中文按字和二元词组切分。"""
        tokens: list[str] = []
        for segment in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
                tokens.extend(segment)
                tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
            else:
                tokens.append(segment)
        return tokens

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

    def get_ids_by_document_id(self, document_id: str) -> list[str]:
        """返回一个文档当前的向量 ID，重建索引成功后可删除旧版本。"""
        data = self.chroma._collection.get(where={"document_id": document_id}, include=[])
        return list(data.get("ids") or [])

    def delete_by_ids(self, ids: list[str]) -> int:
        if ids:
            self.chroma._collection.delete(ids=ids)
        return len(ids)

    def delete_by_document_id(self, document_id: str) -> int:
        """按 document_id 删除向量，用于入库失败后的补偿清理。"""
        data = self.chroma._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids = data.get("ids") or []
        if ids:
            self.chroma._collection.delete(ids=ids)
        return len(ids)

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """按配置的批大小调用 Embedding API，避免单次请求过大。"""
        vectors: list[list[float]] = []
        batch_size = settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents(batch))
        return vectors

    def _get_collection_name(self) -> str:
        provider = settings.embedding_provider.lower()
        if provider == "qwen":
            return f"knowledge_base_qwen_{settings.qwen_embedding_dimensions}"
        if provider == "siliconflow":
            return f"knowledge_base_siliconflow_{settings.siliconflow_embedding_dimensions}"
        if provider == "openai":
            return "knowledge_base_openai"
        raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: '{settings.embedding_provider}'")
