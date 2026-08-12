from langchain_core.documents import Document

from app.core.config import settings
from app.rag.vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def get(self, include=None, where=None, **kwargs):
        documents = self.documents
        if where:
            documents = [
                document
                for document in documents
                if all(document.metadata.get(key) == value for key, value in where.items())
            ]
        return {
            "documents": [document.page_content for document in documents],
            "metadatas": [document.metadata for document in documents],
            "ids": [f"id-{index}" for index, _ in enumerate(documents)],
        }


class FakeChroma:
    def __init__(self, dense_results: list[tuple[Document, float]], collection: FakeCollection) -> None:
        self.dense_results = dense_results
        self._collection = collection
        self.last_filter = None

    def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        self.last_filter = filter
        results = self.dense_results
        if filter:
            results = [
                item
                for item in results
                if all(item[0].metadata.get(key) == value for key, value in filter.items())
            ]
        return results[:k]


def make_store(documents: list[Document], dense_results: list[tuple[Document, float]]):
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.chroma = FakeChroma(dense_results, FakeCollection(documents))
    return store


def test_hybrid_retrieval_fuses_dense_and_bm25(monkeypatch):
    documents = [
        Document(page_content="错误码 E20015 的参数校验失败", metadata={"document_id": "doc-1", "chunk_index": 0}),
        Document(page_content="审批流程需要部门负责人确认", metadata={"document_id": "doc-3", "chunk_index": 0}),
        Document(page_content="向量检索使用语义相似度找到相关文本", metadata={"document_id": "doc-2", "chunk_index": 0}),
    ]
    store = make_store(
        documents,
        dense_results=[(documents[2], 0.1), (documents[0], 0.2), (documents[1], 0.3)],
    )
    monkeypatch.setattr(settings, "hybrid_candidate_multiplier", 3)

    results = store.search("E20015 参数校验", k=2, retrieval_strategy="hybrid")

    assert len(results) == 2
    assert results[0].retrieval_strategy == "hybrid"
    assert results[0].document.metadata["document_id"] == "doc-1"
    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].dense_distance == 0.2
    assert results[0].score > 0


def test_hybrid_retrieval_applies_document_filter():
    documents = [
        Document(page_content="文档 A 的内容", metadata={"document_id": "doc-a", "filename": "a.md"}),
        Document(page_content="文档 B 的内容", metadata={"document_id": "doc-b", "filename": "b.md"}),
    ]
    store = make_store(documents, dense_results=[(documents[0], 0.1), (documents[1], 0.2)])

    results = store.search("文档", k=3, retrieval_strategy="hybrid", document_id="doc-b")

    assert [result.document.metadata["document_id"] for result in results] == ["doc-b"]
    assert store.chroma.last_filter == {"document_id": "doc-b"}
