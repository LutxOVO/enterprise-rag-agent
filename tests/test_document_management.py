from pathlib import Path

import pytest

from app.core.config import settings
from app.services import document_management_service as management
from app.storage.database import (
    get_document,
    init_db,
    save_document,
    upsert_document_fingerprint,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted_document_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.documents: dict[str, list] = {}
        self.old_ids = ["old-vector"]

    def delete_by_document_id(self, document_id: str) -> int:
        self.deleted_document_ids.append(document_id)
        return 1

    def get_ids_by_document_id(self, document_id: str) -> list[str]:
        return list(self.old_ids)

    def delete_by_ids(self, ids: list[str]) -> int:
        self.deleted_ids.extend(ids)
        return len(ids)

    def add_documents(self, documents: list) -> None:
        document_id = documents[0].metadata["document_id"]
        self.documents[document_id] = documents


@pytest.fixture
def management_data(tmp_path, monkeypatch):
    original_data_dir = settings.data_dir
    original_pdf_parser = settings.pdf_parser
    settings.data_dir = tmp_path / "data"
    settings.pdf_parser = "pypdf"
    settings.ensure_dirs()
    init_db()
    fake_store = FakeVectorStore()
    monkeypatch.setattr(management, "ChromaVectorStore", lambda: fake_store)
    yield fake_store
    settings.data_dir = original_data_dir
    settings.pdf_parser = original_pdf_parser


def prepare_document(document_id: str = "doc-management") -> Path:
    source_path = settings.upload_dir / f"{document_id}_demo.md"
    source_path.write_text("# Demo\n\nOriginal content.", encoding="utf-8")
    save_document(document_id, "demo.md", "md", source_path, 1)
    upsert_document_fingerprint(
        "original-hash",
        document_id,
        "demo.md",
        source_path,
        status="indexed",
    )
    (settings.mineru_output_dir / source_path.stem).mkdir(parents=True)
    return source_path


def test_delete_document_removes_vectors_record_and_files(management_data):
    source_path = prepare_document()

    result = management.delete_document("doc-management")

    assert result.deleted is True
    assert result.removed_chunks == 1
    assert not source_path.exists()
    assert not (settings.mineru_output_dir / source_path.stem).exists()
    assert get_document("doc-management") is None
    assert management_data.deleted_document_ids == ["doc-management"]


@pytest.mark.anyio
async def test_reindex_keeps_document_id_and_removes_old_vectors(management_data):
    source_path = prepare_document("doc-reindex")

    result = await management.reindex_document("doc-reindex")

    assert result.status == "indexed"
    assert result.document_id == "doc-reindex"
    assert result.chunk_count >= 1
    assert management_data.deleted_ids == ["old-vector"]
    assert source_path.exists()
    assert get_document("doc-reindex")["chunk_count"] == result.chunk_count
