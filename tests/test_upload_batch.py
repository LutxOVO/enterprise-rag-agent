import io
import pytest
from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.services.document_service import DocumentService as RealDocumentService
from app.services.document_service import SavedUpload
from app.services.upload_batch_service import retry_batch_item, run_batch_upload
from app.storage.database import (
    create_upload_batch,
    create_upload_batch_item,
    get_document,
    get_upload_batch,
    init_db,
    update_upload_batch_item,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, list] = {}

    def add_documents(self, documents: list) -> None:
        document_id = documents[0].metadata["document_id"]
        self.documents[document_id] = documents

    def delete_by_document_id(self, document_id: str) -> int:
        documents = self.documents.pop(document_id, [])
        return len(documents)


def upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    original_data_dir = settings.data_dir
    original_pdf_parser = settings.pdf_parser
    original_max_files = settings.max_batch_files
    original_max_size = settings.max_upload_file_size_mb
    settings.data_dir = tmp_path / "data"
    settings.pdf_parser = "pypdf"
    settings.max_batch_files = 10
    settings.max_upload_file_size_mb = 50
    settings.ensure_dirs()
    init_db()
    fake_vector_store = FakeVectorStore()
    factory = lambda: RealDocumentService(vector_store=fake_vector_store)
    monkeypatch.setattr("app.services.upload_batch_service.DocumentService", factory)
    monkeypatch.setattr("app.workflows.document_batch.DocumentService", factory)
    yield fake_vector_store
    settings.data_dir = original_data_dir
    settings.pdf_parser = original_pdf_parser
    settings.max_batch_files = original_max_files
    settings.max_upload_file_size_mb = original_max_size


@pytest.mark.anyio
async def test_batch_success_and_duplicate(isolated_data):
    content = b"# RAG\n\nRAG uses retrieval before generation."
    result = await run_batch_upload(
        [
            upload("first.md", content),
            upload("same-content.md", content),
        ]
    )

    assert result.status == "completed"
    assert result.total == 2
    assert result.succeeded == 1
    assert result.skipped == 1
    assert [item.status for item in result.items] == ["indexed", "duplicate"]
    assert get_upload_batch(result.batch_id)["status"] == "completed"


@pytest.mark.anyio
async def test_batch_partial_success_for_invalid_file(isolated_data):
    result = await run_batch_upload(
        [
            upload("valid.md", b"# Valid\n\nThis file is valid."),
            upload("bad.exe", b"not supported"),
        ]
    )

    assert result.status == "partial_success"
    assert result.succeeded == 1
    assert result.failed == 1
    failed = next(item for item in result.items if item.status == "failed")
    assert failed.error_stage == "validation"


@pytest.mark.anyio
async def test_failed_item_can_be_retried(isolated_data, monkeypatch):
    should_fail = {"value": True}

    class FailingDocumentService(RealDocumentService):
        def ingest_saved_file(self, saved_upload: SavedUpload, stage_callback=None):
            if should_fail["value"]:
                if stage_callback:
                    stage_callback("parsing")
                raise HTTPException(status_code=500, detail="simulated parser failure")
            return super().ingest_saved_file(saved_upload, stage_callback)

    factory = lambda: FailingDocumentService(vector_store=isolated_data)
    monkeypatch.setattr("app.services.upload_batch_service.DocumentService", factory)
    monkeypatch.setattr("app.workflows.document_batch.DocumentService", factory)

    result = await run_batch_upload([upload("retry.md", b"# Retry\n\nRetry me.")])
    assert result.status == "failed"
    failed_item = result.items[0]
    assert failed_item.error_stage == "parsing"
    assert failed_item.retry_count == 0

    should_fail["value"] = False
    retried = await retry_batch_item(result.batch_id, failed_item.item_id)
    assert retried.status == "completed"
    assert retried.items[0].status == "indexed"
    assert retried.items[0].retry_count == 1


@pytest.mark.anyio
async def test_batch_file_limit(isolated_data):
    settings.max_batch_files = 1
    with pytest.raises(HTTPException) as error:
        await run_batch_upload(
            [
                upload("one.md", b"one"),
                upload("two.md", b"two"),
            ]
        )
    assert error.value.status_code == 400


@pytest.mark.anyio
async def test_file_size_limit_is_recorded(isolated_data):
    settings.max_upload_file_size_mb = 0
    result = await run_batch_upload([upload("large.md", b"content")])

    assert result.status == "failed"
    assert result.items[0].error_stage == "validation"
    assert "too large" in (result.items[0].error or "").lower()


@pytest.mark.anyio
async def test_vector_failure_is_compensated(isolated_data, monkeypatch):
    class FailingVectorStore(FakeVectorStore):
        def add_documents(self, documents: list) -> None:
            raise RuntimeError("simulated vector write failure")

    failing_store = FailingVectorStore()
    factory = lambda: RealDocumentService(vector_store=failing_store)
    monkeypatch.setattr("app.services.upload_batch_service.DocumentService", factory)
    monkeypatch.setattr("app.workflows.document_batch.DocumentService", factory)

    result = await run_batch_upload([upload("vector-failure.md", b"# Failure\n\nContent")])

    assert result.status == "failed"
    document_id = result.items[0].document_id
    assert document_id is not None
    assert get_document(document_id) is None
    assert failing_store.documents == {}


def test_interrupted_items_become_retryable_failures(isolated_data):
    create_upload_batch("batch-interrupted", 1)
    create_upload_batch_item("item-interrupted", "batch-interrupted", "lost.md", "md")
    update_upload_batch_item("item-interrupted", status="parsing")

    init_db()
    batch = get_upload_batch("batch-interrupted")
    assert batch["status"] == "failed"
    assert batch["items"][0]["status"] == "failed"
    assert batch["items"][0]["error_stage"] == "interrupted"
