from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.storage.database import init_db


def test_health_has_request_id_and_documents_contract(tmp_path):
    original_data_dir = settings.data_dir
    settings.data_dir = tmp_path / "data"
    try:
        with TestClient(app) as client:
            health = client.get("/api/health", headers={"X-Request-ID": "contract-test"})
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.headers["X-Request-ID"] == "contract-test"

            documents = client.get("/api/documents")
            assert documents.status_code == 200
            assert isinstance(documents.json(), list)
    finally:
        settings.data_dir = original_data_dir


def test_upload_contract_keeps_single_file_fields(monkeypatch, tmp_path):
    original_data_dir = settings.data_dir
    settings.data_dir = tmp_path / "data"
    settings.ensure_dirs()
    init_db()

    class FakeDocumentService:
        async def ingest_upload(self, file, upload_dir):
            return {
                "document_id": "contract-doc",
                "filename": file.filename,
                "chunk_count": 2,
                "status": "indexed",
            }

    import app.api.routes as routes

    monkeypatch.setattr(routes, "DocumentService", FakeDocumentService)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("contract.md", b"# Contract", "text/markdown")},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["document_id"] == "contract-doc"
        assert payload["filename"] == "contract.md"
        assert payload["chunk_count"] == 2
        assert payload["status"] == "indexed"
    finally:
        settings.data_dir = original_data_dir


def test_rag_request_rejects_unknown_retrieval_strategy(tmp_path):
    original_data_dir = settings.data_dir
    settings.data_dir = tmp_path / "data"
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/rag/ask",
                json={"question": "测试", "retrieval_strategy": "unknown"},
            )
        assert response.status_code == 422
    finally:
        settings.data_dir = original_data_dir
