import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.storage.database import (
    clear_documents,
    clear_messages,
    clear_upload_tracking,
    dispose_database_engine,
    init_db,
)


def _database_name(database_url: str) -> str:
    return urlsplit(database_url.replace("postgresql+psycopg://", "postgresql://")).path.strip("/").lower()


@pytest.fixture(autouse=True)
def isolated_postgres_database():
    """所有测试共享独立 rag_test 库，并在每个测试前后清理业务表。"""
    original_url = settings.database_url
    test_url = os.getenv("TEST_DATABASE_URL", "").strip() or original_url.strip()
    if not test_url:
        pytest.fail(
            "TEST_DATABASE_URL is required for tests, for example "
            "postgresql+psycopg://rag:password@localhost:5432/rag_test"
        )
    if _database_name(test_url) != "rag_test":
        pytest.fail("Tests must use the dedicated PostgreSQL database named rag_test.")

    settings.database_url = test_url
    dispose_database_engine()
    try:
        init_db()
        clear_documents()
        clear_messages()
        clear_upload_tracking()
        yield
    finally:
        clear_documents()
        clear_messages()
        clear_upload_tracking()
        settings.database_url = original_url
        dispose_database_engine()
