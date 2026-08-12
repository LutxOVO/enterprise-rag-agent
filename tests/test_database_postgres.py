from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import text

from app.storage.database import (
    create_upload_batch,
    create_upload_batch_item,
    get_database_engine,
    get_upload_batch,
    reserve_document_fingerprint,
    save_document,
    save_message,
    get_messages,
    update_upload_batch_item,
    refresh_upload_batch,
)


def test_postgres_round_trip_keeps_existing_storage_contract(tmp_path):
    source_path = tmp_path / "demo.md"
    source_path.write_text("# PostgreSQL", encoding="utf-8")

    save_document("doc-postgres", "demo.md", "md", source_path, 2)
    save_message("thread-postgres", "user", "问题")
    save_message("thread-postgres", "assistant", "回答")

    create_upload_batch("batch-postgres", 1)
    create_upload_batch_item("item-postgres", "batch-postgres", "demo.md", "md")
    update_upload_batch_item(
        "item-postgres",
        document_id="doc-postgres",
        status="indexed",
        chunk_count=2,
    )
    batch = refresh_upload_batch("batch-postgres")

    assert batch["status"] == "completed"
    assert batch["succeeded"] == 1
    assert batch["items"][0]["document_id"] == "doc-postgres"
    assert [message["role"] for message in get_messages("thread-postgres")] == ["user", "assistant"]
    assert get_upload_batch("batch-postgres")["status"] == "completed"


def test_fingerprint_reservation_is_atomic_under_concurrency(tmp_path):
    source_path = tmp_path / "same.md"
    source_path.write_text("same content", encoding="utf-8")

    def reserve(index: int):
        return reserve_document_fingerprint(
            "same-sha256",
            f"doc-{index}",
            "same.md",
            Path(source_path),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, [1, 2]))

    assert sum(1 for result in results if result["reserved"]) == 1
    assert sum(1 for result in results if not result["reserved"]) == 1
    loser = next(result for result in results if not result["reserved"])
    assert loser["status"] == "processing"


def test_schema_migration_creates_typed_timestamps_json_and_indexes():
    with get_database_engine().connect() as connection:
        columns = connection.execute(
            text(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND ((table_name = 'documents' AND column_name IN ('created_at', 'updated_at'))
                    OR (table_name = 'upload_batches' AND column_name = 'graph_path')
                    OR (table_name = 'upload_batch_items' AND column_name = 'item_order'))
                ORDER BY table_name, column_name
                """
            )
        ).all()
        indexes = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public'"
                )
            ).all()
        }
        version = connection.execute(
            text("SELECT MAX(version) FROM schema_migrations")
        ).scalar_one()

    assert {(row[0], row[1], row[2]) for row in columns} == {
        ("documents", "created_at", "timestamp with time zone"),
        ("documents", "updated_at", "timestamp with time zone"),
        ("upload_batch_items", "item_order", "integer"),
        ("upload_batches", "graph_path", "jsonb"),
    }
    assert version == 3
    assert {
        "idx_documents_created_at",
        "idx_messages_thread_id_id",
        "idx_upload_batch_items_batch_status",
        "idx_document_fingerprints_document_id",
    }.issubset(indexes)
