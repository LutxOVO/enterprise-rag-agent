from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.storage.database import (
    create_upload_batch,
    create_upload_batch_item,
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
