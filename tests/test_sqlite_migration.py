import sqlite3
from pathlib import Path

from app.storage.database import get_document, get_document_fingerprint, get_upload_batch
from scripts.migrate_sqlite_to_postgres import _read_source, map_data_path, migrate


def test_windows_data_path_maps_to_target_directory(tmp_path):
    target = tmp_path / "data"
    assert map_data_path(
        r"D:\old\project\data\uploads\demo.pdf",
        "data",
        target,
    ) == str(target / "uploads" / "demo.pdf")


def test_sqlite_migration_is_idempotent_and_preserves_ids(tmp_path):
    source_path = tmp_path / "app.db"
    target_data_dir = tmp_path / "target-data"
    (target_data_dir / "uploads").mkdir(parents=True)
    (target_data_dir / "uploads" / "demo.md").write_text("# Demo", encoding="utf-8")

    old_path = r"D:\old\project\data\uploads\demo.md"
    with sqlite3.connect(source_path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE upload_batches (
                batch_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL,
                succeeded INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                graph_path TEXT NOT NULL DEFAULT '[]',
                duration_ms INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE upload_batch_items (
                item_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                document_id TEXT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_hash TEXT,
                file_path TEXT,
                status TEXT NOT NULL,
                error_stage TEXT,
                error_message TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                duplicate_of_document_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE document_fingerprints (
                file_hash TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            ("doc-migrate", "demo.md", "md", old_path, 1, "2026-01-01T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO upload_batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "batch-migrate",
                "completed",
                1,
                1,
                0,
                0,
                '["END"]',
                10,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO upload_batch_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "item-migrate",
                "batch-migrate",
                "doc-migrate",
                "demo.md",
                "md",
                "hash-migrate",
                old_path,
                "indexed",
                None,
                None,
                1,
                10,
                0,
                None,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO document_fingerprints VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "hash-migrate",
                "doc-migrate",
                "indexed",
                "demo.md",
                old_path,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
            ),
        )

    rows = _read_source(source_path)
    first_processed, first_missing = migrate(rows, "data", target_data_dir)
    second_processed, second_missing = migrate(rows, "data", target_data_dir)

    assert first_processed == second_processed == 4
    assert first_missing == second_missing == []
    assert get_document("doc-migrate")["file_path"] == str(target_data_dir / "uploads" / "demo.md")
    assert get_upload_batch("batch-migrate")["items"][0]["status"] == "indexed"
    assert get_document_fingerprint("doc-migrate")["file_path"] == str(
        target_data_dir / "uploads" / "demo.md"
    )
