"""把旧版 SQLite 元数据幂等迁移到 PostgreSQL。

脚本只迁移业务元数据，不重新生成 Chroma 向量；document_id 保持不变，
因此同一个 data/ 目录中的 Chroma 记录仍然可以和迁移后的文档记录关联。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.storage.database import get_database_engine, init_db


TABLE_NAMES = (
    "documents",
    "messages",
    "upload_batches",
    "upload_batch_items",
    "document_fingerprints",
)


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _read_table(connection: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    if not _table_exists(connection, table_name):
        return []
    rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
    return [dict(row) for row in rows]


def _value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if value is None and default is not None else value


def map_data_path(raw_path: str | None, source_data_dir: str | Path, target_data_dir: str | Path) -> str | None:
    """把旧 Windows/Linux data 路径映射到迁移目标目录。"""
    if raw_path is None or not str(raw_path).strip():
        return None

    raw_text = str(raw_path)
    normalized = raw_text.replace("\\", "/")
    lower_normalized = normalized.lower()
    source_text = str(source_data_dir).replace("\\", "/").rstrip("/")
    source_lower = source_text.lower()
    target = Path(target_data_dir)

    markers = [f"{source_lower}/"]
    source_name = Path(source_text).name.lower()
    if source_name:
        markers.append(f"/{source_name}/")
        markers.append(f"{source_name}/")

    relative_path: str | None = None
    for marker in markers:
        index = lower_normalized.rfind(marker)
        if index >= 0:
            relative_path = normalized[index + len(marker) :]
            break

    if relative_path is None:
        return raw_text

    return str(target.joinpath(*relative_path.split("/")))


def _path_and_missing(
    raw_path: str | None,
    source_data_dir: str | Path,
    target_data_dir: str | Path,
    missing_paths: list[str],
) -> str | None:
    mapped = map_data_path(raw_path, source_data_dir, target_data_dir)
    if mapped and not Path(mapped).is_file():
        missing_paths.append(mapped)
    return mapped


def _read_source(sqlite_path: Path) -> dict[str, list[dict[str, Any]]]:
    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        return {table_name: _read_table(connection, table_name) for table_name in TABLE_NAMES}


def _print_dry_run(rows_by_table: dict[str, list[dict[str, Any]]], missing_paths: list[str]) -> None:
    print(f"SQLite source: {sum(len(rows) for rows in rows_by_table.values())} rows")
    for table_name in TABLE_NAMES:
        print(f"  {table_name}: {len(rows_by_table[table_name])}")
    print(f"Mapped missing files: {len(missing_paths)}")
    for path in missing_paths[:20]:
        print(f"  missing: {path}")
    if len(missing_paths) > 20:
        print(f"  ... and {len(missing_paths) - 20} more")


def migrate(
    rows_by_table: dict[str, list[dict[str, Any]]],
    source_data_dir: str | Path,
    target_data_dir: str | Path,
) -> tuple[int, list[str]]:
    """将源表幂等 upsert 到 PostgreSQL，返回处理行数和缺失文件列表。"""
    missing_paths: list[str] = []
    documents = rows_by_table["documents"]
    messages = rows_by_table["messages"]
    batches = rows_by_table["upload_batches"]
    items = rows_by_table["upload_batch_items"]
    fingerprints = rows_by_table["document_fingerprints"]

    init_db()
    engine = get_database_engine()
    with engine.begin() as connection:
        for row in documents:
            connection.execute(
                text(
                    """
                    INSERT INTO documents
                    (document_id, filename, file_type, file_path, chunk_count, created_at)
                    VALUES (:document_id, :filename, :file_type, :file_path, :chunk_count, :created_at)
                    ON CONFLICT (document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        file_type = EXCLUDED.file_type,
                        file_path = EXCLUDED.file_path,
                        chunk_count = EXCLUDED.chunk_count,
                        created_at = EXCLUDED.created_at
                    """
                ),
                {
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_path": _path_and_missing(
                        row.get("file_path"), source_data_dir, target_data_dir, missing_paths
                    ),
                    "chunk_count": int(row.get("chunk_count") or 0),
                    "created_at": row.get("created_at") or "",
                },
            )

        for row in messages:
            connection.execute(
                text(
                    """
                    INSERT INTO messages (id, thread_id, role, content, created_at)
                    VALUES (:id, :thread_id, :role, :content, :created_at)
                    ON CONFLICT (id) DO UPDATE SET
                        thread_id = EXCLUDED.thread_id,
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        created_at = EXCLUDED.created_at
                    """
                ),
                {
                    "id": int(row["id"]),
                    "thread_id": row["thread_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row.get("created_at") or "",
                },
            )

        for row in batches:
            connection.execute(
                text(
                    """
                    INSERT INTO upload_batches
                    (batch_id, status, total, succeeded, failed, skipped, graph_path,
                     duration_ms, created_at, completed_at)
                    VALUES (:batch_id, :status, :total, :succeeded, :failed, :skipped,
                            :graph_path, :duration_ms, :created_at, :completed_at)
                    ON CONFLICT (batch_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        total = EXCLUDED.total,
                        succeeded = EXCLUDED.succeeded,
                        failed = EXCLUDED.failed,
                        skipped = EXCLUDED.skipped,
                        graph_path = EXCLUDED.graph_path,
                        duration_ms = EXCLUDED.duration_ms,
                        created_at = EXCLUDED.created_at,
                        completed_at = EXCLUDED.completed_at
                    """
                ),
                {
                    "batch_id": row["batch_id"],
                    "status": row["status"],
                    "total": int(row.get("total") or 0),
                    "succeeded": int(row.get("succeeded") or 0),
                    "failed": int(row.get("failed") or 0),
                    "skipped": int(row.get("skipped") or 0),
                    "graph_path": row.get("graph_path") or "[]",
                    "duration_ms": int(row.get("duration_ms") or 0),
                    "created_at": row.get("created_at") or "",
                    "completed_at": row.get("completed_at"),
                },
            )

        for row in items:
            connection.execute(
                text(
                    """
                    INSERT INTO upload_batch_items
                    (item_id, batch_id, document_id, filename, file_type, file_hash, file_path,
                     status, error_stage, error_message, chunk_count, duration_ms, retry_count,
                     duplicate_of_document_id, created_at, updated_at)
                    VALUES (:item_id, :batch_id, :document_id, :filename, :file_type, :file_hash,
                            :file_path, :status, :error_stage, :error_message, :chunk_count,
                            :duration_ms, :retry_count, :duplicate_of_document_id,
                            :created_at, :updated_at)
                    ON CONFLICT (item_id) DO UPDATE SET
                        batch_id = EXCLUDED.batch_id,
                        document_id = EXCLUDED.document_id,
                        filename = EXCLUDED.filename,
                        file_type = EXCLUDED.file_type,
                        file_hash = EXCLUDED.file_hash,
                        file_path = EXCLUDED.file_path,
                        status = EXCLUDED.status,
                        error_stage = EXCLUDED.error_stage,
                        error_message = EXCLUDED.error_message,
                        chunk_count = EXCLUDED.chunk_count,
                        duration_ms = EXCLUDED.duration_ms,
                        retry_count = EXCLUDED.retry_count,
                        duplicate_of_document_id = EXCLUDED.duplicate_of_document_id,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "item_id": row["item_id"],
                    "batch_id": row["batch_id"],
                    "document_id": row.get("document_id"),
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_hash": row.get("file_hash"),
                    "file_path": _path_and_missing(
                        row.get("file_path"), source_data_dir, target_data_dir, missing_paths
                    ),
                    "status": row["status"],
                    "error_stage": row.get("error_stage"),
                    "error_message": row.get("error_message"),
                    "chunk_count": int(row.get("chunk_count") or 0),
                    "duration_ms": int(row.get("duration_ms") or 0),
                    "retry_count": int(row.get("retry_count") or 0),
                    "duplicate_of_document_id": row.get("duplicate_of_document_id"),
                    "created_at": row.get("created_at") or "",
                    "updated_at": row.get("updated_at") or row.get("created_at") or "",
                },
            )

        for row in fingerprints:
            connection.execute(
                text(
                    """
                    INSERT INTO document_fingerprints
                    (file_hash, document_id, status, filename, file_path, created_at, updated_at)
                    VALUES (:file_hash, :document_id, :status, :filename, :file_path,
                            :created_at, :updated_at)
                    ON CONFLICT (file_hash) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        status = EXCLUDED.status,
                        filename = EXCLUDED.filename,
                        file_path = EXCLUDED.file_path,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "file_hash": row["file_hash"],
                    "document_id": row["document_id"],
                    "status": row["status"],
                    "filename": row["filename"],
                    "file_path": _path_and_missing(
                        row.get("file_path"), source_data_dir, target_data_dir, missing_paths
                    ),
                    "created_at": row.get("created_at") or "",
                    "updated_at": row.get("updated_at") or row.get("created_at") or "",
                },
            )

        # 显式迁移 identity 列后，修正下一个消息 ID，避免新消息主键冲突。
        connection.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('messages', 'id'),
                    COALESCE((SELECT MAX(id) FROM messages), 1),
                    (SELECT COUNT(*) > 0 FROM messages)
                )
                """
            )
        )

    processed = sum(len(rows) for rows in rows_by_table.values())
    return processed, missing_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将旧 SQLite 元数据迁移到 PostgreSQL")
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/app.db"))
    parser.add_argument("--source-data-dir", default="data")
    parser.add_argument("--target-data-dir", default=str(settings.data_dir))
    parser.add_argument("--dry-run", action="store_true", help="只读取并检查，不写入 PostgreSQL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.sqlite_path.is_file():
        raise SystemExit(f"SQLite source was not found: {args.sqlite_path}")

    rows_by_table = _read_source(args.sqlite_path)
    missing_paths: list[str] = []
    for table_name in ("documents", "upload_batch_items", "document_fingerprints"):
        for row in rows_by_table[table_name]:
            _path_and_missing(
                row.get("file_path"),
                args.source_data_dir,
                args.target_data_dir,
                missing_paths,
            )

    if args.dry_run:
        _print_dry_run(rows_by_table, missing_paths)
        return

    processed, missing_paths = migrate(
        rows_by_table,
        args.source_data_dir,
        args.target_data_dir,
    )
    print(f"Migrated or updated rows: {processed}")
    print(f"Mapped missing files: {len(missing_paths)}")
    for path in missing_paths[:20]:
        print(f"  missing: {path}")
    if len(missing_paths) > 20:
        print(f"  ... and {len(missing_paths) - 20} more")


if __name__ == "__main__":
    main()
