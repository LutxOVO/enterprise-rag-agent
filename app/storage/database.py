import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


TERMINAL_UPLOAD_ITEM_STATUSES = ("indexed", "duplicate", "failed")
PROCESSING_UPLOAD_ITEM_STATUSES = ("queued", "saving", "parsing", "splitting", "embedding")


def _connect() -> sqlite3.Connection:
    """连接本地 SQLite，并让查询结果可以像字典一样读取。"""
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.sqlite_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """初始化业务表，并兼容已经存在的旧版 app.db。"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_batches (
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
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_batch_items (
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
                updated_at TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES upload_batches(batch_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_fingerprints (
                file_hash TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_upload_batch_items_batch_id "
            "ON upload_batch_items(batch_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_upload_batches_created_at "
            "ON upload_batches(created_at DESC)"
        )
    mark_interrupted_uploads()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(created_at: str) -> int:
    try:
        started = datetime.fromisoformat(created_at)
        return max(0, int((datetime.now(timezone.utc) - started).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def save_document(
    document_id: str,
    filename: str,
    file_type: str,
    file_path: Path,
    chunk_count: int,
) -> None:
    """保存成功入库的文档信息；向量内容存在 Chroma。"""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, filename, file_type, file_path, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, filename, file_type, str(file_path), chunk_count, utc_now()),
        )


def get_document(document_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT document_id, filename, file_type, file_path, chunk_count, created_at
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_document_record(document_id: str) -> dict[str, Any] | None:
    """删除 SQLite 中的文档记录和对应指纹，返回删除前的文档信息。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.execute(
            "DELETE FROM document_fingerprints WHERE document_id = ?",
            (document_id,),
        )
    return dict(row)


def list_documents() -> list[dict]:
    """给文档列表接口和 Agent 工具使用。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT document_id, filename, file_type, chunk_count, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def count_documents() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
    return int(row["count"])


def clear_documents() -> int:
    """清空文档元数据，并返回删除数量。"""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        removed_count = int(row["count"])
        conn.execute("DELETE FROM documents")
    return removed_count


def save_message(thread_id: str, role: str, content: str) -> None:
    """保存一条对话消息，thread_id 用来区分不同会话。"""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (thread_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, role, content, utc_now()),
        )


def get_messages(thread_id: str, limit: int = 8) -> list[dict]:
    """读取最近几轮对话，返回时恢复为从旧到新的顺序。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()
    return list(reversed([dict(row) for row in rows]))


def clear_messages() -> int:
    """清空所有会话历史，并返回删除数量。"""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM messages").fetchone()
        removed_count = int(row["count"])
        conn.execute("DELETE FROM messages")
    return removed_count


def create_upload_batch(batch_id: str, total: int) -> None:
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO upload_batches (batch_id, status, total, created_at)
            VALUES (?, 'running', ?, ?)
            """,
            (batch_id, total, now),
        )


def create_upload_batch_item(
    item_id: str,
    batch_id: str,
    filename: str,
    file_type: str,
) -> None:
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO upload_batch_items
            (item_id, batch_id, filename, file_type, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (item_id, batch_id, filename, file_type, now, now),
        )


def update_upload_batch_item(item_id: str, **fields: Any) -> None:
    """更新单个文件的处理状态；字段名来自内部白名单，避免拼接任意 SQL。"""
    allowed = {
        "document_id",
        "file_hash",
        "file_path",
        "status",
        "error_stage",
        "error_message",
        "chunk_count",
        "duration_ms",
        "retry_count",
        "duplicate_of_document_id",
    }
    updates = {key: value for key, value in fields.items() if key in allowed}
    if not updates:
        return
    updates["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = [updates[key] for key in updates]
    values.append(item_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE upload_batch_items SET {assignments} WHERE item_id = ?",
            values,
        )


def get_upload_batch_item(item_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM upload_batch_items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
    return dict(row) if row else None


def refresh_upload_batch(batch_id: str, graph_path: list[str] | None = None) -> dict[str, Any]:
    """根据 item 状态重新计算批次汇总，避免手工计数产生不一致。"""
    with _connect() as conn:
        batch = conn.execute(
            "SELECT * FROM upload_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not batch:
            raise KeyError(f"Upload batch not found: {batch_id}")

        items = conn.execute(
            "SELECT status FROM upload_batch_items WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()
        statuses = [row["status"] for row in items]
        succeeded = statuses.count("indexed")
        failed = statuses.count("failed")
        skipped = statuses.count("duplicate")
        finished = len(statuses) == int(batch["total"]) and all(
            status in TERMINAL_UPLOAD_ITEM_STATUSES for status in statuses
        )

        if not finished:
            status = "running"
            completed_at = None
        elif failed == int(batch["total"]):
            status = "failed"
            completed_at = utc_now()
        elif failed and succeeded:
            status = "partial_success"
            completed_at = utc_now()
        elif failed:
            status = "failed"
            completed_at = utc_now()
        else:
            status = "completed"
            completed_at = utc_now()

        path_text = batch["graph_path"]
        if graph_path is not None:
            path_text = json.dumps(graph_path, ensure_ascii=False)

        duration_ms = _duration_ms(batch["created_at"]) if finished else 0
        conn.execute(
            """
            UPDATE upload_batches
            SET status = ?, succeeded = ?, failed = ?, skipped = ?,
                graph_path = ?, duration_ms = ?, completed_at = ?
            WHERE batch_id = ?
            """,
            (
                status,
                succeeded,
                failed,
                skipped,
                path_text,
                duration_ms,
                completed_at,
                batch_id,
            ),
        )

    result = get_upload_batch(batch_id)
    if result is None:
        raise KeyError(f"Upload batch not found after refresh: {batch_id}")
    return result


def get_upload_batch(batch_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        batch = conn.execute(
            "SELECT * FROM upload_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not batch:
            return None
        items = conn.execute(
            """
            SELECT item_id, batch_id, document_id, filename, file_type, file_hash,
                   file_path, status, error_stage, error_message, chunk_count,
                   duration_ms, retry_count, duplicate_of_document_id,
                   created_at, updated_at
            FROM upload_batch_items
            WHERE batch_id = ?
            ORDER BY created_at, item_id
            """,
            (batch_id,),
        ).fetchall()

    result = dict(batch)
    result["graph_path"] = json.loads(result.get("graph_path") or "[]")
    result["items"] = [dict(item) for item in items]
    return result


def list_upload_batches(limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT batch_id, status, total, succeeded, failed, skipped,
                   duration_ms, created_at, completed_at
            FROM upload_batches
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def reserve_document_fingerprint(
    file_hash: str,
    document_id: str,
    filename: str,
    file_path: Path,
) -> dict[str, Any]:
    """原子地登记文件指纹，保证同一内容不会被并发重复处理。"""
    now = utc_now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_fingerprints WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO document_fingerprints
                (file_hash, document_id, status, filename, file_path, created_at, updated_at)
                VALUES (?, ?, 'processing', ?, ?, ?, ?)
                """,
                (file_hash, document_id, filename, str(file_path), now, now),
            )
            return {"reserved": True, "document_id": document_id, "status": "processing"}

        if row["status"] == "failed":
            reclaimed_document_id = row["document_id"] or document_id
            conn.execute(
                """
                UPDATE document_fingerprints
                SET document_id = ?, status = 'processing', filename = ?, file_path = ?, updated_at = ?
                WHERE file_hash = ?
                """,
                (reclaimed_document_id, filename, str(file_path), now, file_hash),
            )
            return {
                "reserved": True,
                "document_id": reclaimed_document_id,
                "status": "processing",
                "reclaimed": True,
            }

        return {
            "reserved": False,
            "document_id": row["document_id"],
            "status": row["status"],
            "filename": row["filename"],
            "file_path": row["file_path"],
        }


def set_document_fingerprint_status(file_hash: str, status: str, file_path: Path | None = None) -> None:
    now = utc_now()
    with _connect() as conn:
        if file_path is None:
            conn.execute(
                "UPDATE document_fingerprints SET status = ?, updated_at = ? WHERE file_hash = ?",
                (status, now, file_hash),
            )
        else:
            conn.execute(
                """
                UPDATE document_fingerprints
                SET status = ?, file_path = ?, updated_at = ?
                WHERE file_hash = ?
                """,
                (status, str(file_path), now, file_hash),
            )


def get_document_fingerprint(document_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_fingerprints WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def upsert_document_fingerprint(
    file_hash: str,
    document_id: str,
    filename: str,
    file_path: Path,
    status: str = "processing",
) -> None:
    """为旧数据或重建索引任务补登记指纹。"""
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO document_fingerprints
            (file_hash, document_id, status, filename, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                document_id = excluded.document_id,
                status = excluded.status,
                filename = excluded.filename,
                file_path = excluded.file_path,
                updated_at = excluded.updated_at
            """,
            (file_hash, document_id, status, filename, str(file_path), now, now),
        )


def clear_upload_tracking() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM document_fingerprints")
        conn.execute("DELETE FROM upload_batch_items")
        conn.execute("DELETE FROM upload_batches")


def mark_interrupted_uploads() -> None:
    """应用重启后，把未完成的本地任务标记为可重试的失败状态。"""
    placeholders = ", ".join("?" for _ in PROCESSING_UPLOAD_ITEM_STATUSES)
    batch_ids: list[str] = []
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT batch_id FROM upload_batch_items WHERE status IN ({placeholders})",
            PROCESSING_UPLOAD_ITEM_STATUSES,
        ).fetchall()
        batch_ids = [row["batch_id"] for row in rows]
        conn.execute(
            f"""
            UPDATE upload_batch_items
            SET status = 'failed', error_stage = 'interrupted',
                error_message = 'Application restarted before this file finished.',
                updated_at = ?
            WHERE status IN ({placeholders})
            """,
            (utc_now(), *PROCESSING_UPLOAD_ITEM_STATUSES),
        )
        conn.execute(
            """
            UPDATE document_fingerprints
            SET status = 'failed', updated_at = ?
            WHERE status = 'processing'
            """,
            (utc_now(),),
        )

    for batch_id in batch_ids:
        refresh_upload_batch(batch_id)
