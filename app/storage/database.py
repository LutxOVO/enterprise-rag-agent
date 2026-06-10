import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


def _connect() -> sqlite3.Connection:
    """连接本地 SQLite，并让查询结果可以像字典一样读取。"""
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.sqlite_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """启动时初始化两张小表：文档元数据和聊天历史。"""
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_document(
    document_id: str,
    filename: str,
    file_type: str,
    file_path: Path,
    chunk_count: int,
) -> None:
    """保存文档信息；真正的向量内容存在 Chroma，不存在 SQLite。"""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, filename, file_type, file_path, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, filename, file_type, str(file_path), chunk_count, utc_now()),
        )


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
