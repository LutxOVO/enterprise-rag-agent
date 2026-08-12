import asyncio
import hashlib
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.text_splitter import split_markdown_text, split_text
from app.rag.vector_store import ChromaVectorStore
from app.services.mineru_client import MinerUClient
from app.storage.database import (
    get_document,
    reserve_document_fingerprint,
    save_document,
    set_document_fingerprint_status,
)


TEXT_EXTENSIONS = {".txt", ".md"}
MINERU_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".webp",
    ".gif",
    ".bmp",
    ".html",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | MINERU_EXTENSIONS
StageCallback = Callable[[str], None]

# Chroma 本地 collection 写入时使用一个进程内锁，避免并发 upsert/补偿删除相互干扰。
VECTOR_WRITE_LOCK = threading.RLock()


@dataclass
class SavedUpload:
    document_id: str
    filename: str
    file_type: str
    suffix: str
    path: Path
    file_hash: str
    size_bytes: int


class DocumentService:
    """文档入库服务：保存文件、提取文本、切分、向量化和记录元数据。"""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        # 预处理阶段只需要保存文件，不应因为还没走到 embedding 就初始化 Chroma/API。
        self.vector_store = vector_store

    def _get_vector_store(self) -> ChromaVectorStore:
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore()
        return self.vector_store

    async def save_upload_file(
        self,
        file: UploadFile,
        upload_dir: Path,
        document_id: str | None = None,
    ) -> SavedUpload:
        """分块保存上传文件，同时计算 hash，避免一次性读入整个文件。"""
        original_name = file.filename or ""
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported extensions: {supported}",
            )

        upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = document_id or uuid.uuid4().hex
        safe_name = Path(original_name).name or f"{document_id}{suffix}"
        saved_path = upload_dir / f"{document_id}_{safe_name}"
        hasher = hashlib.sha256()
        total_size = 0

        try:
            with saved_path.open("wb") as output:
                while True:
                    chunk = await file.read(settings.upload_read_chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > settings.max_upload_file_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File is too large. Maximum size is "
                                f"{settings.max_upload_file_size_mb} MB."
                            ),
                        )
                    hasher.update(chunk)
                    output.write(chunk)
        except Exception:
            saved_path.unlink(missing_ok=True)
            raise

        return SavedUpload(
            document_id=document_id,
            filename=safe_name,
            file_type=suffix.lstrip("."),
            suffix=suffix,
            path=saved_path,
            file_hash=hasher.hexdigest(),
            size_bytes=total_size,
        )

    @staticmethod
    def cleanup_saved_upload(saved_upload: SavedUpload) -> None:
        saved_upload.path.unlink(missing_ok=True)

    async def ingest_upload(self, file: UploadFile, upload_dir: Path) -> dict:
        """单文件入口，也使用指纹去重和线程隔离。"""
        saved_upload = await self.save_upload_file(file, upload_dir)
        reservation = await asyncio.to_thread(
            reserve_document_fingerprint,
            saved_upload.file_hash,
            saved_upload.document_id,
            saved_upload.filename,
            saved_upload.path,
        )
        if not reservation["reserved"] and reservation.get("status") == "processing":
            self.cleanup_saved_upload(saved_upload)
            raise HTTPException(
                status_code=409,
                detail=(
                    "The same file content is currently being processed by another task; "
                    "retry after that task finishes."
                ),
            )
        if not reservation["reserved"]:
            self.cleanup_saved_upload(saved_upload)
            existing = await asyncio.to_thread(get_document, str(reservation["document_id"])) or {}
            return {
                "document_id": reservation["document_id"],
                "filename": existing.get("filename") or reservation.get("filename") or saved_upload.filename,
                "chunk_count": int(existing.get("chunk_count") or 0),
                "status": "duplicate",
                "duplicate_of_document_id": reservation["document_id"],
            }

        try:
            result = await asyncio.to_thread(self.ingest_saved_file, saved_upload)
            await asyncio.to_thread(
                set_document_fingerprint_status,
                saved_upload.file_hash,
                "indexed",
                saved_upload.path,
            )
            return {**result, "status": "indexed"}
        except HTTPException:
            await asyncio.to_thread(
                set_document_fingerprint_status,
                saved_upload.file_hash,
                "failed",
                saved_upload.path,
            )
            await asyncio.to_thread(self._delete_vectors_safely, saved_upload.document_id)
            raise
        except Exception as exc:
            await asyncio.to_thread(
                set_document_fingerprint_status,
                saved_upload.file_hash,
                "failed",
                saved_upload.path,
            )
            await asyncio.to_thread(self._delete_vectors_safely, saved_upload.document_id)
            raise HTTPException(status_code=500, detail=f"Embedding/vector store failed: {exc}") from exc

    def ingest_saved_file(
        self,
        saved_upload: SavedUpload,
        stage_callback: StageCallback | None = None,
    ) -> dict:
        """处理已经保存到磁盘的文件；该方法可在线程中执行，便于批量 worker 复用。"""
        self._notify_stage(stage_callback, "parsing")
        text = self._extract_text(saved_upload.path, saved_upload.suffix)

        self._notify_stage(stage_callback, "splitting")
        chunks = self._split_extracted_text(text, saved_upload.suffix)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text was found in this file.")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "document_id": saved_upload.document_id,
                    "filename": saved_upload.filename,
                    "file_type": saved_upload.file_type,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

        self._notify_stage(stage_callback, "embedding")
        try:
            with VECTOR_WRITE_LOCK:
                vector_store = self._get_vector_store()
                vector_store.add_documents(documents)
                # PostgreSQL 没有和 Chroma 共用事务，成功登记失败时下面的 except 会回滚向量。
                save_document(
                    saved_upload.document_id,
                    saved_upload.filename,
                    saved_upload.file_type,
                    saved_upload.path,
                    len(chunks),
                )
        except Exception:
            self._delete_vectors_safely(saved_upload.document_id)
            raise

        self._notify_stage(stage_callback, "indexed")
        return {
            "document_id": saved_upload.document_id,
            "filename": saved_upload.filename,
            "chunk_count": len(chunks),
        }

    @staticmethod
    def _notify_stage(callback: StageCallback | None, stage: str) -> None:
        if callback:
            try:
                callback(stage)
            except Exception:
                # 状态记录是可观测性增强，不能把“已经成功写入向量库”的结果变成失败。
                pass

    def _delete_vectors_safely(self, document_id: str) -> None:
        try:
            with VECTOR_WRITE_LOCK:
                self._get_vector_store().delete_by_document_id(document_id)
        except Exception:
            # 原始异常更重要；补偿失败会保留在服务日志/批次错误中进一步排查。
            pass

    def remove_document_vectors(self, document_id: str) -> None:
        """供失败重试使用：先清理同一 document_id 的可能残留向量。"""
        self._delete_vectors_safely(document_id)

    def _extract_text(self, path: Path, suffix: str) -> str:
        """根据文件类型选择文本提取方式。"""
        if suffix in TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix in MINERU_EXTENSIONS and settings.pdf_parser.lower() == "mineru_api":
            return self._extract_with_mineru_api(path)

        if suffix == ".pdf":
            return self._extract_pdf_with_pypdf(path)

        raise HTTPException(
            status_code=400,
            detail=f"{suffix} files require PDF_PARSER=mineru_api because local text extraction is not implemented.",
        )

    def _extract_with_mineru_api(self, path: Path) -> str:
        output_dir = settings.mineru_output_dir / path.stem
        try:
            return MinerUClient().parse_file_to_markdown(path, output_dir)
        except HTTPException:
            if settings.mineru_fallback_to_pypdf and path.suffix.lower() == ".pdf":
                return self._extract_pdf_with_pypdf(path)
            raise

    def _extract_pdf_with_pypdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="Install pypdf to parse pdf files.") from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def _split_extracted_text(self, text: str, suffix: str) -> list[str]:
        """MinerU 和 md 文件会产出 Markdown，优先按标题切分。"""
        if suffix == ".md" or (suffix in MINERU_EXTENSIONS and settings.pdf_parser.lower() == "mineru_api"):
            return split_markdown_text(text)
        return split_text(text)
