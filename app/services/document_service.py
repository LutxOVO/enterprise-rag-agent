import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.text_splitter import split_markdown_text, split_text
from app.rag.vector_store import ChromaVectorStore
from app.services.mineru_client import MinerUClient
from app.storage.database import save_document


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


class DocumentService:
    """文档入库服务：上传保存、文本提取、切分、向量化、记录元数据。"""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store or ChromaVectorStore()

    async def ingest_upload(self, file: UploadFile, upload_dir: Path) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported extensions: {supported}")

        upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = str(uuid.uuid4())
        safe_name = Path(file.filename or f"{document_id}{suffix}").name
        file_type = suffix.lstrip(".")
        saved_path = upload_dir / f"{document_id}_{safe_name}"
        content_bytes = await file.read()
        saved_path.write_bytes(content_bytes)

        # 入库主流程：提取文本 -> 切分 chunk -> 写入 Chroma -> SQLite 记录文档信息。
        text = self._extract_text(saved_path, suffix)
        chunks = self._split_extracted_text(text, suffix)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text was found in this file.")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "document_id": document_id,
                    "filename": safe_name,
                    "file_type": file_type,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        try:
            self.vector_store.add_documents(documents)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Embedding/vector store failed: {exc}") from exc

        save_document(document_id, safe_name, file_type, saved_path, len(chunks))

        return {"document_id": document_id, "filename": safe_name, "chunk_count": len(chunks)}

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
