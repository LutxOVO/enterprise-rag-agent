from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import settings


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ".", " ", ""],
    )


def split_text(text: str) -> list[str]:
    """普通 txt 文本：直接按长度递归切分，并保留 overlap。"""
    return _recursive_splitter().split_text(text)


def split_markdown_text(text: str) -> list[str]:
    """
    Markdown 文本：先按标题层级切，再对过长小节做二次切分。

    MinerU 输出通常是 Markdown。保留标题可以让 chunk 带上章节语义，
    检索时比纯按字数切分更容易命中正确内容。
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=True,
    )
    recursive_splitter = _recursive_splitter()
    header_documents = header_splitter.split_text(text)
    chunks: list[str] = []

    for document in header_documents:
        section_text = add_markdown_headers_back(
            content=document.page_content,
            metadata=document.metadata,
        )

        for chunk in recursive_splitter.split_text(section_text):
            cleaned = chunk.strip()
            if cleaned:
                chunks.append(cleaned)

    return chunks or split_text(text)


def add_markdown_headers_back(content: str, metadata: dict) -> str:
    """strip_headers=True 后，手动把标题加回正文，方便后续二次切分。"""
    headers = []
    for level, key in enumerate(("h1", "h2", "h3", "h4"), start=1):
        title = metadata.get(key)
        if title:
            headers.append(f"{'#' * level} {title}")

    body = content.strip()
    if not headers:
        return body
    if not body:
        return "\n".join(headers)
    return "\n".join(headers) + "\n\n" + body
