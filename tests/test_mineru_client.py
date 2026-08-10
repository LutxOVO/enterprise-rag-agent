from pathlib import Path

import pytest
from fastapi import HTTPException
from pypdf import PdfReader, PdfWriter

from app.core.config import settings
from app.services.mineru_client import MinerUClient


def create_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)
    return path


def test_pdf_within_limit_is_sent_once(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "small.pdf", page_count=2)
    output_dir = tmp_path / "mineru-output"
    client = MinerUClient()
    client.token = "test-token"
    monkeypatch.setattr(settings, "mineru_max_pages_per_request", 2)

    calls: list[tuple[Path, Path]] = []

    def fake_parse(file_path: Path, part_output_dir: Path) -> str:
        calls.append((file_path, part_output_dir))
        return "small markdown"

    monkeypatch.setattr(client, "_parse_single_file", fake_parse)

    result = client.parse_file_to_markdown(pdf_path, output_dir)

    assert result == "small markdown"
    assert calls == [(pdf_path, output_dir)]


def test_large_pdf_is_split_in_order_and_temporary_parts_are_removed(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "large.pdf", page_count=5)
    output_dir = tmp_path / "mineru-output"
    client = MinerUClient()
    client.token = "test-token"
    monkeypatch.setattr(settings, "mineru_max_pages_per_request", 2)

    calls: list[dict] = []

    def fake_parse(file_path: Path, part_output_dir: Path) -> str:
        calls.append(
            {
                "file_path": file_path,
                "output_dir": part_output_dir,
                "page_count": len(PdfReader(str(file_path)).pages),
            }
        )
        return f"markdown from {part_output_dir.name}"

    monkeypatch.setattr(client, "_parse_single_file", fake_parse)

    result = client.parse_file_to_markdown(pdf_path, output_dir)

    assert result == (
        "markdown from part_001\n\n"
        "markdown from part_002\n\n"
        "markdown from part_003"
    )
    assert [call["page_count"] for call in calls] == [2, 2, 1]
    assert [call["output_dir"].name for call in calls] == ["part_001", "part_002", "part_003"]
    assert [call["file_path"].name for call in calls] == [
        "large_part_001.pdf",
        "large_part_002.pdf",
        "large_part_003.pdf",
    ]
    assert not (output_dir / "_split_pdfs").exists()
    assert all(not call["file_path"].exists() for call in calls)


def test_split_parts_are_removed_when_one_mineru_request_fails(tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path / "failed.pdf", page_count=3)
    output_dir = tmp_path / "mineru-output"
    client = MinerUClient()
    client.token = "test-token"
    monkeypatch.setattr(settings, "mineru_max_pages_per_request", 2)

    def fake_parse(file_path: Path, part_output_dir: Path) -> str:
        if part_output_dir.name == "part_002":
            raise HTTPException(status_code=500, detail="simulated MinerU failure")
        return "first part"

    monkeypatch.setattr(client, "_parse_single_file", fake_parse)

    with pytest.raises(HTTPException, match="simulated MinerU failure"):
        client.parse_file_to_markdown(pdf_path, output_dir)

    assert not (output_dir / "_split_pdfs").exists()
