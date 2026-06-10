import time
import uuid
import zipfile
from pathlib import Path

import requests
from fastapi import HTTPException

from app.core.config import settings


class MinerUClient:
    """通过 MinerU 官方 API 把 PDF、Word、PPT、图片等文件转换成 Markdown。"""

    def __init__(self) -> None:
        self.base_url = settings.mineru_api_base_url.rstrip("/")
        self.token = settings.mineru_api_token

    def parse_file_to_markdown(self, file_path: Path, output_dir: Path) -> str:
        if not self.token:
            raise HTTPException(
                status_code=500,
                detail="MinerU API requires MINERU_API_TOKEN in .env.",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        # MinerU 的流程是：申请上传地址 -> 上传文件 -> 轮询解析结果 -> 下载 zip -> 读取 Markdown。
        batch_id, upload_url = self._create_upload_url(file_path.name)
        self._upload_file(upload_url, file_path)
        result = self._poll_result(batch_id)
        zip_url = self._find_zip_url(result)
        zip_path = self._download_zip(zip_url, output_dir)
        return self._extract_markdown(zip_path, output_dir)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _create_upload_url(self, filename: str) -> tuple[str, str]:
        payload = {
            "files": [{"name": filename, "data_id": str(uuid.uuid4())}],
            "model_version": settings.mineru_model_version,
            "language": settings.mineru_language,
            "enable_formula": settings.mineru_enable_formula,
            "enable_table": settings.mineru_enable_table,
            "is_ocr": settings.mineru_is_ocr,
        }
        try:
            response = requests.post(
                f"{self.base_url}/file-urls/batch",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=500, detail=f"MinerU create upload URL request failed: {exc}") from exc

        data = self._json_or_error(response, "create MinerU upload URL")
        if data.get("code") != 0:
            raise HTTPException(status_code=500, detail=f"MinerU create upload URL failed: {data}")

        result = data.get("data") or {}
        file_urls = result.get("file_urls") or result.get("urls") or []
        if not result.get("batch_id") or not file_urls:
            raise HTTPException(status_code=500, detail=f"MinerU upload URL response missing fields: {data}")

        return result["batch_id"], file_urls[0]

    def _upload_file(self, upload_url: str, pdf_path: Path) -> None:
        try:
            with pdf_path.open("rb") as f:
                response = requests.put(upload_url, data=f, timeout=120)
        except requests.RequestException as exc:
            raise HTTPException(status_code=500, detail=f"MinerU file upload request failed: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise HTTPException(
                status_code=500,
                detail=f"MinerU file upload failed: status={response.status_code}, body={response.text[:500]}",
            )

    def _poll_result(self, batch_id: str) -> dict:
        deadline = time.monotonic() + settings.mineru_timeout_seconds
        url = f"{self.base_url}/extract-results/batch/{batch_id}"

        while time.monotonic() < deadline:
            try:
                response = requests.get(url, headers=self._headers(), timeout=30)
            except requests.RequestException as exc:
                raise HTTPException(status_code=500, detail=f"MinerU poll request failed: {exc}") from exc

            data = self._json_or_error(response, "poll MinerU result")
            if data.get("code") != 0:
                raise HTTPException(status_code=500, detail=f"MinerU poll failed: {data}")

            extract_result = self._first_extract_result(data)
            state = str(
                extract_result.get("state")
                or extract_result.get("status")
                or extract_result.get("extract_state")
                or ""
            ).lower()

            if state in {"done", "success", "finished", "completed"} or self._find_zip_url(data):
                return data
            if state in {"failed", "fail", "error"}:
                raise HTTPException(status_code=500, detail=f"MinerU parsing failed: {data}")

            time.sleep(settings.mineru_poll_interval_seconds)

        raise HTTPException(status_code=500, detail="MinerU parsing timed out.")

    def _download_zip(self, zip_url: str, output_dir: Path) -> Path:
        if not zip_url:
            raise HTTPException(status_code=500, detail="MinerU result did not contain a zip download URL.")

        try:
            response = requests.get(zip_url, timeout=120)
        except requests.RequestException as exc:
            raise HTTPException(status_code=500, detail=f"MinerU result zip download request failed: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise HTTPException(
                status_code=500,
                detail=f"MinerU result zip download failed: status={response.status_code}",
            )

        zip_path = output_dir / "mineru_result.zip"
        zip_path.write_bytes(response.content)
        return zip_path

    def _extract_markdown(self, zip_path: Path, output_dir: Path) -> str:
        extract_dir = output_dir / "unzipped"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        markdown_files = sorted(extract_dir.rglob("full.md"))
        if not markdown_files:
            markdown_files = sorted(extract_dir.rglob("*.md"))
        if not markdown_files:
            raise HTTPException(status_code=500, detail="MinerU result zip did not contain markdown files.")

        return "\n\n".join(file.read_text(encoding="utf-8", errors="ignore") for file in markdown_files)

    def _find_zip_url(self, data: dict) -> str:
        extract_result = self._first_extract_result(data)
        candidates = [
            extract_result.get("full_zip_url"),
            extract_result.get("zip_url"),
            extract_result.get("result_zip_url"),
            extract_result.get("file_url"),
            extract_result.get("download_url"),
            extract_result.get("full_zip"),
            extract_result.get("extract_result", {}).get("full_zip_url")
            if isinstance(extract_result.get("extract_result"), dict)
            else None,
        ]
        for item in candidates:
            if item:
                return str(item)
        return ""

    def _first_extract_result(self, data: dict) -> dict:
        payload = data.get("data") or {}
        result = payload.get("extract_result") or payload.get("extract_results") or {}
        if isinstance(result, list):
            return result[0] if result else {}
        if isinstance(result, dict):
            return result
        return {}

    def _json_or_error(self, response: requests.Response, action: str) -> dict:
        if not 200 <= response.status_code < 300:
            raise HTTPException(
                status_code=500,
                detail=f"MinerU {action} failed: status={response.status_code}, body={response.text[:500]}",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=f"MinerU {action} returned non-JSON response.") from exc
