from pathlib import Path
import os
import shutil
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

TEST_DATA_DIR = ROOT_DIR / "scripts" / "data"
if TEST_DATA_DIR.exists():
    shutil.rmtree(TEST_DATA_DIR)

os.environ["DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["CHAT_PROVIDER"] = "local"
os.environ["PDF_PARSER"] = "pypdf"

from app.main import app


def main() -> None:
    """Run a small end-to-end check without polluting the real data directory."""
    with TestClient(app) as client:
        health = client.get("/api/health")
        health.raise_for_status()

        sample_path = ROOT_DIR / "sample_docs" / "company_handbook.md"
        with sample_path.open("rb") as f:
            upload = client.post(
                "/api/documents/upload",
                files={"file": (sample_path.name, f, "text/markdown")},
            )
        upload.raise_for_status()

        ask = client.post(
            "/api/rag/ask",
            json={
                "question": "报销流程需要准备哪些材料？",
                "thread_id": "smoke-test",
                "top_k": 3,
            },
        )
        ask.raise_for_status()

        agent = client.post(
            "/api/agent/invoke",
            json={"input": "查询当前系统状态", "thread_id": "smoke-test"},
        )
        agent.raise_for_status()

        status = client.get("/api/status")
        status.raise_for_status()

        print("Smoke test passed")
        print("Upload:", upload.json())
        print("RAG answer preview:", ask.json()["answer"][:120])
        print("Agent tool:", agent.json()["tool_used"])
        print("Status:", status.json())


if __name__ == "__main__":
    main()
