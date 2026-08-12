from pathlib import Path
import os
import re
import shutil
import sys

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

TEST_DATA_DIR = ROOT_DIR / "scripts" / "data"
if TEST_DATA_DIR.exists():
    shutil.rmtree(TEST_DATA_DIR)


def _test_database_url() -> str:
    """读取本地配置并切换到独立的 smoke_test 数据库。"""
    database_url = os.getenv("SMOKE_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        env_path = ROOT_DIR / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DATABASE_URL="):
                    database_url = line.split("=", 1)[1].strip().strip('"')
                    break
    database_url = database_url or (
        "postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag"
    )
    return re.sub(r"/rag(?=\?|$)", "/rag_test", database_url)

os.environ["DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["PDF_PARSER"] = "pypdf"
os.environ["QWEN_API_KEY"] = ""
# 冒烟测试固定使用独立 PostgreSQL 数据库，避免清理演示库中的真实数据。
os.environ["DATABASE_URL"] = _test_database_url()

from app.main import app
from app.core.config import settings


class FakeEmbeddings(Embeddings):
    """Smoke test 使用确定性向量，避免调用真实 Embedding 服务。"""

    def _embed(self, text: str) -> list[float]:
        return [
            float(len(text) % 101),
            float(sum(text.encode("utf-8")) % 101),
            float(text.count("#")),
            1.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


async def fake_generate_answer(question: str, context: str, history_text: str) -> str:
    return f"[offline smoke answer] {question}\n{context[:120]}"


# 在模块加载后替换运行时依赖；Chroma 和 RAG 服务会从这些模块变量读取实现。
import app.rag.vector_store as vector_store_module
import app.services.rag_service as rag_service_module

vector_store_module.get_embeddings = lambda: FakeEmbeddings()
rag_service_module.generate_answer = fake_generate_answer
settings.qwen_api_key = ""
settings.dashscope_api_key = ""
settings.openai_api_key = ""
settings.qwen_embedding_dimensions = 4
settings.deepseek_api_key = ""


class FakeAgentModel:
    """冒烟测试中的 Agent 只调用只读系统状态工具，不访问真实对话模型。"""

    async def ainvoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="[offline smoke agent answer]")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_system_status",
                    "args": {},
                    "id": "smoke-agent-call",
                    "type": "tool_call",
                }
            ],
        )


def main() -> None:
    """Run a small end-to-end check without polluting the real data directory."""
    with TestClient(app) as client:
        # lifespan 会按配置创建 Agent；这里替换为内存 checkpoint + 假模型，确保 smoke 不消耗云端 token。
        from app.agent.tools import get_system_status
        from app.services.agent_service import agent_service

        agent_service.stop()
        agent_service.configure(
            InMemorySaver(),
            model=FakeAgentModel(),
            tools=[get_system_status],
        )
        health = client.get("/api/health")
        health.raise_for_status()

        sample_path = ROOT_DIR / "sample_docs" / "company_handbook.md"
        with sample_path.open("rb") as f:
            upload = client.post(
                "/api/documents/upload",
                files={"file": (sample_path.name, f, "text/markdown")},
            )
        upload.raise_for_status()

        original_bytes = sample_path.read_bytes()
        second_bytes = original_bytes + b"\n\n## Smoke test extra\nThis is a second document payload.\n"
        batch = client.post(
            "/api/documents/upload-batch",
            files=[
                ("files", ("duplicate-handbook.md", original_bytes, "text/markdown")),
                ("files", ("second-handbook.md", second_bytes, "text/markdown")),
            ],
        )
        batch.raise_for_status()
        batch_payload = batch.json()
        assert batch_payload["succeeded"] == 1
        assert batch_payload["skipped"] == 1

        batch_detail = client.get(f"/api/documents/upload-batches/{batch_payload['batch_id']}")
        batch_detail.raise_for_status()
        batch_history = client.get("/api/documents/upload-batches?limit=20")
        batch_history.raise_for_status()

        ask = client.post(
            "/api/rag/ask",
            json={
                "question": "报销流程需要准备哪些材料？",
                "thread_id": "smoke-test",
                "top_k": 3,
            },
        )
        ask.raise_for_status()

        hybrid_ask = client.post(
            "/api/rag/ask",
            json={
                "question": "报销流程需要准备哪些材料？",
                "thread_id": "smoke-test-hybrid",
                "top_k": 3,
                "retrieval_strategy": "hybrid",
            },
        )
        hybrid_ask.raise_for_status()
        assert hybrid_ask.json()["sources"]

        reindex = client.post(f"/api/documents/{upload.json()['document_id']}/reindex")
        reindex.raise_for_status()

        deleted = client.delete(
            f"/api/documents/{upload.json()['document_id']}?delete_files=true"
        )
        deleted.raise_for_status()

        agent = client.post(
            "/api/agent/invoke",
            json={"input": "查询当前系统状态", "thread_id": "smoke-test"},
        )
        agent.raise_for_status()

        status = client.get("/api/status")
        status.raise_for_status()

        print("Smoke test passed")
        print("Upload:", upload.json())
        print("Batch upload:", batch_payload)
        print("Batch detail items:", len(batch_detail.json()["items"]))
        print("Batch history count:", len(batch_history.json()))
        print("RAG answer preview:", ask.json()["answer"][:120])
        print("Hybrid sources:", len(hybrid_ask.json()["sources"]))
        print("Reindex:", reindex.json())
        print("Delete:", deleted.json())
        print("Agent tool:", agent.json()["tool_used"])
        print("Status:", status.json())


if __name__ == "__main__":
    main()
