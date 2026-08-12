import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import tools as agent_tools
from app.main import app
from app.services.agent_service import agent_service


class ApiFakeModel:
    def __init__(self, tool_name: str | None = None):
        self.tool_name = tool_name

    async def ainvoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="API 测试完成")
        if self.tool_name == "retry_upload_item":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": {"batch_id": "batch-api", "item_id": "item-api"},
                        "id": "api-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="API 普通回答")


def parse_events(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        line = next((line for line in block.splitlines() if line.startswith("data: ")), None)
        if line and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_agent_sse_contract_and_state_endpoint():
    @tool("read_api")
    async def read_api(value: str) -> dict:
        """API contract read tool."""
        return {"ok": True, "message": value}

    with TestClient(app) as client:
        agent_service.stop()
        agent_service.configure(InMemorySaver(), model=ApiFakeModel(), tools=[read_api])
        response = client.post(
            "/api/agent/runs/stream",
            json={"input": "测试普通 Agent", "thread_id": "api-thread"},
        )
        assert response.status_code == 200
        events = parse_events(response.text)
        event_names = [event["event"] for event in events]
        assert event_names[0] == "run_started"
        assert event_names[-2:] == ["answer", "done"]
        assert [event["status"] for event in events if event["event"] == "llm_stage"] == [
            "started",
            "completed",
        ]
        assert events[-1]["status"] == "completed"

        state = client.get("/api/agent/threads/api-thread/state")
        assert state.status_code == 200
        assert state.json()["last_answer"] == "API 普通回答"
        assert len(state.json()["llm_trace"]) == 1


def test_agent_approval_api_resumes_once(monkeypatch):
    executed: list[tuple[str, str]] = []

    async def fake_retry(batch_id: str, item_id: str):
        executed.append((batch_id, item_id))
        return {"batch_id": batch_id, "item_id": item_id, "status": "completed"}

    monkeypatch.setattr(agent_tools, "retry_batch_item", fake_retry)

    with TestClient(app) as client:
        agent_service.stop()
        agent_service.configure(
            InMemorySaver(),
            model=ApiFakeModel("retry_upload_item"),
            tools=[agent_tools.retry_upload_item],
        )
        first = client.post(
            "/api/agent/runs/stream",
            json={"input": "请求重试", "thread_id": "approval-api-thread"},
        )
        assert first.status_code == 200
        first_events = parse_events(first.text)
        assert first_events[-1]["status"] == "awaiting_approval"
        pending = client.get("/api/agent/threads/approval-api-thread/state").json()["pending_approval"]
        assert pending["tool_name"] == "retry_upload_item"
        assert executed == []

        resumed = client.post(
            "/api/agent/threads/approval-api-thread/resume/stream",
            json={"approval_id": pending["approval_id"], "decision": "approve"},
        )
        assert resumed.status_code == 200
        assert parse_events(resumed.text)[-1]["status"] == "completed"
        assert executed == [("batch-api", "item-api")]

        repeated = client.post(
            "/api/agent/threads/approval-api-thread/resume/stream",
            json={"approval_id": pending["approval_id"], "decision": "approve"},
        )
        assert repeated.status_code == 409


def test_agent_thread_delete_api_is_idempotent():
    with TestClient(app) as client:
        agent_service.stop()
        agent_service.configure(InMemorySaver(), model=ApiFakeModel())
        response = client.post(
            "/api/agent/runs/stream",
            json={"input": "建立删除测试会话", "thread_id": "delete-api-thread"},
        )
        assert response.status_code == 200

        deleted = client.delete("/api/agent/threads/delete-api-thread")
        assert deleted.status_code == 200
        assert deleted.json() == {
            "thread_id": "delete-api-thread",
            "deleted": True,
            "message": "会话已删除。",
        }

        repeated = client.delete("/api/agent/threads/delete-api-thread")
        assert repeated.status_code == 200
        assert repeated.json()["deleted"] is True


def test_agent_thread_delete_api_rejects_pending_approval():
    with TestClient(app) as client:
        agent_service.stop()
        agent_service.configure(
            InMemorySaver(),
            model=ApiFakeModel("retry_upload_item"),
            tools=[agent_tools.retry_upload_item],
        )
        first = client.post(
            "/api/agent/runs/stream",
            json={"input": "请求审批", "thread_id": "delete-pending-api-thread"},
        )
        assert first.status_code == 200
        deleted = client.delete("/api/agent/threads/delete-pending-api-thread")
        assert deleted.status_code == 409
        assert "审批" in deleted.json()["detail"]["message"]
